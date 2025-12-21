"""
PyTorch Geometric GNN for failure prediction.

Optional dependency - falls back to pure Python if not available.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    from torch_geometric.data import Data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch Geometric not available. Install with:")
    print("  pip install torch torch-geometric")


if HAS_TORCH:
    
    class FailureGNN(nn.Module):
        """
        Graph Neural Network for job failure prediction.
        
        Architecture options:
            - GCN: Graph Convolutional Network
            - SAGE: GraphSAGE (sampling-based)
            - GAT: Graph Attention Network
        """
        
        def __init__(self, input_dim: int, hidden_dim: int = 64, 
                     output_dim: int = 8, n_layers: int = 2,
                     dropout: float = 0.3, conv_type: str = 'sage'):
            super().__init__()
            
            self.dropout = dropout
            self.n_layers = n_layers
            
            # Select convolution type
            ConvClass = {
                'gcn': GCNConv,
                'sage': SAGEConv,
                'gat': GATConv
            }.get(conv_type.lower(), SAGEConv)
            
            # Build layers
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            
            # Input layer
            self.convs.append(ConvClass(input_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
            # Hidden layers
            for _ in range(n_layers - 1):
                self.convs.append(ConvClass(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            
            # Output layer
            self.classifier = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x, edge_index):
            """
            Forward pass.
            
            Args:
                x: Node features [n_nodes, input_dim]
                edge_index: Edge indices [2, n_edges]
                
            Returns:
                Logits [n_nodes, output_dim]
            """
            for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            return self.classifier(x)
        
        def predict(self, x, edge_index):
            """Get predictions with probabilities."""
            self.eval()
            with torch.no_grad():
                logits = self.forward(x, edge_index)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
            return preds, probs


    class GNNTrainer:
        """Trainer for PyTorch GNN."""
        
        def __init__(self, model: FailureGNN, lr: float = 0.01, 
                     weight_decay: float = 5e-4, device: str = 'auto'):
            self.model = model
            
            # Select device
            if device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
            
            self.model.to(self.device)
            
            self.optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
            
            self.history = []
            
        def train_epoch(self, data: Data, mask=None) -> dict:
            """Train for one epoch."""
            self.model.train()
            self.optimizer.zero_grad()
            
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index)
            
            
            # Get class weights if available
            weights = data.class_weights.to(self.device) if hasattr(data, "class_weights") else None
            
            if mask is not None:
                loss = F.cross_entropy(out[mask], data.y[mask], weight=weights)
            else:
                loss = F.cross_entropy(out, data.y, weight=weights)
            
            loss.backward()
            self.optimizer.step()
            
            return {'loss': loss.item()}
        
        @torch.no_grad()
        def evaluate(self, data: Data, mask=None) -> dict:
            """Evaluate model."""
            self.model.eval()
            data = data.to(self.device)
            
            out = self.model(data.x, data.edge_index)
            
            if mask is not None:
                pred = out[mask].argmax(dim=1)
                correct = (pred == data.y[mask]).sum().item()
                total = mask.sum().item()
            else:
                pred = out.argmax(dim=1)
                correct = (pred == data.y).sum().item()
                total = data.y.size(0)
            
            acc = correct / total if total > 0 else 0
            
            # Per-class accuracy
            per_class = {}
            for c in range(out.size(1)):
                if mask is not None:
                    c_mask = (data.y[mask] == c)
                    c_pred = pred[c_mask]
                    c_total = c_mask.sum().item()
                else:
                    c_mask = (data.y == c)
                    c_pred = pred[c_mask]
                    c_total = c_mask.sum().item()
                
                if c_total > 0:
                    c_correct = (c_pred == c).sum().item()
                    per_class[c] = {'accuracy': c_correct / c_total, 'count': c_total}
            
            return {
                'accuracy': acc,
                'per_class': per_class,
                'n_samples': total
            }
        
        def train(self, data: Data, epochs: int = 100, 
                  train_mask=None, val_mask=None, 
                  verbose: bool = True) -> list:
            """
            Full training loop.
            
            Args:
                data: PyG Data object
                epochs: Number of epochs
                train_mask: Mask for training nodes
                val_mask: Mask for validation nodes
                verbose: Print progress
                
            Returns:
                Training history
            """
            best_val_acc = 0
            
            for epoch in range(epochs):
                train_result = self.train_epoch(data, train_mask)
                
                train_eval = self.evaluate(data, train_mask)
                val_eval = self.evaluate(data, val_mask) if val_mask is not None else None
                
                record = {
                    'epoch': epoch,
                    'loss': train_result['loss'],
                    'train_acc': train_eval['accuracy'],
                }
                
                if val_eval:
                    record['val_acc'] = val_eval['accuracy']
                    if val_eval['accuracy'] > best_val_acc:
                        best_val_acc = val_eval['accuracy']
                        record['best'] = True
                
                self.history.append(record)
                
                if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                    msg = f"Epoch {epoch+1:3d}: loss={train_result['loss']:.4f}, train_acc={train_eval['accuracy']:.2%}"
                    if val_eval:
                        msg += f", val_acc={val_eval['accuracy']:.2%}"
                    print(msg)
            
            return self.history


    def prepare_pyg_data(jobs: list, edges: list, 
                         feature_names: list = None) -> Data:
        """
        Convert job data to PyTorch Geometric format.
        
        Args:
            jobs: List of job dicts
            edges: List of edge dicts with 'source', 'target'
            feature_names: Features to use
            
        Returns:
            PyG Data object
        """
        if feature_names is None:
            feature_names = [
                'nfs_write_gb', 'local_write_gb', 'io_wait_pct',
                'runtime_sec', 'req_mem_mb', 'req_cpus', 'wait_time_sec'
            ]
        
        # Available features (some may be missing)
        available = []
        for f in feature_names:
            if any(j.get(f) is not None for j in jobs):
                available.append(f)
        
        if not available:
            available = ['runtime_sec']  # Fallback
        
        # Extract features
        features = []
        for job in jobs:
            feat = []
            for f in available:
                val = job.get(f, 0) or 0
                feat.append(float(val))
            features.append(feat)
        
        # Convert to tensor and normalize
        x = torch.tensor(features, dtype=torch.float)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
        
        # Labels
        labels = [job.get('failure_reason', 0) for job in jobs]
        y = torch.tensor(labels, dtype=torch.long)
        
        # Edge index
        edge_src = [e['source'] for e in edges]
        edge_dst = [e['target'] for e in edges]
        # Make bidirectional
        edge_index = torch.tensor([
            edge_src + edge_dst,
            edge_dst + edge_src
        ], dtype=torch.long)
        
        # Create train/val/test masks (70/15/15 split)
        n = len(jobs)
        perm = torch.randperm(n)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        
        train_mask[perm[:train_size]] = True
        val_mask[perm[train_size:train_size+val_size]] = True
        test_mask[perm[train_size+val_size:]] = True
        
        # Compute class weights for imbalanced data
        class_counts = torch.bincount(y, minlength=8).float()
        class_counts = class_counts.clamp(min=1)  # Avoid division by zero
        class_weights = 1.0 / torch.sqrt(class_counts)  # Dampened weights
        class_weights = class_weights / class_weights.sum() * len(class_counts)  # Normalize
        
        return Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            feature_names=available,
            class_weights=class_weights
        )


    def train_failure_gnn(jobs: list, edges: list, 
                          epochs: int = 100,
                          hidden_dim: int = 64,
                          conv_type: str = 'sage',
                          verbose: bool = True) -> dict:
        """
        Train GNN on job failure data.
        
        Args:
            jobs: List of job dicts
            edges: Similarity edges
            epochs: Training epochs
            hidden_dim: Hidden dimension
            conv_type: 'gcn', 'sage', or 'gat'
            verbose: Print progress
            
        Returns:
            Dict with model, history, and metrics
        """
        # Prepare data
        data = prepare_pyg_data(jobs, edges)
        
        if verbose:
            print(f"Data: {data.x.size(0)} nodes, {data.edge_index.size(1)//2} edges")
            print(f"Features: {data.feature_names}")
            print(f"Classes: {data.y.unique().tolist()}")
            print(f"Train/Val/Test: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")
            print()
        
        # Create model
        model = FailureGNN(
            input_dim=data.x.size(1),
            hidden_dim=hidden_dim,
            output_dim=8,  # 8 failure classes
            conv_type=conv_type
        )
        
        # Train
        trainer = GNNTrainer(model, lr=0.01)
        history = trainer.train(
            data, 
            epochs=epochs,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            verbose=verbose
        )
        
        # Final evaluation on test set
        test_results = trainer.evaluate(data, data.test_mask)
        
        if verbose:
            print(f"\nTest accuracy: {test_results['accuracy']:.2%}")
            print("Per-class:")
            from nomade.ml.gnn import FAILURE_NAMES
            for c, stats in sorted(test_results['per_class'].items()):
                name = FAILURE_NAMES.get(c, f'Class {c}')
                print(f"  {name}: {stats['accuracy']:.2%} (n={stats['count']})")
        
        return {
            'model': model,
            'trainer': trainer,
            'data': data,
            'history': history,
            'test_results': test_results
        }


else:
    # Fallback when PyTorch not available
    class FailureGNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch Geometric required. Use pure Python GNN instead.")
    
    class GNNTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch Geometric required. Use pure Python GNN instead.")
    
    def train_failure_gnn(*args, **kwargs):
        raise ImportError("PyTorch Geometric required. Use pure Python GNN instead.")
    
    def prepare_pyg_data(*args, **kwargs):
        raise ImportError("PyTorch Geometric required. Use pure Python GNN instead.")


def is_torch_available() -> bool:
    """Check if PyTorch Geometric is available."""
    return HAS_TORCH


if __name__ == '__main__':
    if not HAS_TORCH:
        print("PyTorch not available. Skipping test.")
    else:
        print("Testing PyTorch GNN...")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # Generate test data
        n_nodes = 500
        n_edges = 2000
        n_features = 7
        
        x = torch.randn(n_nodes, n_features)
        y = torch.randint(0, 8, (n_nodes,))
        edge_index = torch.randint(0, n_nodes, (2, n_edges * 2))
        
        data = Data(x=x, edge_index=edge_index, y=y)
        
        # Random masks
        perm = torch.randperm(n_nodes)
        data.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        data.train_mask[perm[:350]] = True
        data.val_mask[perm[350:425]] = True
        
        # Train
        model = FailureGNN(input_dim=n_features, hidden_dim=32, conv_type='sage')
        trainer = GNNTrainer(model)
        trainer.train(data, epochs=50, train_mask=data.train_mask, 
                     val_mask=data.val_mask, verbose=True)
