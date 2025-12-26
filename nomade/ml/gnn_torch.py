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

    class FocalLoss(nn.Module):
        """
        Focal Loss for imbalanced classification.
        Down-weights easy examples, focuses on hard ones.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        """
        
        def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction
        
        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            
            if self.reduction == "mean":
                return focal_loss.mean()
            elif self.reduction == "sum":
                return focal_loss.sum()
            return focal_loss


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
            
            ConvClass = {
                'gcn': GCNConv,
                'sage': SAGEConv,
                'gat': GATConv
            }.get(conv_type.lower(), SAGEConv)
            
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            
            self.convs.append(ConvClass(input_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
            for _ in range(n_layers - 1):
                self.convs.append(ConvClass(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            
            self.classifier = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x, edge_index):
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            return self.classifier(x)
        
        def predict(self, x, edge_index):
            self.eval()
            with torch.no_grad():
                logits = self.forward(x, edge_index)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
            return preds, probs


    class GNNTrainer:
        """Trainer for PyTorch GNN with Focal Loss."""
        
        def __init__(self, model: FailureGNN, lr: float = 0.01, 
                     weight_decay: float = 5e-4, device: str = 'auto',
                     gamma: float = 2.0):
            self.model = model
            self.gamma = gamma
            
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
            self.model.train()
            self.optimizer.zero_grad()
            
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index)
            
            weights = data.class_weights.to(self.device) if hasattr(data, 'class_weights') else None
            focal = FocalLoss(alpha=weights, gamma=self.gamma)
            
            if mask is not None:
                loss = focal(out[mask], data.y[mask])
            else:
                loss = focal(out, data.y)
            
            loss.backward()
            self.optimizer.step()
            
            return {'loss': loss.item()}
        
        @torch.no_grad()
        def evaluate(self, data: Data, mask=None) -> dict:
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


    def prepare_pyg_data(jobs: list, edges: list, feature_names: list = None) -> Data:
        """Convert job data to PyTorch Geometric format."""
        if feature_names is None:
            feature_names = [
                'req_gpus', 'req_time_seconds',
                'runtime_seconds', 'req_mem_mb', 'req_cpus', 'wait_time_seconds'
            ]
        
        available = [f for f in feature_names if any(j.get(f) is not None for j in jobs)]
        if not available:
            available = ['runtime_sec']
        
        features = []
        for job in jobs:
            feat = [float(job.get(f, 0) or 0) for f in available]
            features.append(feat)
        
        x = torch.tensor(features, dtype=torch.float)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
        
        labels = [job.get('failure_reason', 0) for job in jobs]
        y = torch.tensor(labels, dtype=torch.long)
        
        edge_src = [e['source'] for e in edges]
        edge_dst = [e['target'] for e in edges]
        edge_index = torch.tensor([
            edge_src + edge_dst,
            edge_dst + edge_src
        ], dtype=torch.long)
        
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
        
        # Class weights (sqrt dampened for balance)
        class_counts = torch.bincount(y, minlength=8).float().clamp(min=1)
        class_weights = 1.0 / (class_counts ** 0.6)  # Balanced weighting
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        
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


    def train_failure_gnn(jobs: list, edges: list, epochs: int = 100,
                          hidden_dim: int = 64, conv_type: str = 'sage',
                          gamma: float = 2.0, verbose: bool = True) -> dict:
        """Train GNN on job failure data."""
        data = prepare_pyg_data(jobs, edges)
        
        if verbose:
            print(f"Data: {data.x.size(0)} nodes, {data.edge_index.size(1)//2} edges")
            print(f"Features: {data.feature_names}")
            print(f"Classes: {data.y.unique().tolist()}")
            print(f"Train/Val/Test: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")
            print()
        
        model = FailureGNN(
            input_dim=data.x.size(1),
            hidden_dim=hidden_dim,
            output_dim=8,
            conv_type=conv_type
        )
        
        trainer = GNNTrainer(model, lr=0.01, gamma=gamma)
        history = trainer.train(
            data, 
            epochs=epochs,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            verbose=verbose
        )
        
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
    class FocalLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required")
    
    class FailureGNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch Geometric required")
    
    class GNNTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch Geometric required")
    
    def train_failure_gnn(*args, **kwargs):
        raise ImportError("PyTorch Geometric required")
    
    def prepare_pyg_data(*args, **kwargs):
        raise ImportError("PyTorch Geometric required")


def is_torch_available() -> bool:
    return HAS_TORCH


if __name__ == '__main__':
    if not HAS_TORCH:
        print("PyTorch not available")
    else:
        print("Testing PyTorch GNN with Focal Loss...")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        n_nodes, n_edges, n_features = 500, 2000, 7
        
        x = torch.randn(n_nodes, n_features)
        y = torch.randint(0, 8, (n_nodes,))
        edge_index = torch.randint(0, n_nodes, (2, n_edges * 2))
        
        data = Data(x=x, edge_index=edge_index, y=y)
        
        perm = torch.randperm(n_nodes)
        data.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        data.train_mask[perm[:350]] = True
        data.val_mask[perm[350:425]] = True
        
        class_counts = torch.bincount(y, minlength=8).float().clamp(min=1)
        data.class_weights = 1.0 / (class_counts ** 0.6)  # Balanced weighting
        
        model = FailureGNN(input_dim=n_features, hidden_dim=32, conv_type='sage')
        trainer = GNNTrainer(model, gamma=2.0)
        trainer.train(data, epochs=50, train_mask=data.train_mask, 
                     val_mask=data.val_mask, verbose=True)
