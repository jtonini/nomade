"""
Graph Neural Network for failure prediction.

Uses job similarity network to predict failure types.
Architecture: Message passing on Simpson similarity network.
"""

import math
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class GNNConfig:
    """Configuration for GNN model."""
    input_dim: int = 8          # Number of job features
    hidden_dim: int = 32        # Hidden layer size
    output_dim: int = 8         # Number of failure classes (0-7)
    n_layers: int = 2           # Number of message passing layers
    dropout: float = 0.1        # Dropout rate
    learning_rate: float = 0.01
    

class SimpleGNN:
    """
    Pure Python GNN implementation.
    No PyTorch dependency - works anywhere.
    
    Architecture:
        1. Node features → Linear → Hidden
        2. Message passing (aggregate neighbor features)
        3. Hidden → Linear → Output (failure class logits)
    """
    
    def __init__(self, config: GNNConfig):
        self.config = config
        self.weights = {}
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        cfg = self.config
        
        # Input projection
        self.weights['W_in'] = self._xavier_init(cfg.input_dim, cfg.hidden_dim)
        self.weights['b_in'] = [0.0] * cfg.hidden_dim
        
        # Message passing layers
        for i in range(cfg.n_layers):
            # Aggregate weight (combines self + neighbors)
            self.weights[f'W_msg_{i}'] = self._xavier_init(cfg.hidden_dim * 2, cfg.hidden_dim)
            self.weights[f'b_msg_{i}'] = [0.0] * cfg.hidden_dim
        
        # Output projection
        self.weights['W_out'] = self._xavier_init(cfg.hidden_dim, cfg.output_dim)
        self.weights['b_out'] = [0.0] * cfg.output_dim
        
    def _xavier_init(self, fan_in: int, fan_out: int) -> list:
        """Xavier/Glorot initialization."""
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return [[random.uniform(-limit, limit) for _ in range(fan_out)] 
                for _ in range(fan_in)]
    
    def _relu(self, x: list) -> list:
        """ReLU activation."""
        return [max(0, v) for v in x]
    
    def _softmax(self, x: list) -> list:
        """Softmax activation."""
        max_x = max(x)
        exp_x = [math.exp(v - max_x) for v in x]
        sum_exp = sum(exp_x)
        return [v / sum_exp for v in exp_x]
    
    def _matmul(self, x: list, W: list) -> list:
        """Matrix multiply: x (1 x in_dim) @ W (in_dim x out_dim) -> (1 x out_dim)."""
        out_dim = len(W[0])
        result = [0.0] * out_dim
        for j in range(out_dim):
            for i, xi in enumerate(x):
                result[j] += xi * W[i][j]
        return result
    
    def _add(self, x: list, b: list) -> list:
        """Element-wise addition."""
        return [xi + bi for xi, bi in zip(x, b)]
    
    def forward(self, node_features: list, adjacency: dict) -> list:
        """
        Forward pass through GNN.
        
        Args:
            node_features: List of feature vectors, one per node
            adjacency: Dict mapping node_idx -> list of neighbor indices
            
        Returns:
            List of output logits, one per node
        """
        n_nodes = len(node_features)
        
        # Input projection
        hidden = []
        for feat in node_features:
            h = self._matmul(feat, self.weights['W_in'])
            h = self._add(h, self.weights['b_in'])
            h = self._relu(h)
            hidden.append(h)
        
        # Message passing layers
        for layer in range(self.config.n_layers):
            new_hidden = []
            W = self.weights[f'W_msg_{layer}']
            b = self.weights[f'b_msg_{layer}']
            
            for i in range(n_nodes):
                # Aggregate neighbor features (mean)
                neighbors = adjacency.get(i, [])
                if neighbors:
                    neighbor_feats = [hidden[j] for j in neighbors if j < len(hidden)]
                    if neighbor_feats:
                        agg = [sum(nf[k] for nf in neighbor_feats) / len(neighbor_feats) 
                               for k in range(len(hidden[0]))]
                    else:
                        agg = [0.0] * len(hidden[0])
                else:
                    agg = [0.0] * len(hidden[0])
                
                # Concatenate self + aggregated neighbors
                combined = hidden[i] + agg
                
                # Transform
                h = self._matmul(combined, W)
                h = self._add(h, b)
                h = self._relu(h)
                new_hidden.append(h)
            
            hidden = new_hidden
        
        # Output projection
        outputs = []
        for h in hidden:
            logits = self._matmul(h, self.weights['W_out'])
            logits = self._add(logits, self.weights['b_out'])
            outputs.append(logits)
        
        return outputs
    
    def predict(self, node_features: list, adjacency: dict) -> list:
        """Predict failure class for each node."""
        logits = self.forward(node_features, adjacency)
        predictions = []
        for l in logits:
            probs = self._softmax(l)
            pred_class = probs.index(max(probs))
            predictions.append({
                'class': pred_class,
                'confidence': max(probs),
                'probs': probs
            })
        return predictions


def prepare_job_features(jobs: list, feature_names: list = None) -> tuple:
    """
    Prepare job data for GNN.
    
    Args:
        jobs: List of job dicts from database
        feature_names: Features to use (default: I/O and resource features)
        
    Returns:
        (node_features, labels, feature_names)
    """
    if feature_names is None:
        feature_names = [
            'nfs_write_gb', 'local_write_gb', 'io_wait_pct',
            'runtime_sec', 'req_mem_mb', 'req_cpus', 'wait_time_sec'
        ]
    
    # Extract features and normalize
    raw_features = []
    labels = []
    
    for job in jobs:
        feat = []
        for f in feature_names:
            val = job.get(f, 0) or 0
            feat.append(float(val))
        raw_features.append(feat)
        labels.append(job.get('failure_reason', 0))
    
    # Normalize features (z-score)
    n_features = len(feature_names)
    means = [0.0] * n_features
    stds = [1.0] * n_features
    
    for i in range(n_features):
        vals = [rf[i] for rf in raw_features]
        means[i] = sum(vals) / len(vals) if vals else 0
        variance = sum((v - means[i])**2 for v in vals) / len(vals) if vals else 1
        stds[i] = math.sqrt(variance) if variance > 0 else 1
    
    normalized = []
    for rf in raw_features:
        normalized.append([(rf[i] - means[i]) / stds[i] for i in range(n_features)])
    
    return normalized, labels, feature_names


def build_adjacency_from_edges(edges: list, n_nodes: int) -> dict:
    """Convert edge list to adjacency dict."""
    adj = {i: [] for i in range(n_nodes)}
    for edge in edges:
        src, tgt = edge['source'], edge['target']
        if src < n_nodes and tgt < n_nodes:
            adj[src].append(tgt)
            adj[tgt].append(src)
    return adj


def evaluate_gnn(model: SimpleGNN, features: list, adjacency: dict, 
                 labels: list) -> dict:
    """
    Evaluate GNN predictions.
    
    Returns:
        Dict with accuracy, per-class metrics, confusion matrix
    """
    predictions = model.predict(features, adjacency)
    
    correct = 0
    class_correct = {}
    class_total = {}
    confusion = {}  # (true, pred) -> count
    
    for pred, true_label in zip(predictions, labels):
        pred_class = pred['class']
        
        if pred_class == true_label:
            correct += 1
            class_correct[true_label] = class_correct.get(true_label, 0) + 1
        
        class_total[true_label] = class_total.get(true_label, 0) + 1
        
        key = (true_label, pred_class)
        confusion[key] = confusion.get(key, 0) + 1
    
    accuracy = correct / len(labels) if labels else 0
    
    per_class = {}
    for c in class_total:
        per_class[c] = {
            'accuracy': class_correct.get(c, 0) / class_total[c],
            'count': class_total[c]
        }
    
    return {
        'accuracy': accuracy,
        'per_class': per_class,
        'confusion': confusion,
        'n_samples': len(labels)
    }


# Failure class names for display
FAILURE_NAMES = {
    0: 'SUCCESS',
    1: 'TIMEOUT',
    2: 'CANCELLED',
    3: 'FAILED',
    4: 'OOM',
    5: 'SEGFAULT',
    6: 'NODE_FAIL',
    7: 'DEPENDENCY'
}


if __name__ == '__main__':
    # Quick test with random data
    print("Testing SimpleGNN...")
    
    config = GNNConfig(input_dim=7, hidden_dim=16, output_dim=8)
    model = SimpleGNN(config)
    
    # Generate random test data
    n_nodes = 100
    features = [[random.random() for _ in range(7)] for _ in range(n_nodes)]
    
    # Random adjacency (sparse)
    adj = {i: random.sample(range(n_nodes), min(5, n_nodes)) for i in range(n_nodes)}
    
    # Random labels
    labels = [random.randint(0, 7) for _ in range(n_nodes)]
    
    # Evaluate
    results = evaluate_gnn(model, features, adj, labels)
    
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Samples: {results['n_samples']}")
    print("Per-class accuracy:")
    for c, stats in sorted(results['per_class'].items()):
        print(f"  {FAILURE_NAMES.get(c, c)}: {stats['accuracy']:.2%} (n={stats['count']})")


class GNNTrainer:
    """Simple trainer for GNN using numerical gradients."""
    
    def __init__(self, model: SimpleGNN, learning_rate: float = 0.01):
        self.model = model
        self.lr = learning_rate
        
    def _cross_entropy_loss(self, logits: list, label: int) -> float:
        """Compute cross-entropy loss for single sample."""
        probs = self.model._softmax(logits)
        # Clip for numerical stability
        prob = max(probs[label], 1e-10)
        return -math.log(prob)
    
    def compute_loss(self, features: list, adjacency: dict, labels: list) -> float:
        """Compute average loss over all samples."""
        outputs = self.model.forward(features, adjacency)
        total_loss = 0
        for logits, label in zip(outputs, labels):
            total_loss += self._cross_entropy_loss(logits, label)
        return total_loss / len(labels)
    
    def train_epoch(self, features: list, adjacency: dict, labels: list,
                    eps: float = 1e-4) -> float:
        """
        One epoch of training using numerical gradients.
        
        This is slow but works without autograd.
        """
        base_loss = self.compute_loss(features, adjacency, labels)
        
        # Update each weight using numerical gradient
        for name, W in self.model.weights.items():
            if isinstance(W[0], list):  # Matrix
                for i in range(len(W)):
                    for j in range(len(W[0])):
                        # Compute gradient numerically
                        original = W[i][j]
                        W[i][j] = original + eps
                        loss_plus = self.compute_loss(features, adjacency, labels)
                        W[i][j] = original
                        
                        grad = (loss_plus - base_loss) / eps
                        W[i][j] = original - self.lr * grad
            else:  # Bias vector
                for i in range(len(W)):
                    original = W[i]
                    W[i] = original + eps
                    loss_plus = self.compute_loss(features, adjacency, labels)
                    W[i] = original
                    
                    grad = (loss_plus - base_loss) / eps
                    W[i] = original - self.lr * grad
        
        return base_loss


def train_on_jobs(jobs: list, edges: list, epochs: int = 10, 
                  sample_size: int = 100) -> dict:
    """
    Train GNN on job data.
    
    Args:
        jobs: List of job dicts
        edges: Similarity edges
        epochs: Training epochs
        sample_size: Subsample for faster training
        
    Returns:
        Dict with model and training history
    """
    # Prepare data
    features, labels, feat_names = prepare_job_features(jobs)
    adjacency = build_adjacency_from_edges(edges, len(jobs))
    
    # Subsample for speed (numerical gradients are slow)
    if len(jobs) > sample_size:
        indices = random.sample(range(len(jobs)), sample_size)
        features = [features[i] for i in indices]
        labels = [labels[i] for i in indices]
        # Remap adjacency
        idx_map = {old: new for new, old in enumerate(indices)}
        adjacency = {}
        for new_idx, old_idx in enumerate(indices):
            neighbors = []
            for n in build_adjacency_from_edges(edges, len(jobs)).get(old_idx, []):
                if n in idx_map:
                    neighbors.append(idx_map[n])
            adjacency[new_idx] = neighbors
    
    # Create model and trainer
    config = GNNConfig(input_dim=len(feat_names), hidden_dim=16, output_dim=8)
    model = SimpleGNN(config)
    trainer = GNNTrainer(model, learning_rate=0.1)
    
    # Training loop
    history = []
    for epoch in range(epochs):
        loss = trainer.train_epoch(features, adjacency, labels)
        results = evaluate_gnn(model, features, adjacency, labels)
        history.append({
            'epoch': epoch,
            'loss': loss,
            'accuracy': results['accuracy']
        })
        print(f"Epoch {epoch+1}/{epochs}: loss={loss:.4f}, acc={results['accuracy']:.2%}")
    
    return {
        'model': model,
        'history': history,
        'final_accuracy': history[-1]['accuracy'],
        'feature_names': feat_names
    }
