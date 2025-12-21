"""
NØMADE Machine Learning Module

Provides GNN-based failure prediction:
- Pure Python implementation (no dependencies)
- PyTorch Geometric implementation (optional, for GPU training)
"""

from .gnn import (
    SimpleGNN,
    GNNConfig,
    prepare_job_features,
    build_adjacency_from_edges,
    evaluate_gnn,
    FAILURE_NAMES
)

# Try to import PyTorch version
try:
    from .gnn_torch import (
        is_torch_available,
        FailureGNN,
        GNNTrainer,
        train_failure_gnn,
        prepare_pyg_data
    )
except ImportError:
    is_torch_available = lambda: False

__all__ = [
    # Pure Python
    'SimpleGNN',
    'GNNConfig', 
    'prepare_job_features',
    'build_adjacency_from_edges',
    'evaluate_gnn',
    'FAILURE_NAMES',
    # PyTorch (optional)
    'is_torch_available',
    'FailureGNN',
    'GNNTrainer', 
    'train_failure_gnn',
    'prepare_pyg_data'
]
