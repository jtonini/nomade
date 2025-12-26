"""
Ensemble model combining GNN, LSTM, and Autoencoder.

- GNN: Network structure (what fails)
- LSTM: Temporal patterns (when it fails)
- Autoencoder: Anomaly detection (is this normal)

Combined prediction provides higher confidence alerts.
"""

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .gnn import FAILURE_NAMES


if HAS_TORCH:
    
    class FailureEnsemble:
        """
        Ensemble of GNN, LSTM, and Autoencoder for failure prediction.
        
        Combines predictions using weighted voting or stacking.
        """
        
        def __init__(self, gnn_model=None, lstm_model=None, autoencoder=None,
                     weights: dict = None):
            """
            Args:
                gnn_model: Trained FailureGNN
                lstm_model: Trained FailureLSTM
                autoencoder: Trained JobAutoencoder
                weights: Dict of model weights {'gnn': 0.4, 'lstm': 0.3, 'ae': 0.3}
            """
            self.gnn = gnn_model
            self.lstm = lstm_model
            self.ae = autoencoder
            
            self.weights = weights or {'gnn': 0.5, 'lstm': 0.3, 'ae': 0.2}
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Move models to device
            if self.gnn:
                self.gnn.to(self.device)
            if self.lstm:
                self.lstm.to(self.device)
            if self.ae:
                self.ae.to(self.device)
        
        def predict_gnn(self, x, edge_index):
            """GNN prediction: class probabilities."""
            if self.gnn is None:
                return None
            self.gnn.eval()
            with torch.no_grad():
                x = x.to(self.device)
                edge_index = edge_index.to(self.device)
                logits = self.gnn(x, edge_index)
                return F.softmax(logits, dim=1)
        
        def predict_lstm(self, trajectories):
            """LSTM prediction: class probabilities."""
            if self.lstm is None:
                return None
            self.lstm.eval()
            with torch.no_grad():
                x = trajectories.to(self.device)
                logits = self.lstm(x)
                return F.softmax(logits, dim=1)
        
        def predict_anomaly(self, features):
            """Autoencoder: anomaly scores (higher = more anomalous)."""
            if self.ae is None:
                return None
            self.ae.eval()
            with torch.no_grad():
                x = features.to(self.device)
                errors = self.ae.reconstruction_error(x)
                # Normalize to 0-1 range
                errors = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
                return errors
        
        def predict(self, gnn_data=None, lstm_data=None, ae_data=None,
                    return_components: bool = False) -> dict:
            """
            Combined ensemble prediction.
            
            Args:
                gnn_data: Tuple of (x, edge_index) for GNN
                lstm_data: Tensor of trajectories for LSTM
                ae_data: Tensor of features for Autoencoder
                return_components: If True, return individual model predictions
                
            Returns:
                Dict with predictions, confidences, and optionally components
            """
            n_samples = None
            components = {}
            
            # GNN predictions
            gnn_probs = None
            if gnn_data and self.gnn:
                x, edge_index = gnn_data
                gnn_probs = self.predict_gnn(x, edge_index)
                n_samples = gnn_probs.size(0)
                components['gnn'] = gnn_probs
            
            # LSTM predictions
            lstm_probs = None
            if lstm_data is not None and self.lstm:
                lstm_probs = self.predict_lstm(lstm_data)
                n_samples = n_samples or lstm_probs.size(0)
                components['lstm'] = lstm_probs
            
            # Autoencoder anomaly scores
            ae_scores = None
            if ae_data is not None and self.ae:
                ae_scores = self.predict_anomaly(ae_data)
                n_samples = n_samples or ae_scores.size(0)
                components['ae_anomaly'] = ae_scores
            
            if n_samples is None:
                return {'error': 'No valid predictions'}
            
            # Combine predictions
            combined_probs = torch.zeros(n_samples, 8, device=self.device)
            total_weight = 0
            
            if gnn_probs is not None:
                combined_probs += self.weights['gnn'] * gnn_probs
                total_weight += self.weights['gnn']
            
            if lstm_probs is not None:
                combined_probs += self.weights['lstm'] * lstm_probs
                total_weight += self.weights['lstm']
            
            if ae_scores is not None:
                # Convert anomaly score to failure probability boost
                # High anomaly = boost failure classes, reduce SUCCESS
                ae_boost = ae_scores.unsqueeze(1).expand(-1, 8)
                # Reduce success probability, boost failure probabilities
                ae_adjustment = torch.zeros_like(combined_probs)
                ae_adjustment[:, 0] = -ae_scores  # Reduce SUCCESS
                ae_adjustment[:, 1:] = ae_scores.unsqueeze(1).expand(-1, 7) / 7  # Boost failures
                combined_probs += self.weights['ae'] * ae_adjustment
                total_weight += self.weights['ae']
            
            # Normalize
            if total_weight > 0:
                combined_probs = combined_probs / total_weight
            
            # Ensure valid probabilities
            combined_probs = F.softmax(combined_probs, dim=1)
            
            # Get predictions
            pred_classes = combined_probs.argmax(dim=1)
            confidences = combined_probs.max(dim=1).values
            
            result = {
                'predictions': pred_classes.cpu().tolist(),
                'confidences': confidences.cpu().tolist(),
                'probabilities': combined_probs.cpu(),
                'predicted_names': [FAILURE_NAMES.get(p, f'Class {p}') for p in pred_classes.cpu().tolist()]
            }
            
            if return_components:
                result['components'] = {k: v.cpu() if torch.is_tensor(v) else v 
                                        for k, v in components.items()}
            
            return result
        
        def get_high_risk_jobs(self, predictions: dict, threshold: float = 0.7) -> list:
            """
            Identify jobs at high risk of failure.
            
            Args:
                predictions: Output from predict()
                threshold: Confidence threshold for failure prediction
                
            Returns:
                List of (job_idx, predicted_failure, confidence) tuples
            """
            high_risk = []
            
            for idx, (pred, conf) in enumerate(zip(predictions['predictions'], 
                                                    predictions['confidences'])):
                if pred != 0 and conf >= threshold:  # Not SUCCESS and high confidence
                    high_risk.append({
                        'job_idx': idx,
                        'predicted_failure': FAILURE_NAMES.get(pred, f'Class {pred}'),
                        'confidence': conf,
                        'probabilities': predictions['probabilities'][idx].tolist()
                    })
            
            # Sort by confidence
            high_risk.sort(key=lambda x: -x['confidence'])
            return high_risk


    def train_ensemble(jobs: list, edges: list, epochs: int = 100,
                       verbose: bool = True) -> dict:
        """
        Train all three models and create ensemble.
        
        Args:
            jobs: List of job dicts
            edges: Similarity edges
            epochs: Training epochs per model
            verbose: Print progress
            
        Returns:
            Dict with ensemble and individual models
        """
        from .gnn_torch import train_failure_gnn, prepare_pyg_data
        from .lstm import train_failure_lstm, generate_synthetic_trajectories
        from .autoencoder import train_anomaly_detector
        
        results = {}
        
        # 1. Train GNN
        if verbose:
            print("=" * 60)
            print("Training GNN (network structure)")
            print("=" * 60)
        gnn_result = train_failure_gnn(jobs, edges, epochs=epochs, verbose=verbose)
        results['gnn'] = gnn_result
        
        # 2. Train LSTM
        if verbose:
            print("\n" + "=" * 60)
            print("Training LSTM (temporal patterns)")
            print("=" * 60)
        lstm_result = train_failure_lstm(jobs, epochs=epochs, verbose=verbose)
        results['lstm'] = lstm_result
        
        # 3. Train Autoencoder
        if verbose:
            print("\n" + "=" * 60)
            print("Training Autoencoder (anomaly detection)")
            print("=" * 60)
        ae_result = train_anomaly_detector(jobs, epochs=epochs, verbose=verbose)
        results['autoencoder'] = ae_result
        
        # 4. Create ensemble
        ensemble = FailureEnsemble(
            gnn_model=gnn_result['model'],
            lstm_model=lstm_result['model'],
            autoencoder=ae_result['model']
        )
        results['ensemble'] = ensemble
        
        # 5. Evaluate ensemble
        if verbose:
            print("\n" + "=" * 60)
            print("Ensemble Summary")
            print("=" * 60)
            print(f"GNN test accuracy:    {gnn_result['test_results']['accuracy']:.2%}")
            print(f"LSTM test accuracy:   {lstm_result['test_accuracy']:.2%}")
            print(f"Autoencoder F1:       {ae_result['results'].get('f1', 0):.2%}")
        
        return results


else:
    class FailureEnsemble:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required")
    
    def train_ensemble(*args, **kwargs):
        raise ImportError("PyTorch required")


if __name__ == '__main__':
    if not HAS_TORCH:
        print("PyTorch not available")
    else:
        import sqlite3
        
        print("Training Ensemble on simulation data...")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print()
        
        # Load data
        conn = sqlite3.connect('vm-simulation/nomade.db')
        conn.row_factory = sqlite3.Row
        jobs = [dict(row) for row in conn.execute("SELECT * FROM jobs").fetchall()]
        print(f"Loaded {len(jobs)} jobs")
        
        # Build edges
        from nomade.viz.server import build_bipartite_network
        network = build_bipartite_network(jobs, threshold=0.5, max_edges=15000)
        edges = [{'source': e['source'], 'target': e['target']} for e in network['edges']]
        print(f"Built {len(edges)} Simpson edges")
        print()
        
        # Train ensemble
        results = train_ensemble(jobs, edges, epochs=100, verbose=True)
        
        print("\n" + "=" * 60)
        print("ENSEMBLE READY")
        print("=" * 60)
