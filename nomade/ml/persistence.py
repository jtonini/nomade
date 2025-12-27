"""
Model persistence and prediction storage.

Saves trained models and predictions to database for fast dashboard loading.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def init_ml_tables(db_path: str):
    """Create ML tables in database if they don't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Model metadata table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ml_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            n_jobs_trained INTEGER,
            metrics TEXT,
            model_path TEXT,
            is_active INTEGER DEFAULT 1
        )
    ''')
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            n_jobs INTEGER,
            n_anomalies INTEGER,
            threshold REAL,
            high_risk_jobs TEXT,
            summary TEXT,
            FOREIGN KEY (model_id) REFERENCES ml_models(id)
        )
    ''')
    
    # Job-level predictions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_predictions (
            job_id TEXT PRIMARY KEY,
            prediction_id INTEGER,
            predicted_class INTEGER,
            confidence REAL,
            anomaly_score REAL,
            is_anomaly INTEGER,
            gnn_probs TEXT,
            lstm_probs TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prediction_id) REFERENCES ml_predictions(id)
        )
    ''')
    
    conn.commit()
    conn.close()


def save_ensemble_models(models_dir: Path, gnn_model, lstm_model, ae_model, metadata: dict):
    """Save trained ensemble models to disk."""
    if not HAS_TORCH:
        return None
    
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save models
    if gnn_model:
        torch.save(gnn_model.state_dict(), models_dir / f'gnn_{timestamp}.pt')
    if lstm_model:
        torch.save(lstm_model.state_dict(), models_dir / f'lstm_{timestamp}.pt')
    if ae_model:
        torch.save(ae_model.state_dict(), models_dir / f'ae_{timestamp}.pt')
    
    # Save metadata
    metadata['timestamp'] = timestamp
    metadata['saved_at'] = datetime.now().isoformat()
    with open(models_dir / f'metadata_{timestamp}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return timestamp


def load_latest_models(models_dir: Path):
    """Load most recent trained models."""
    if not HAS_TORCH:
        return None
    
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return None
    
    # Find latest metadata
    metadata_files = sorted(models_dir.glob('metadata_*.json'), reverse=True)
    if not metadata_files:
        return None
    
    with open(metadata_files[0]) as f:
        metadata = json.load(f)
    
    timestamp = metadata.get('timestamp')
    if not timestamp:
        return None
    
    return {
        'metadata': metadata,
        'gnn_path': models_dir / f'gnn_{timestamp}.pt',
        'lstm_path': models_dir / f'lstm_{timestamp}.pt',
        'ae_path': models_dir / f'ae_{timestamp}.pt'
    }


def save_predictions_to_db(db_path: str, predictions: dict, model_id: int = None):
    """Save ML predictions to database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Ensure tables exist
    init_ml_tables(db_path)
    
    # Insert prediction summary
    cursor.execute('''
        INSERT INTO ml_predictions 
        (model_id, n_jobs, n_anomalies, threshold, high_risk_jobs, summary)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        model_id,
        predictions.get('n_jobs', 0),
        predictions.get('n_anomalies', 0),
        predictions.get('threshold', 0),
        json.dumps(predictions.get('high_risk', [])),
        json.dumps(predictions.get('summary', {}))
    ))
    
    prediction_id = cursor.lastrowid
    
    # Insert job-level predictions
    for job_pred in predictions.get('job_predictions', []):
        cursor.execute('''
            INSERT OR REPLACE INTO job_predictions
            (job_id, prediction_id, predicted_class, confidence, anomaly_score, is_anomaly, gnn_probs, lstm_probs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job_pred.get('job_id'),
            prediction_id,
            job_pred.get('predicted_class', 0),
            job_pred.get('confidence', 0),
            job_pred.get('anomaly_score', 0),
            1 if job_pred.get('is_anomaly') else 0,
            json.dumps(job_pred.get('gnn_probs', [])),
            json.dumps(job_pred.get('lstm_probs', []))
        ))
    
    conn.commit()
    conn.close()
    
    return prediction_id


def load_predictions_from_db(db_path: str):
    """Load latest ML predictions from database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ml_predictions'")
    if not cursor.fetchone():
        conn.close()
        return None
    
    # Get latest prediction
    cursor.execute('''
        SELECT * FROM ml_predictions 
        ORDER BY created_at DESC 
        LIMIT 1
    ''')
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return None
    
    predictions = {
        'status': 'loaded',
        'prediction_id': row['id'],
        'created_at': row['created_at'],
        'n_jobs': row['n_jobs'],
        'n_anomalies': row['n_anomalies'],
        'threshold': row['threshold'],
        'high_risk': json.loads(row['high_risk_jobs']) if row['high_risk_jobs'] else [],
        'summary': json.loads(row['summary']) if row['summary'] else {}
    }
    
    # Get job-level predictions
    cursor.execute('''
        SELECT * FROM job_predictions 
        WHERE prediction_id = ?
        ORDER BY anomaly_score DESC
    ''', (row['id'],))
    
    predictions['job_predictions'] = [dict(r) for r in cursor.fetchall()]
    
    conn.close()
    return predictions


def get_prediction_history(db_path: str, limit: int = 10):
    """Get history of ML predictions."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ml_predictions'")
    if not cursor.fetchone():
        conn.close()
        return []
    
    cursor.execute('''
        SELECT id, created_at, n_jobs, n_anomalies, threshold
        FROM ml_predictions 
        ORDER BY created_at DESC 
        LIMIT ?
    ''', (limit,))
    
    history = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return history
