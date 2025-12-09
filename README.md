# NOMADE

**NOde MAnagement DEvice** — A lightweight HPC monitoring and predictive analytics tool.

> *"Travels light, adapts to its environment, and doesn't need permanent infrastructure."*

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

NOMADE is a lightweight, self-contained monitoring and prediction system for HPC clusters. Unlike heavyweight monitoring solutions that require complex infrastructure, NOMADE is designed to be deployed quickly, run with minimal resources, and provide actionable insights through both real-time alerts and predictive analytics.

### Key Features

- **Real-time Monitoring**: Track disk usage, SLURM queues, node health, license servers, and job metrics
- **Derivative Analysis**: Detect accelerating trends before they become critical (not just threshold alerts)
- **Predictive Analytics**: ML-based job health prediction using similarity networks
- **Actionable Recommendations**: Data-driven defaults and user-specific suggestions
- **3D Visualization**: Interactive network visualization with safe/danger zones
- **Lightweight**: SQLite database, minimal dependencies, no external services required

### Philosophy

NOMADE is inspired by nomadic principles:
- **Travels light**: Minimal dependencies, single SQLite database, no complex infrastructure
- **Adapts to its environment**: Configurable collectors, flexible alert rules, cluster-agnostic
- **Leaves no trace**: Clean uninstall, no system modifications required (except optional SLURM hooks)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              NOMADE                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      ALERT DISPATCHER                            │   │
│  │         📧 Email    💬 Slack    🔔 Webhook    📊 Dashboard       │   │
│  └─────────────────────────────┬───────────────────────────────────┘   │
│                                │                                        │
│  ┌─────────────────────────────┴───────────────────────────────────┐   │
│  │                      ALERT ENGINE                                │   │
│  │       Rules · Derivatives · Deduplication · Cooldowns            │   │
│  └─────────────────────────────┬───────────────────────────────────┘   │
│                                │                                        │
│         ┌──────────────────────┴──────────────────────┐                │
│         ▼                                             ▼                │
│  ┌─────────────────────┐                ┌─────────────────────────┐   │
│  │  MONITORING ENGINE  │                │   PREDICTION ENGINE     │   │
│  │  Threshold-based    │                │   Pattern-based ML      │   │
│  │  Immediate alerts   │                │   Recommendations       │   │
│  └─────────┬───────────┘                └────────────┬────────────┘   │
│            │                                          │                │
│            └──────────────────┬───────────────────────┘                │
│                               │                                        │
│  ┌────────────────────────────┴────────────────────────────────────┐  │
│  │                         DATA LAYER                               │  │
│  │            SQLite · Time-series · Job History                    │  │
│  └────────────────────────────┬────────────────────────────────────┘  │
│                               │                                        │
│  ┌────────────────────────────┴────────────────────────────────────┐  │
│  │                        COLLECTORS                                │  │
│  │   Disk │ SLURM │ Nodes │ Licenses │ Jobs │ Network              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Two Engines, One System

1. **Monitoring Engine**: Real-time threshold and derivative-based alerts
   - Catches immediate issues (disk full, node down, stuck jobs)
   - Uses first and second derivatives for early warning
   - "Your disk fill rate is *accelerating* — full in 3 days, not 10"

2. **Prediction Engine**: Pattern-based ML analytics
   - Catches patterns before they become issues
   - Uses job similarity networks and health prediction
   - "Jobs with your I/O pattern have 72% failure rate"

---

## Monitoring Capabilities

### Disk Storage
- Filesystem usage monitoring (/, /home, /scratch, /project)
- Per-user and per-group quota tracking
- Fill rate calculation and projection
- **Derivative analysis**: Detect accelerating growth before thresholds trigger
- Orphan file and stale data detection
- Localscratch cleanup verification

### SLURM Queue
- Queue depth and wait time tracking
- Stuck and zombie job detection
- Node drain status monitoring
- Fairshare imbalance alerts
- Pending job analysis (why is my job waiting?)
- Job array health monitoring

### Node Health
- Node up/down/drain status
- Hardware error detection (ECC, GPU, disk)
- Temperature monitoring (CPU, GPU)
- NFS mount health
- Service status (slurmctld, slurmd, munge)
- Network connectivity checks

### License Servers
- FlexLM and RLM license tracking
- Real-time availability monitoring
- Usage pattern analysis
- Server connectivity alerts
- Expiration warnings

### Job Metrics
- Per-job resource usage (CPU, memory, GPU)
- I/O patterns (NFS vs local storage)
- Runtime and efficiency metrics
- Collected via SLURM prolog/epilog hooks

---

## Prediction Capabilities

### Quantitative Similarity Network

NOMADE builds a similarity network from job metrics:

- **Raw quantitative metrics**: No arbitrary thresholds or binary labels
  - CPU%, VRAM (GB), Memory (GB), Swap (GB)
  - NFS read/write (GB), Local read/write (GB)
  - I/O wait (%), Runtime (hrs)
  
- **Non-redundant features**: `vram_gb > 0` implies GPU used (no separate flag)

- **Cosine similarity**: On normalized feature vectors

- **Continuous health score**: 0 (catastrophic) → 1 (perfect), not binary

### Simulation & Validation

- **Generative model**: Learn distributions from empirical data
- **Simulation cloud**: Thousands of synthetic jobs for coverage validation
- **Anomaly detection**: Real jobs outside simulation bounds
- **Temporal drift**: Monitor for model staleness

### Error Analysis & Defaults

- **Type 1 errors** (false alarms): Predicted failure, actually succeeded
- **Type 2 errors** (missed failures): Predicted success, actually failed
- **Threshold optimization**: Balance alert fatigue vs missed problems
- **Data-driven defaults**: "Use localscratch → +23% success rate"

### Visualization

- **3D network visualization**: Three.js interactive display
- **Axes**: NFS Write / Local Write / I/O Wait
- **Safe zone**: Low NFS, high local, low I/O wait (green region)
- **Danger zone**: High NFS, low local, high I/O wait (red region)
- **Real-time tracking**: Watch jobs move through feature space

---

## Derivative Analysis

A key innovation in NOMADE is the use of first and second derivatives for early warning:

```
VALUE (0th derivative):     "Disk is at 850 GB"
FIRST DERIVATIVE:           "Disk is filling at 15 GB/day"  
SECOND DERIVATIVE:          "Fill rate is ACCELERATING at 3 GB/day²"
```

### Why Second Derivatives Matter

Traditional threshold alerts only trigger when a value crosses a limit. By monitoring the second derivative (acceleration), NOMADE can detect:

- **Exponential growth**: Before linear projections underestimate
- **Sudden changes**: Spikes in usage patterns
- **Developing problems**: I/O storms, memory leaks, cascading failures

### Applications

| Metric | Accelerating (d²>0) | Decelerating (d²<0) |
|--------|---------------------|---------------------|
| Disk usage | ⚠️ Exponential fill | ✓ Cleanup in progress |
| Queue depth | ⚠️ System issue | ✓ Draining normally |
| Failure rate | 🔴 Cascading problem | ✓ Issue resolving |
| NFS latency | ⚠️ I/O storm developing | ✓ Load decreasing |
| Job memory | ⚠️ Memory leak / OOM | ✓ Normal variation |
| GPU temp | ⚠️ Cooling issue | ✓ Throttling working |

---

## Installation

### Requirements

- Python 3.9+
- SQLite 3.35+
- SLURM (for queue and job monitoring)
- Root access (optional, for cgroup metrics)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/jtonini/nomade.git
cd nomade

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
nomade init

# Start monitoring daemon
nomade start

# View dashboard
nomade dashboard
```

### SLURM Integration (Optional)

For per-job metrics collection, install prolog/epilog hooks:

```bash
# Copy hooks to SLURM configuration
sudo cp scripts/prolog.sh /etc/slurm/prolog.d/nomade.sh
sudo cp scripts/epilog.sh /etc/slurm/epilog.d/nomade.sh

# Update slurm.conf
# Prolog=/etc/slurm/prolog.d/*
# Epilog=/etc/slurm/epilog.d/*

# Restart SLURM
sudo systemctl restart slurmctld
```

---

## Configuration

NOMADE uses a TOML configuration file:

```toml
# nomade.toml

[general]
cluster_name = "mycluster"
data_dir = "/var/lib/nomade"
log_level = "INFO"

[collectors]
# Enable/disable collectors
disk = true
slurm = true
nodes = true
licenses = true
jobs = true
network = true

# Collection intervals (seconds)
disk_interval = 300
slurm_interval = 60
nodes_interval = 120

[collectors.disk]
# Filesystems to monitor
filesystems = ["/", "/home", "/scratch", "/project"]

# Quota sources
quota_command = "quota -g"

[collectors.licenses]
# License servers to monitor
[[collectors.licenses.servers]]
name = "matlab"
type = "flexlm"
host = "license.example.edu"
port = 27000

[[collectors.licenses.servers]]
name = "gaussian"
type = "flexlm"
host = "license.example.edu"
port = 27001

[alerts]
# Alert dispatch configuration
email_enabled = true
email_to = ["admin@example.edu"]
email_from = "nomade@cluster.example.edu"
smtp_host = "smtp.example.edu"

slack_enabled = false
slack_webhook = ""

# Alert thresholds
disk_warning_percent = 85
disk_critical_percent = 95
queue_stuck_days = 7
gpu_temp_warning = 83

[alerts.derivatives]
# Second derivative thresholds
disk_acceleration_warning = 1.0  # GB/day²
queue_acceleration_warning = 5   # jobs/hour²

[prediction]
# Prediction engine settings
enabled = true
similarity_threshold = 0.85
health_threshold = 0.5
retrain_interval_days = 7

[dashboard]
host = "0.0.0.0"
port = 8080
```

---

## Usage

### Command Line Interface

```bash
# Start/stop monitoring daemon
nomade start
nomade stop
nomade status

# View current state
nomade disk          # Disk usage summary
nomade queue         # Queue status
nomade nodes         # Node health
nomade licenses      # License availability
nomade alerts        # Recent alerts

# Prediction features
nomade predict <job_id>     # Predict job health
nomade recommend <user>     # User-specific recommendations
nomade defaults             # Show data-driven defaults

# Dashboard
nomade dashboard            # Start web dashboard

# Database management
nomade init                 # Initialize database
nomade export               # Export data for analysis
nomade prune --days 90      # Remove old data
```

### Python API

```python
from nomade import Nomade

# Initialize
nm = Nomade(config_path='nomade.toml')

# Get current disk status
disk_status = nm.collectors.disk.get_status()
for fs in disk_status:
    print(f"{fs['path']}: {fs['used_pct']:.1f}%")
    
# Analyze trends
analysis = nm.analysis.analyze_disk('/scratch')
print(f"Fill rate: {analysis['first_derivative']:.1f} GB/day")
print(f"Acceleration: {analysis['second_derivative']:.2f} GB/day²")
print(f"Trend: {analysis['trend']}")

# Predict job health
prediction = nm.prediction.predict_job(job_metrics)
print(f"Predicted health: {prediction['health']:.2f}")
print(f"Risk level: {prediction['risk_level']}")
print(f"Recommendations: {prediction['recommendations']}")

# Get recommendations for a user
recs = nm.prediction.recommend_for_user('alice')
for rec in recs:
    print(f"- {rec['message']}")
```

---

## Repository Structure

```
nomade/
├── README.md                 # This file
├── LICENSE                   # AGPL v3
├── pyproject.toml           # Package configuration
├── requirements.txt         # Dependencies
├── nomade.toml.example      # Example configuration
│
├── nomade/                  # Main package
│   ├── __init__.py
│   ├── cli.py               # Command-line interface
│   ├── daemon.py            # Main monitoring daemon
│   ├── config.py            # Configuration handling
│   │
│   ├── collectors/          # Data collectors
│   │   ├── __init__.py
│   │   ├── base.py          # Base collector class
│   │   ├── disk.py          # Disk & quota monitoring
│   │   ├── slurm.py         # SLURM queue & jobs
│   │   ├── nodes.py         # Node health
│   │   ├── licenses.py      # License servers
│   │   ├── jobs.py          # Per-job metrics
│   │   └── network.py       # Network monitoring
│   │
│   ├── db/                  # Database layer
│   │   ├── __init__.py
│   │   ├── schema.sql       # SQLite schema
│   │   ├── models.py        # Data models
│   │   └── queries.py       # Common queries
│   │
│   ├── analysis/            # Analysis utilities
│   │   ├── __init__.py
│   │   ├── derivatives.py   # Derivative calculations
│   │   ├── projections.py   # Trend projections
│   │   └── timeseries.py    # Time-series utilities
│   │
│   ├── alerts/              # Alert system
│   │   ├── __init__.py
│   │   ├── engine.py        # Alert evaluation
│   │   ├── rules.py         # Alert rule definitions
│   │   └── dispatch.py      # Email/Slack/webhook
│   │
│   ├── prediction/          # ML prediction
│   │   ├── __init__.py
│   │   ├── similarity.py    # Cosine similarity
│   │   ├── network.py       # Similarity network
│   │   ├── health.py        # Health score prediction
│   │   ├── simulation.py    # Simulation model
│   │   ├── errors.py        # Type 1/2 error analysis
│   │   └── recommendations.py  # Defaults generation
│   │
│   └── viz/                 # Visualization
│       ├── __init__.py
│       ├── dashboard.py     # Web dashboard
│       └── static/          # React frontend
│           ├── index.html
│           └── components/
│               ├── Network3D.jsx
│               ├── DiskStatus.jsx
│               ├── QueueStatus.jsx
│               └── Alerts.jsx
│
├── scripts/                 # Utility scripts
│   ├── prolog.sh           # SLURM prolog hook
│   ├── epilog.sh           # SLURM epilog hook
│   └── install_hooks.sh    # Hook installer
│
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_collectors.py
│   ├── test_analysis.py
│   ├── test_alerts.py
│   └── test_prediction.py
│
└── docs/                    # Documentation
    ├── installation.md
    ├── configuration.md
    ├── collectors.md
    ├── alerts.md
    ├── prediction.md
    └── api.md
```

---

## Theoretical Background

NOMADE's prediction engine is inspired by biogeographical network analysis, particularly the work of Vilhena & Antonelli (2015) on mapping biomes using species occurrence data.

### Biogeography → HPC Analogy

| Biogeography | HPC Infrastructure |
|--------------|-------------------|
| Species | Jobs |
| Geographic regions | Resources (nodes, storage) |
| Biomes | Emergent behavior clusters |
| Species ranges | Job resource usage patterns |
| Transition zones | Domain boundaries (CPU↔GPU, NFS↔local) |

### Key Insight

Just as biogeographical regions emerge from species distribution data rather than being predefined, NOMADE allows behavior patterns to emerge from job metrics rather than imposing arbitrary categories.

### Dual-View Analysis

1. **Data space**: Jobs as points in feature space, clustered by similarity
2. **Real space**: Jobs mapped to physical resources, showing actual infrastructure usage

---

## Roadmap

### Phase 1: Monitoring Foundation ✓
- [x] Design architecture
- [x] Define data model
- [ ] Implement collectors
- [ ] Implement alert engine
- [ ] Basic dashboard

### Phase 2: Prediction Engine
- [ ] Similarity network computation
- [ ] Health score prediction
- [ ] Simulation framework
- [ ] Error analysis
- [ ] Recommendations

### Phase 3: Visualization
- [ ] 3D network visualization
- [ ] Real-time job tracking
- [ ] Interactive dashboard
- [ ] Safe/danger zone display

### Phase 4: Advanced ML
- [ ] GNN for network-aware prediction
- [ ] LSTM for early warning
- [ ] Ensemble methods
- [ ] Continuous learning

### Phase 5: Community
- [ ] Multi-cluster federation
- [ ] Anonymized data sharing
- [ ] Community benchmarks

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/jtonini/nomade.git
cd nomade
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Build documentation
cd docs && make html
```

---

## License

NOMADE is dual-licensed:

- **AGPL v3**: Free for academic, educational, and open-source use
- **Commercial License**: Available for proprietary/commercial deployments

See [LICENSE](LICENSE) for details.

---

## Citation

If you use NOMADE in your research, please cite:

```bibtex
@software{nomade2026,
  author = {Tonini, Joao},
  title = {NOMADE: A Lightweight HPC Monitoring and Prediction Tool},
  year = {2026},
  url = {https://github.com/jtonini/nomade}
}
```

---

## Acknowledgments

- Biogeographical network analysis inspired by Vilhena & Antonelli (2015)

---

## Contact

- **Author**: Joao Tonini
- **Email**: jtonini@richmond.edu
- **Issues**: [GitHub Issues](https://github.com/jtonini/nomade/issues)
