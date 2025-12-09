# NOMADE Development Roadmap

## Timeline Overview

```
2025 Q1 (Jan-Mar)     Phase 1: Monitoring Foundation
2025 Q2 (Apr-Jun)     Phase 2: Prediction Engine
2025 Q3 (Jul-Sep)     Phase 3: Visualization & Integration
2025 Q4 (Oct-Dec)     Phase 4: Paper 1 & Release
2026 Q1-Q2            Phase 5: Advanced ML & Paper 2
```

---

## Phase 1: Monitoring Foundation (Jan-Mar 2025)

### Milestone 1.1: Core Infrastructure
**Target: End of January**

- [ ] **Database Layer**
  - [ ] SQLite schema design and implementation
  - [ ] Data models (Python dataclasses)
  - [ ] Query utilities
  - [ ] Migration system for schema updates

- [ ] **Configuration System**
  - [ ] TOML parser and validation
  - [ ] Default configuration
  - [ ] Environment variable overrides
  - [ ] Config hot-reload support

- [ ] **Logging & Error Handling**
  - [ ] Structured logging setup
  - [ ] Log rotation
  - [ ] Error categorization

### Milestone 1.2: Collectors
**Target: End of February**

- [ ] **Base Collector Framework**
  - [ ] Abstract base class
  - [ ] Collection scheduling
  - [ ] Error handling and retry logic
  - [ ] Metrics storage interface

- [ ] **Disk Collector**
  - [ ] Filesystem usage (df parsing)
  - [ ] Quota tracking (quota command)
  - [ ] Fill rate calculation
  - [ ] Derivative analysis integration
  - [ ] Large file detection

- [ ] **SLURM Collector**
  - [ ] Queue state (squeue)
  - [ ] Job history (sacct)
  - [ ] Node state (sinfo)
  - [ ] Partition statistics
  - [ ] Pending job analysis

- [ ] **Node Collector**
  - [ ] Node status from SLURM
  - [ ] SSH-based health checks
  - [ ] Temperature monitoring (sensors, nvidia-smi)
  - [ ] NFS mount verification
  - [ ] Service status checks

- [ ] **License Collector**
  - [ ] FlexLM query parsing
  - [ ] RLM support
  - [ ] Generic license server interface
  - [ ] Expiration tracking

### Milestone 1.3: Alert System
**Target: End of March**

- [ ] **Alert Engine**
  - [ ] Rule evaluation framework
  - [ ] Threshold-based rules
  - [ ] Derivative-based rules
  - [ ] Alert deduplication
  - [ ] Cooldown management
  - [ ] Severity levels

- [ ] **Derivative Analysis**
  - [ ] First derivative calculation
  - [ ] Second derivative calculation
  - [ ] Trend classification
  - [ ] Projection (linear and quadratic)
  - [ ] Smoothing options

- [ ] **Dispatch System**
  - [ ] Email dispatcher
  - [ ] Slack webhook dispatcher
  - [ ] Generic webhook support
  - [ ] Alert acknowledgment tracking

- [ ] **CLI Interface**
  - [ ] `nomade init`
  - [ ] `nomade start/stop/status`
  - [ ] `nomade disk/queue/nodes/licenses`
  - [ ] `nomade alerts`

### Phase 1 Deliverables
- Working monitoring daemon
- Email alerts for threshold and derivative triggers
- CLI for status checking
- SQLite database with historical data
- Basic documentation

---

## Phase 2: Prediction Engine (Apr-Jun 2025)

### Milestone 2.1: Job Metrics Collection
**Target: End of April**

- [ ] **SLURM Hooks**
  - [ ] Prolog script (job start)
  - [ ] Epilog script (job end)
  - [ ] cgroup metrics extraction
  - [ ] GPU metrics (nvidia-smi)
  - [ ] I/O metrics (from /proc or cgroup)

- [ ] **Job Data Model**
  - [ ] Job metadata table
  - [ ] Job metrics table (time-series)
  - [ ] Job summary table (computed at end)
  - [ ] Health score storage

### Milestone 2.2: Similarity Network
**Target: End of May**

- [ ] **Feature Engineering**
  - [ ] Raw metric extraction
  - [ ] Normalization (z-score, min-max)
  - [ ] Non-redundant feature set
  - [ ] Feature correlation analysis

- [ ] **Similarity Computation**
  - [ ] Cosine similarity implementation
  - [ ] Efficient pairwise computation
  - [ ] Threshold-based edge creation
  - [ ] Network storage format

- [ ] **Health Score Model**
  - [ ] Initial formula (domain knowledge)
  - [ ] Continuous score (0→1)
  - [ ] Calibration against outcomes
  - [ ] Cluster-based prediction

### Milestone 2.3: Simulation & Validation
**Target: End of June**

- [ ] **Generative Model**
  - [ ] Fit distributions to empirical data
  - [ ] Profile-based simulation
  - [ ] Correlation preservation
  - [ ] Synthetic job generation

- [ ] **Validation Framework**
  - [ ] Coverage analysis
  - [ ] Anomaly detection
  - [ ] Distribution comparison
  - [ ] Temporal drift monitoring

- [ ] **Error Analysis**
  - [ ] Confusion matrix computation
  - [ ] Type 1/Type 2 error rates
  - [ ] ROC curve generation
  - [ ] Threshold optimization
  - [ ] Defaults derivation

- [ ] **Recommendations**
  - [ ] Feature impact analysis
  - [ ] Threshold extraction
  - [ ] User-specific suggestions
  - [ ] Training identification

### Phase 2 Deliverables
- Per-job metrics collection via SLURM hooks
- Similarity network from real cluster data
- Health score prediction
- Data-driven recommendations
- Simulation validation framework

---

## Phase 3: Visualization & Integration (Jul-Sep 2025)

### Milestone 3.1: Dashboard Backend
**Target: End of July**

- [ ] **API Server**
  - [ ] FastAPI or Flask backend
  - [ ] REST endpoints for all data
  - [ ] WebSocket for real-time updates
  - [ ] Authentication (optional)

- [ ] **Data Aggregation**
  - [ ] Time-series aggregation
  - [ ] Rollup tables for performance
  - [ ] Efficient queries for dashboard

### Milestone 3.2: Dashboard Frontend
**Target: End of August**

- [ ] **Monitoring Views**
  - [ ] Disk usage dashboard
  - [ ] Queue status display
  - [ ] Node health grid
  - [ ] License availability
  - [ ] Alert management

- [ ] **Prediction Views**
  - [ ] 2D similarity network
  - [ ] Health score distribution
  - [ ] Feature correlation panel
  - [ ] Recommendations display

- [ ] **3D Visualization**
  - [ ] Three.js network rendering
  - [ ] Interactive rotation/zoom
  - [ ] Safe/danger zone display
  - [ ] Simulation cloud overlay
  - [ ] Real-time job tracking

### Milestone 3.3: Integration & Testing
**Target: End of September**

- [ ] **End-to-End Testing**
  - [ ] Collector integration tests
  - [ ] Alert system tests
  - [ ] Prediction accuracy tests
  - [ ] Dashboard functional tests

- [ ] **Documentation**
  - [ ] Installation guide
  - [ ] Configuration reference
  - [ ] API documentation
  - [ ] User guide

- [ ] **Performance Optimization**
  - [ ] Database query optimization
  - [ ] Collection efficiency
  - [ ] Dashboard responsiveness

### Phase 3 Deliverables
- Complete web dashboard
- 3D network visualization
- Real-time updates
- Comprehensive documentation
- Performance-tested system

---

## Phase 4: Paper 1 & Release (Oct-Dec 2025)

### Milestone 4.1: Case Study
**Target: End of October**

- [ ] **Production Cluster Deployment**
  - [ ] Full deployment on Production Cluster
  - [ ] 2+ months of production data
  - [ ] User feedback collection

- [ ] **Metrics Collection**
  - [ ] Alert effectiveness analysis
  - [ ] Prediction accuracy metrics
  - [ ] System overhead measurements
  - [ ] User satisfaction survey

- [ ] **VM Simulation Environment**
  - [ ] Data anonymization pipeline (remove users, paths, hostnames)
  - [ ] Export tool for Production Cluster data → portable dataset
  - [ ] Data replay engine (feed historical data as "live" events)
  - [ ] Mock SLURM commands (squeue, sacct responses from data)
  - [ ] VM image or Docker container with full NOMADE stack
  - [ ] Documentation for reproducibility

### Milestone 4.2: Paper Writing
**Target: End of November**

- [ ] **Paper 1 Draft**
  - [ ] Introduction and motivation
  - [ ] Architecture description
  - [ ] Feature documentation
  - [ ] Case study results
  - [ ] Performance analysis

- [ ] **Figures and Tables**
  - [ ] Architecture diagram
  - [ ] Screenshot gallery
  - [ ] Performance charts
  - [ ] Comparison tables

### Milestone 4.3: Release
**Target: End of December**

- [ ] **Open Source Release**
  - [ ] Code cleanup
  - [ ] License files
  - [ ] GitHub repository setup
  - [ ] PyPI package

- [ ] **Paper Submission**
  - [ ] JOSS or SoftwareX submission
  - [ ] Reviewer response preparation

### Phase 4 Deliverables
- Production deployment on Production Cluster
- Tool paper submitted
- Open source release v1.0
- PyPI package

---

## Phase 5: Advanced ML & Paper 2 (2026 Q1-Q2)

### Milestone 5.1: Advanced Models
**Target: End of February 2026**

- [ ] **GNN Implementation**
  - [ ] PyTorch Geometric setup
  - [ ] Graph construction from similarity network
  - [ ] Node-level prediction (job health)
  - [ ] Training pipeline

- [ ] **LSTM Early Warning**
  - [ ] Time-series feature extraction
  - [ ] Derivative features
  - [ ] Early warning prediction
  - [ ] Alert integration

- [ ] **Ensemble Methods**
  - [ ] Model combination
  - [ ] Confidence estimation
  - [ ] Disagreement detection

### Milestone 5.2: Partnerships
**Target: End of April 2026**

- [ ] **Partner Outreach**
  - [ ] Contact potential partners
  - [ ] Data sharing agreements
  - [ ] Deployment assistance

- [ ] **Multi-Cluster Data**
  - [ ] Anonymization pipeline
  - [ ] Cross-cluster analysis
  - [ ] Universal vs local patterns

### Milestone 5.3: Paper 2
**Target: Summer 2026**

- [ ] **Research Analysis**
  - [ ] Emergent pattern discovery
  - [ ] Biogeographical analogy validation
  - [ ] Prediction vs baseline comparison
  - [ ] Cross-institution validation

- [ ] **Paper 2 Writing**
  - [ ] Methods focus
  - [ ] Theoretical framework
  - [ ] Multi-cluster results
  - [ ] Nature Computational Science target

### Phase 5 Deliverables
- GNN and LSTM models
- Multi-cluster deployment
- Paper 2 submitted
- Community data federation

---

## Task Tracking

### Priority Labels
- 🔴 **P0**: Critical path, blocks other work
- 🟠 **P1**: Important, should be done soon
- 🟡 **P2**: Nice to have, can be deferred
- 🟢 **P3**: Future enhancement

### Status Labels
- ⬜ Not started
- 🟨 In progress
- ✅ Complete
- ❌ Blocked

---

## Dependencies

### External Dependencies
- Python 3.9+
- SQLite 3.35+
- SLURM (for queue monitoring)
- nvidia-smi (for GPU monitoring)
- React/Three.js (for visualization)

### Python Dependencies
```
# Core
toml>=0.10
click>=8.0
sqlalchemy>=2.0

# Analysis
numpy>=1.21
scipy>=1.7
pandas>=1.3

# Prediction (Phase 2)
scikit-learn>=1.0
torch>=2.0
torch-geometric>=2.0

# Visualization (Phase 3)
fastapi>=0.100
uvicorn>=0.20
jinja2>=3.0

# Development
pytest>=7.0
ruff>=0.1
black>=23.0
```

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| SLURM access restrictions | Can't collect job metrics | Fallback to sacct-only data |
| No root on cluster | Limited cgroup access | Use available SLURM data |
| ML model underperforms | Poor predictions | Start with simple rules, add ML later |
| Dashboard too complex | Delayed release | MVP first, enhance iteratively |
| Partner data unavailable | Paper 2 scope limited | Focus on single-cluster depth |

---

## Success Metrics

### Phase 1
- [ ] Monitoring daemon runs 7+ days without crash
- [ ] Alerts delivered within 60 seconds of trigger
- [ ] <1% CPU overhead on head node

### Phase 2
- [ ] >80% jobs have metrics collected
- [ ] Prediction accuracy >70%
- [ ] Recommendations improve success rate by >10%

### Phase 3
- [ ] Dashboard loads in <3 seconds
- [ ] 3D visualization runs at 30+ FPS
- [ ] Real-time updates within 5 seconds

### Phase 4
- [ ] Paper 1 submitted to JOSS/SoftwareX
- [ ] >10 GitHub stars within 3 months
- [ ] At least 1 external deployment inquiry
