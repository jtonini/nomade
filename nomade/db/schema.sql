-- NOMADE Database Schema
-- SQLite 3.35+

-- ============================================
-- INFRASTRUCTURE STATE (Monitoring)
-- ============================================

-- Filesystem usage snapshots
CREATE TABLE IF NOT EXISTS filesystems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    total_bytes INTEGER NOT NULL,
    used_bytes INTEGER NOT NULL,
    available_bytes INTEGER NOT NULL,
    used_percent REAL NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Computed fields (updated by analysis)
    fill_rate_bytes_per_day REAL,
    days_until_full REAL,
    first_derivative REAL,  -- bytes/second
    second_derivative REAL  -- bytes/second²
);

CREATE INDEX idx_filesystems_path_ts ON filesystems(path, timestamp);

-- User/group quotas
CREATE TABLE IF NOT EXISTS quotas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filesystem_path TEXT NOT NULL,
    entity_type TEXT NOT NULL CHECK (entity_type IN ('user', 'group')),
    entity_name TEXT NOT NULL,
    limit_bytes INTEGER,
    used_bytes INTEGER NOT NULL,
    used_percent REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_quotas_entity_ts ON quotas(entity_name, timestamp);

-- Node status
CREATE TABLE IF NOT EXISTS nodes (
    hostname TEXT PRIMARY KEY,
    partition TEXT,
    status TEXT NOT NULL,  -- UP, DOWN, DRAIN, FAIL, etc.
    drain_reason TEXT,
    cpu_count INTEGER,
    gpu_count INTEGER,
    memory_mb INTEGER,
    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
    hardware_errors TEXT  -- JSON
);

-- Node metrics (time-series)
CREATE TABLE IF NOT EXISTS node_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hostname TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    cpu_load_1m REAL,
    cpu_load_5m REAL,
    cpu_load_15m REAL,
    memory_used_mb INTEGER,
    swap_used_mb INTEGER,
    cpu_temp_c REAL,
    gpu_temp_c REAL,
    nfs_latency_ms REAL,
    
    FOREIGN KEY (hostname) REFERENCES nodes(hostname)
);

CREATE INDEX idx_node_metrics_host_ts ON node_metrics(hostname, timestamp);

-- License server status
CREATE TABLE IF NOT EXISTS licenses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    software TEXT NOT NULL,
    server_host TEXT NOT NULL,
    server_port INTEGER,
    total_licenses INTEGER,
    in_use INTEGER,
    available INTEGER,
    server_status TEXT NOT NULL,  -- UP, DOWN, UNKNOWN
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_date DATE
);

CREATE INDEX idx_licenses_software_ts ON licenses(software, timestamp);

-- SLURM queue state
CREATE TABLE IF NOT EXISTS queue_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    partition TEXT NOT NULL,
    pending_jobs INTEGER NOT NULL,
    running_jobs INTEGER NOT NULL,
    total_jobs INTEGER NOT NULL,
    avg_wait_seconds REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Derivatives
    first_derivative REAL,  -- jobs/second
    second_derivative REAL  -- jobs/second²
);

CREATE INDEX idx_queue_state_partition_ts ON queue_state(partition, timestamp);

-- ============================================
-- JOB DATA (Prediction)
-- ============================================

-- Job metadata
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    user_name TEXT NOT NULL,
    group_name TEXT,
    partition TEXT,
    node_list TEXT,  -- Comma-separated
    job_name TEXT,
    submit_time DATETIME,
    start_time DATETIME,
    end_time DATETIME,
    state TEXT,  -- PENDING, RUNNING, COMPLETED, FAILED, TIMEOUT, etc.
    exit_code INTEGER,
    
    -- Requested resources
    req_cpus INTEGER,
    req_mem_mb INTEGER,
    req_gpus INTEGER,
    req_time_seconds INTEGER,
    
    -- Computed
    runtime_seconds INTEGER,
    wait_time_seconds INTEGER
);

CREATE INDEX idx_jobs_user ON jobs(user_name);
CREATE INDEX idx_jobs_partition ON jobs(partition);
CREATE INDEX idx_jobs_submit ON jobs(submit_time);
CREATE INDEX idx_jobs_state ON jobs(state);

-- Job metrics (time-series during job)
CREATE TABLE IF NOT EXISTS job_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Compute
    cpu_percent REAL,  -- 0-100 per core, so can exceed 100
    memory_gb REAL,
    vram_gb REAL,
    swap_gb REAL,
    
    -- I/O
    nfs_read_gb REAL,
    nfs_write_gb REAL,
    local_read_gb REAL,
    local_write_gb REAL,
    io_wait_percent REAL,
    
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE INDEX idx_job_metrics_job_ts ON job_metrics(job_id, timestamp);

-- Job summary (computed at job end)
CREATE TABLE IF NOT EXISTS job_summary (
    job_id TEXT PRIMARY KEY,
    
    -- Peak values
    peak_cpu_percent REAL,
    peak_memory_gb REAL,
    peak_vram_gb REAL,
    peak_swap_gb REAL,
    peak_io_wait_percent REAL,
    
    -- Average values
    avg_cpu_percent REAL,
    avg_memory_gb REAL,
    avg_vram_gb REAL,
    avg_io_wait_percent REAL,
    
    -- Total I/O
    total_nfs_read_gb REAL,
    total_nfs_write_gb REAL,
    total_local_read_gb REAL,
    total_local_write_gb REAL,
    
    -- Derived metrics
    nfs_ratio REAL,  -- nfs_write / (nfs_write + local_write)
    used_gpu BOOLEAN,
    had_swap BOOLEAN,
    
    -- Health and prediction
    health_score REAL,  -- 0.0 to 1.0
    cluster_id INTEGER,
    is_anomaly BOOLEAN DEFAULT FALSE,
    anomaly_distance REAL,
    
    -- Feature vector (JSON for flexibility)
    feature_vector TEXT,
    
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

-- ============================================
-- SIMILARITY NETWORK
-- ============================================

-- Similarity edges between jobs
CREATE TABLE IF NOT EXISTS job_similarity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id_a TEXT NOT NULL,
    job_id_b TEXT NOT NULL,
    similarity REAL NOT NULL,  -- Cosine similarity, 0-1
    
    FOREIGN KEY (job_id_a) REFERENCES jobs(job_id),
    FOREIGN KEY (job_id_b) REFERENCES jobs(job_id),
    UNIQUE(job_id_a, job_id_b)
);

CREATE INDEX idx_similarity_a ON job_similarity(job_id_a);
CREATE INDEX idx_similarity_b ON job_similarity(job_id_b);

-- Job clusters
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT,
    centroid TEXT,  -- JSON feature vector
    job_count INTEGER,
    avg_health REAL,
    failure_rate REAL,
    dominant_issue TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- SIMULATION MODEL
-- ============================================

-- Empirical distributions (learned from data)
CREATE TABLE IF NOT EXISTS distributions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT NOT NULL,
    profile_name TEXT,  -- Optional: per-profile distributions
    distribution_type TEXT NOT NULL,  -- normal, lognormal, beta, etc.
    parameters TEXT NOT NULL,  -- JSON: {mu, sigma} or {alpha, beta}
    min_value REAL,
    max_value REAL,
    sample_count INTEGER,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Simulation runs
CREATE TABLE IF NOT EXISTS simulation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    n_simulated INTEGER,
    coverage_percent REAL,
    avg_distance_to_real REAL,
    anomaly_rate REAL,
    notes TEXT
);

-- ============================================
-- ALERTS & EVENTS
-- ============================================

-- Alert definitions (rules)
CREATE TABLE IF NOT EXISTS alert_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    category TEXT NOT NULL,  -- disk, queue, node, job, license
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
    condition_type TEXT NOT NULL,  -- threshold, derivative, custom
    condition_config TEXT NOT NULL,  -- JSON
    message_template TEXT NOT NULL,
    cooldown_seconds INTEGER DEFAULT 3600,
    enabled BOOLEAN DEFAULT TRUE
);

-- Alert instances
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    severity TEXT NOT NULL,
    category TEXT NOT NULL,
    source TEXT,  -- Filesystem path, node name, job_id, etc.
    message TEXT NOT NULL,
    details TEXT,  -- JSON with context
    
    -- State
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by TEXT,
    acknowledged_at DATETIME,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at DATETIME,
    
    -- Deduplication
    dedup_key TEXT,
    occurrence_count INTEGER DEFAULT 1,
    last_occurrence DATETIME,
    
    FOREIGN KEY (rule_id) REFERENCES alert_rules(id)
);

CREATE INDEX idx_alerts_ts ON alerts(timestamp);
CREATE INDEX idx_alerts_severity ON alerts(severity);
CREATE INDEX idx_alerts_category ON alerts(category);
CREATE INDEX idx_alerts_resolved ON alerts(resolved);
CREATE INDEX idx_alerts_dedup ON alerts(dedup_key);

-- Alert dispatch log
CREATE TABLE IF NOT EXISTS alert_dispatches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id INTEGER NOT NULL,
    channel TEXT NOT NULL,  -- email, slack, webhook
    recipient TEXT NOT NULL,
    sent_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    
    FOREIGN KEY (alert_id) REFERENCES alerts(id)
);

-- ============================================
-- RECOMMENDATIONS
-- ============================================

-- Data-driven defaults
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT NOT NULL,
    threshold REAL NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('above', 'below')),
    success_rate REAL NOT NULL,
    improvement REAL NOT NULL,  -- Percentage point improvement
    sample_size INTEGER NOT NULL,
    confidence REAL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Per-user recommendations
CREATE TABLE IF NOT EXISTS user_recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_name TEXT NOT NULL,
    recommendation_type TEXT NOT NULL,
    message TEXT NOT NULL,
    priority INTEGER DEFAULT 1,
    based_on_jobs INTEGER,  -- Number of jobs analyzed
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    dismissed BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_user_recs_user ON user_recommendations(user_name);

-- ============================================
-- METADATA & SYSTEM
-- ============================================

-- System configuration
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Collection runs log
CREATE TABLE IF NOT EXISTS collection_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collector TEXT NOT NULL,
    started_at DATETIME NOT NULL,
    completed_at DATETIME,
    success BOOLEAN,
    records_collected INTEGER,
    error_message TEXT
);

CREATE INDEX idx_collection_log_collector ON collection_log(collector, started_at);

-- Schema version
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

INSERT INTO schema_version (version, description) VALUES (1, 'Initial schema');

-- ============================================
-- VIEWS (Convenience)
-- ============================================

-- Recent alerts summary
CREATE VIEW IF NOT EXISTS v_recent_alerts AS
SELECT 
    a.id,
    a.timestamp,
    a.severity,
    a.category,
    a.source,
    a.message,
    a.acknowledged,
    a.resolved,
    a.occurrence_count
FROM alerts a
WHERE a.timestamp > datetime('now', '-7 days')
ORDER BY a.timestamp DESC;

-- Job health overview
CREATE VIEW IF NOT EXISTS v_job_health AS
SELECT 
    j.job_id,
    j.user_name,
    j.partition,
    j.state,
    j.runtime_seconds,
    js.health_score,
    js.nfs_ratio,
    js.used_gpu,
    js.had_swap,
    js.is_anomaly,
    c.name as cluster_name
FROM jobs j
LEFT JOIN job_summary js ON j.job_id = js.job_id
LEFT JOIN clusters c ON js.cluster_id = c.cluster_id
WHERE j.end_time IS NOT NULL
ORDER BY j.end_time DESC;

-- Filesystem trends
CREATE VIEW IF NOT EXISTS v_filesystem_trends AS
SELECT 
    path,
    used_percent,
    fill_rate_bytes_per_day / (1024*1024*1024) as fill_rate_gb_per_day,
    first_derivative * 86400 / (1024*1024*1024) as rate_gb_per_day,
    second_derivative * 86400 * 86400 / (1024*1024*1024) as acceleration_gb_per_day2,
    days_until_full,
    timestamp
FROM filesystems
WHERE timestamp > datetime('now', '-24 hours')
ORDER BY path, timestamp;

-- User failure rates
CREATE VIEW IF NOT EXISTS v_user_failure_rates AS
SELECT 
    j.user_name,
    COUNT(*) as total_jobs,
    SUM(CASE WHEN j.state = 'COMPLETED' AND js.health_score > 0.5 THEN 1 ELSE 0 END) as successful_jobs,
    AVG(js.health_score) as avg_health,
    AVG(js.nfs_ratio) as avg_nfs_ratio
FROM jobs j
LEFT JOIN job_summary js ON j.job_id = js.job_id
WHERE j.end_time > datetime('now', '-30 days')
GROUP BY j.user_name
HAVING total_jobs >= 5
ORDER BY avg_health ASC;
