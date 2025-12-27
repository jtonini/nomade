"""
Threshold-based alert triggering for collectors.

Monitors collected data and dispatches alerts when thresholds are exceeded.

Configuration example (nomade.toml):
    [alerts.thresholds.disk]
    warning = 80    # Warn at 80% usage
    critical = 95   # Critical at 95%
    
    [alerts.thresholds.nfs]
    retrans_warning = 1.0    # Retransmit % warning
    latency_critical = 100   # RTT ms critical
    
    [alerts.thresholds.gpu]
    memory_warning = 90
    temperature_critical = 85
"""

import logging
from datetime import datetime
from typing import Any, Optional

from .dispatcher import send_alert, get_dispatcher, init_dispatcher

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_THRESHOLDS = {
    'disk': {
        'used_percent_warning': 80,
        'used_percent_critical': 95,
    },
    'nfs': {
        'retrans_percent_warning': 1.0,
        'retrans_percent_critical': 5.0,
        'avg_rtt_ms_warning': 50,
        'avg_rtt_ms_critical': 100,
    },
    'gpu': {
        'memory_percent_warning': 90,
        'memory_percent_critical': 98,
        'temperature_warning': 80,
        'temperature_critical': 90,
    },
    'node': {
        'load_warning': 0.9,      # load / n_cpus
        'load_critical': 1.5,
        'memory_percent_warning': 90,
        'memory_percent_critical': 98,
    },
    'job': {
        'failure_rate_warning': 0.2,   # 20% failure rate
        'failure_rate_critical': 0.5,  # 50% failure rate
    }
}


class ThresholdChecker:
    """Check collected data against thresholds and trigger alerts."""
    
    def __init__(self, config: dict):
        """
        Initialize with config.
        
        Config structure:
            [alerts]
            enabled = true
            
            [alerts.thresholds.disk]
            used_percent_warning = 80
            used_percent_critical = 95
        """
        self.config = config
        self.enabled = config.get('alerts', {}).get('enabled', True)
        
        # Merge user thresholds with defaults
        self.thresholds = DEFAULT_THRESHOLDS.copy()
        user_thresholds = config.get('alerts', {}).get('thresholds', {})
        for category, values in user_thresholds.items():
            if category in self.thresholds:
                self.thresholds[category].update(values)
            else:
                self.thresholds[category] = values
        
        # Initialize dispatcher if not already done
        if not get_dispatcher():
            init_dispatcher(config)
    
    def check(self, collector_name: str, data: list[dict], host: str = None) -> list[dict]:
        """
        Check collected data against thresholds.
        
        Args:
            collector_name: Name of collector (disk, nfs, gpu, etc.)
            data: List of collected data dicts
            host: Hostname for alerts
        
        Returns:
            List of triggered alerts
        """
        if not self.enabled:
            return []
        
        alerts = []
        
        for item in data:
            item_alerts = self._check_item(collector_name, item, host)
            alerts.extend(item_alerts)
        
        return alerts
    
    def _check_item(self, collector_name: str, item: dict, host: str) -> list[dict]:
        """Check a single data item against thresholds."""
        alerts = []
        thresholds = self.thresholds.get(collector_name, {})
        
        for key, value in item.items():
            if not isinstance(value, (int, float)):
                continue
            
            # Check for critical threshold
            critical_key = f"{key}_critical"
            if critical_key in thresholds:
                if value >= thresholds[critical_key]:
                    alert = self._create_alert(
                        severity='critical',
                        source=collector_name,
                        host=host,
                        metric=key,
                        value=value,
                        threshold=thresholds[critical_key],
                        item=item
                    )
                    alerts.append(alert)
                    continue  # Don't also trigger warning
            
            # Check for warning threshold
            warning_key = f"{key}_warning"
            if warning_key in thresholds:
                if value >= thresholds[warning_key]:
                    alert = self._create_alert(
                        severity='warning',
                        source=collector_name,
                        host=host,
                        metric=key,
                        value=value,
                        threshold=thresholds[warning_key],
                        item=item
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _create_alert(
        self,
        severity: str,
        source: str,
        host: str,
        metric: str,
        value: float,
        threshold: float,
        item: dict
    ) -> dict:
        """Create and dispatch an alert."""
        
        # Generate human-readable message
        message = self._format_message(source, metric, value, threshold, item)
        
        alert = {
            'severity': severity,
            'source': source,
            'host': host or 'unknown',
            'message': message,
            'details': {
                'metric': metric,
                'value': value,
                'threshold': threshold,
                'item': item
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Dispatch alert
        send_alert(severity=alert["severity"], source=alert["source"], message=alert["message"], host=alert["host"], details=alert["details"])
        
        logger.info(f"Alert triggered: {severity} - {source} - {message}")
        
        return alert
    
    def _format_message(
        self,
        source: str,
        metric: str,
        value: float,
        threshold: float,
        item: dict
    ) -> str:
        """Generate human-readable alert message."""
        
        # Source-specific formatting
        if source == 'disk':
            path = item.get('path', 'unknown')
            return f"Disk {path} at {value:.1f}% (threshold: {threshold}%)"
        
        elif source == 'nfs':
            mount = item.get('mount_point', 'unknown')
            if 'retrans' in metric:
                return f"NFS {mount} retransmit rate {value:.2f}% (threshold: {threshold}%)"
            elif 'rtt' in metric:
                return f"NFS {mount} latency {value:.1f}ms (threshold: {threshold}ms)"
            else:
                return f"NFS {mount} {metric}={value:.2f} (threshold: {threshold})"
        
        elif source == 'gpu':
            gpu_id = item.get('index', item.get('gpu_id', '?'))
            if 'memory' in metric:
                return f"GPU {gpu_id} memory at {value:.1f}% (threshold: {threshold}%)"
            elif 'temp' in metric:
                return f"GPU {gpu_id} temperature {value:.0f}°C (threshold: {threshold}°C)"
            else:
                return f"GPU {gpu_id} {metric}={value:.2f} (threshold: {threshold})"
        
        elif source == 'node':
            node_name = item.get('hostname', item.get('node', 'unknown'))
            if 'load' in metric:
                return f"Node {node_name} load {value:.2f} (threshold: {threshold})"
            elif 'memory' in metric:
                return f"Node {node_name} memory at {value:.1f}% (threshold: {threshold}%)"
            else:
                return f"Node {node_name} {metric}={value:.2f} (threshold: {threshold})"
        
        else:
            return f"{source}: {metric}={value:.2f} exceeded threshold {threshold}"


def check_and_alert(collector_name: str, data: list[dict], config: dict, host: str = None) -> list[dict]:
    """
    Convenience function to check data and trigger alerts.
    
    Args:
        collector_name: Name of collector
        data: Collected data
        config: Full nomade config dict
        host: Hostname for alerts
    
    Returns:
        List of triggered alerts
    
    Example:
        from nomade.alerts.thresholds import check_and_alert
        
        data = disk_collector.collect()
        alerts = check_and_alert('disk', data, config, host='compute-01')
    """
    checker = ThresholdChecker(config)
    return checker.check(collector_name, data, host)


class PredictiveChecker:
    """
    Predictive alerts using derivative analysis.
    
    Instead of just threshold alerts (disk > 95%), this predicts
    when resources will be exhausted and alerts early.
    """
    
    def __init__(self, config: dict):
        self.config = config.get('alerts', {}).get('predictive', {})
        self.enabled = self.config.get('enabled', True)
        
        # Default predictive thresholds
        self.days_warning = self.config.get('days_until_full_warning', 7)
        self.days_critical = self.config.get('days_until_full_critical', 1)
        self.accel_warning = self.config.get('acceleration_warning', 0.05)
        self.accel_critical = self.config.get('acceleration_critical', 0.10)
    
    def check_disk_trend(
        self, 
        history: list[dict], 
        limit_bytes: int,
        host: str = None,
        path: str = None
    ) -> Optional[dict]:
        """
        Check disk usage trend and alert if approaching full.
        
        Args:
            history: List of {'timestamp': datetime, 'used_bytes': int}
            limit_bytes: Total disk capacity
            host: Hostname for alerts
            path: Filesystem path
        
        Returns:
            Alert dict if triggered, None otherwise
        """
        if not self.enabled or len(history) < 3:
            return None
        
        try:
            from nomade.analysis.derivatives import analyze_disk_trend, AlertLevel
            
            analysis = analyze_disk_trend(history, limit_bytes=limit_bytes)
            
            alert = None
            
            # Check days until full
            if analysis.days_until_limit is not None:
                if analysis.days_until_limit <= self.days_critical:
                    alert = self._create_alert(
                        severity='critical',
                        source='disk_prediction',
                        host=host,
                        message=f"Disk {path or 'unknown'} will be FULL in {analysis.days_until_limit:.1f} days!",
                        details={
                            'path': path,
                            'days_until_full': analysis.days_until_limit,
                            'current_value': analysis.current_value,
                            'rate_per_day': analysis.first_derivative * 86400 if analysis.first_derivative else None,
                            'trend': analysis.trend.value
                        }
                    )
                elif analysis.days_until_limit <= self.days_warning:
                    alert = self._create_alert(
                        severity='warning',
                        source='disk_prediction',
                        host=host,
                        message=f"Disk {path or 'unknown'} will be full in {analysis.days_until_limit:.1f} days",
                        details={
                            'path': path,
                            'days_until_full': analysis.days_until_limit,
                            'current_value': analysis.current_value,
                            'rate_per_day': analysis.first_derivative * 86400 if analysis.first_derivative else None,
                            'trend': analysis.trend.value
                        }
                    )
            
            # Check acceleration (accelerating growth is dangerous)
            if not alert and analysis.second_derivative is not None:
                d2_per_day = analysis.second_derivative * 86400 * 86400  # per day²
                
                if analysis.first_derivative and analysis.first_derivative > 0:
                    d1_per_day = analysis.first_derivative * 86400
                    accel_ratio = abs(d2_per_day) / (abs(d1_per_day) + 0.001)
                    
                    if d2_per_day > 0 and accel_ratio > self.accel_critical:
                        alert = self._create_alert(
                            severity='critical',
                            source='disk_prediction',
                            host=host,
                            message=f"Disk {path or 'unknown'} growth ACCELERATING rapidly!",
                            details={
                                'path': path,
                                'acceleration_ratio': accel_ratio,
                                'rate_per_day_gb': d1_per_day / 1e9,
                                'trend': analysis.trend.value
                            }
                        )
                    elif d2_per_day > 0 and accel_ratio > self.accel_warning:
                        alert = self._create_alert(
                            severity='warning',
                            source='disk_prediction',
                            host=host,
                            message=f"Disk {path or 'unknown'} growth is accelerating",
                            details={
                                'path': path,
                                'acceleration_ratio': accel_ratio,
                                'rate_per_day_gb': d1_per_day / 1e9,
                                'trend': analysis.trend.value
                            }
                        )
            
            return alert
            
        except ImportError:
            logger.debug("Derivative analysis not available")
            return None
        except Exception as e:
            logger.error(f"Predictive check failed: {e}")
            return None
    
    def _create_alert(self, severity: str, source: str, host: str, message: str, details: dict) -> dict:
        """Create alert dict and dispatch."""
        alert = {
            'severity': severity,
            'source': source,
            'host': host or 'unknown',
            'message': message,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        send_alert(
            severity=severity,
            source=source,
            message=message,
            host=host,
            details=details
        )
        
        logger.info(f"Predictive alert: {severity} - {message}")
        
        return alert


def check_disk_prediction(
    history: list[dict],
    limit_bytes: int,
    config: dict,
    host: str = None,
    path: str = None
) -> Optional[dict]:
    """
    Convenience function for predictive disk checking.
    
    Example:
        history = [
            {'timestamp': datetime(2024, 1, 1), 'used_bytes': 100e9},
            {'timestamp': datetime(2024, 1, 2), 'used_bytes': 120e9},
            {'timestamp': datetime(2024, 1, 3), 'used_bytes': 150e9},  # Accelerating!
        ]
        alert = check_disk_prediction(history, limit_bytes=200e9, config={}, path='/scratch')
    """
    checker = PredictiveChecker(config)
    return checker.check_disk_trend(history, limit_bytes, host, path)
