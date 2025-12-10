"""
NOMADE CLI

Command-line interface for NOMADE monitoring and analysis.

Commands:
    collect     Run collectors once or continuously
    analyze     Analyze collected data
    status      Show system status
    alerts      Show and manage alerts
"""

import json
import logging
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import toml

from nomade.collectors.base import registry
from nomade.collectors.disk import DiskCollector
from nomade.collectors.slurm import SlurmCollector
from nomade.analysis.derivatives import (
    DerivativeAnalyzer,
    analyze_disk_trend,
    AlertLevel,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('nomade')


def load_config(config_path: Path) -> dict[str, Any]:
    """Load TOML configuration file."""
    if not config_path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        return toml.load(f)


def get_db_path(config: dict[str, Any]) -> Path:
    """Get database path from config."""
    data_dir = Path(config.get('general', {}).get('data_dir', '/var/lib/nomade'))
    return data_dir / 'nomade.db'


@click.group()
@click.option('-c', '--config', 'config_path', 
              type=click.Path(exists=True),
              default='/etc/nomade/nomade.toml',
              help='Path to config file')
@click.option('-v', '--verbose', is_flag=True, help='Enable debug logging')
@click.pass_context
def cli(ctx: click.Context, config_path: str, verbose: bool) -> None:
    """NOMADE - NOde MAnagement DEvice
    
    Lightweight HPC monitoring and prediction tool.
    """
    ctx.ensure_object(dict)
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Try to load config, but don't fail if not found
    try:
        ctx.obj['config'] = load_config(Path(config_path))
        ctx.obj['config_path'] = config_path
    except click.ClickException:
        ctx.obj['config'] = {}
        ctx.obj['config_path'] = None


@cli.command()
@click.option('--collector', '-C', multiple=True, help='Specific collectors to run')
@click.option('--once', is_flag=True, help='Run once and exit')
@click.option('--interval', '-i', type=int, default=60, help='Collection interval (seconds)')
@click.option('--db', type=click.Path(), help='Database path override')
@click.pass_context
def collect(ctx: click.Context, collector: tuple, once: bool, interval: int, db: str) -> None:
    """Run data collectors.
    
    By default, runs all enabled collectors continuously.
    Use --once to run a single collection cycle.
    """
    config = ctx.obj['config']
    
    # Determine database path
    if db:
        db_path = Path(db)
    else:
        db_path = get_db_path(config)
    
    click.echo(f"Database: {db_path}")
    
    # Initialize collectors
    collectors = []
    
    # Disk collector
    disk_config = config.get('collectors', {}).get('disk', {})
    if not collector or 'disk' in collector:
        if disk_config.get('enabled', True):
            collectors.append(DiskCollector(disk_config, db_path))
    
    # SLURM collector
    slurm_config = config.get('collectors', {}).get('slurm', {})
    if not collector or 'slurm' in collector:
        if slurm_config.get('enabled', True):
            collectors.append(SlurmCollector(slurm_config, db_path))
    
    if not collectors:
        raise click.ClickException("No collectors enabled")
    
    click.echo(f"Running collectors: {[c.name for c in collectors]}")
    
    if once:
        # Single collection cycle
        for c in collectors:
            result = c.run()
            status = click.style('✓', fg='green') if result.success else click.style('✗', fg='red')
            click.echo(f"  {status} {c.name}: {result.records_collected} records")
    else:
        # Continuous collection
        click.echo(f"Starting continuous collection (interval: {interval}s)")
        click.echo("Press Ctrl+C to stop")
        
        try:
            while True:
                for c in collectors:
                    result = c.run()
                    status = '✓' if result.success else '✗'
                    click.echo(f"[{datetime.now():%H:%M:%S}] {status} {c.name}: {result.records_collected} records")
                
                time.sleep(interval)
        except KeyboardInterrupt:
            click.echo("\nStopping collectors")


@cli.command()
@click.option('--path', '-p', default='/localscratch', help='Filesystem path to analyze')
@click.option('--hours', '-h', type=int, default=24, help='Hours of history')
@click.option('--limit-gb', type=float, help='Disk limit in GB for projection')
@click.option('--db', type=click.Path(), help='Database path override')
@click.pass_context
def analyze(ctx: click.Context, path: str, hours: int, limit_gb: float, db: str) -> None:
    """Analyze filesystem trends using derivatives.
    
    Shows current trend, rate of change, and projections.
    """
    config = ctx.obj['config']
    
    if db:
        db_path = Path(db)
    else:
        db_path = get_db_path(config)
    
    if not db_path.exists():
        raise click.ClickException(f"Database not found: {db_path}")
    
    # Get historical data
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    rows = conn.execute(
        """
        SELECT timestamp, used_bytes, used_percent, total_bytes
        FROM filesystems
        WHERE path = ?
          AND timestamp > datetime('now', ?)
        ORDER BY timestamp ASC
        """,
        (path, f'-{hours} hours')
    ).fetchall()
    
    if not rows:
        raise click.ClickException(f"No data found for {path}")
    
    # Convert to history format
    history = [dict(row) for row in rows]
    
    # Determine limit
    limit_bytes = None
    if limit_gb:
        limit_bytes = int(limit_gb * 1e9)
    elif history:
        limit_bytes = history[-1]['total_bytes']
    
    # Analyze
    analysis = analyze_disk_trend(history, limit_bytes=limit_bytes)
    
    # Display results
    click.echo()
    click.echo(click.style(f"═══ Analysis: {path} ═══", bold=True))
    click.echo(f"  Records:     {analysis.n_points}")
    click.echo(f"  Time span:   {analysis.time_span_hours:.1f} hours")
    click.echo()
    
    # Current state
    current_gb = analysis.current_value / 1e9
    total_gb = limit_bytes / 1e9 if limit_bytes else 0
    pct = (current_gb / total_gb * 100) if total_gb else 0
    
    click.echo(f"  Current:     {current_gb:.2f} GB / {total_gb:.2f} GB ({pct:.1f}%)")
    
    # Trend
    trend_colors = {
        'stable': 'green',
        'increasing_linear': 'yellow',
        'decreasing_linear': 'cyan',
        'accelerating_growth': 'red',
        'decelerating_growth': 'yellow',
        'accelerating_decline': 'cyan',
        'decelerating_decline': 'green',
        'unknown': 'white',
    }
    trend_color = trend_colors.get(analysis.trend.value, 'white')
    click.echo(f"  Trend:       {click.style(analysis.trend.value, fg=trend_color)}")
    
    # Derivatives
    if analysis.first_derivative:
        rate_gb = analysis.first_derivative / 1e9
        direction = "↑" if rate_gb > 0 else "↓" if rate_gb < 0 else "→"
        click.echo(f"  Rate:        {direction} {abs(rate_gb):.4f} GB/day")
    
    if analysis.second_derivative:
        accel_gb = analysis.second_derivative / 1e9
        direction = "↑↑" if accel_gb > 0 else "↓↓" if accel_gb < 0 else "→→"
        click.echo(f"  Accel:       {direction} {abs(accel_gb):.6f} GB/day²")
    
    # Projections
    click.echo()
    if analysis.projected_value_1d:
        proj_1d_gb = analysis.projected_value_1d / 1e9
        click.echo(f"  In 1 day:    {proj_1d_gb:.2f} GB")
    
    if analysis.projected_value_7d:
        proj_7d_gb = analysis.projected_value_7d / 1e9
        click.echo(f"  In 7 days:   {proj_7d_gb:.2f} GB")
    
    if analysis.days_until_limit:
        click.echo(f"  Days until full: {click.style(f'{analysis.days_until_limit:.1f}', fg='red')}")
    
    # Alert level
    click.echo()
    alert_colors = {
        'none': 'green',
        'info': 'blue',
        'warning': 'yellow',
        'critical': 'red',
    }
    alert_color = alert_colors.get(analysis.alert_level.value, 'white')
    click.echo(f"  Alert:       {click.style(analysis.alert_level.value.upper(), fg=alert_color)}")
    click.echo()


@cli.command()
@click.option('--db', type=click.Path(), help='Database path override')
@click.pass_context
def status(ctx: click.Context, db: str) -> None:
    """Show system status overview."""
    config = ctx.obj['config']
    
    if db:
        db_path = Path(db)
    else:
        db_path = get_db_path(config)
    
    if not db_path.exists():
        raise click.ClickException(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    click.echo()
    click.echo(click.style("═══ NOMADE Status ═══", bold=True))
    click.echo()
    
    # Filesystem status
    click.echo(click.style("Filesystems:", bold=True))
    fs_rows = conn.execute(
        """
        SELECT path, 
               round(used_bytes/1e9, 2) as used_gb,
               round(total_bytes/1e9, 2) as total_gb,
               round(used_percent, 1) as pct,
               timestamp
        FROM filesystems f1
        WHERE timestamp = (
            SELECT MAX(timestamp) FROM filesystems f2 WHERE f2.path = f1.path
        )
        ORDER BY path
        """
    ).fetchall()
    
    for row in fs_rows:
        pct = row['pct']
        color = 'green' if pct < 70 else 'yellow' if pct < 85 else 'red'
        bar_len = int(pct / 5)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        click.echo(f"  {row['path']:<20} [{bar}] {click.style(f'{pct}%', fg=color):>6} ({row['used_gb']}/{row['total_gb']} GB)")
    
    click.echo()
    
    # Queue status
    click.echo(click.style("Queue:", bold=True))
    queue_rows = conn.execute(
        """
        SELECT partition, pending_jobs, running_jobs, total_jobs, timestamp
        FROM queue_state q1
        WHERE timestamp = (
            SELECT MAX(timestamp) FROM queue_state q2 WHERE q2.partition = q1.partition
        )
        ORDER BY partition
        """
    ).fetchall()
    
    if queue_rows:
        for row in queue_rows:
            click.echo(f"  {row['partition']:<15} Running: {row['running_jobs']:>3}  Pending: {row['pending_jobs']:>3}")
    else:
        click.echo("  No queue data")
    
    click.echo()
    
    # Recent collection stats
    click.echo(click.style("Collection:", bold=True))
    collection_rows = conn.execute(
        """
        SELECT collector, 
               COUNT(*) as runs,
               SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
               MAX(completed_at) as last_run
        FROM collection_log
        WHERE started_at > datetime('now', '-24 hours')
        GROUP BY collector
        """
    ).fetchall()
    
    if collection_rows:
        for row in collection_rows:
            success_rate = (row['successes'] / row['runs'] * 100) if row['runs'] else 0
            color = 'green' if success_rate == 100 else 'yellow' if success_rate > 90 else 'red'
            click.echo(f"  {row['collector']:<15} {row['runs']:>3} runs  {click.style(f'{success_rate:.0f}% success', fg=color)}")
    else:
        click.echo("  No collection data")
    
    click.echo()


@cli.command()
@click.option('--db', type=click.Path(), help='Database path override')
@click.option('--unresolved', is_flag=True, help='Show only unresolved alerts')
@click.option('--severity', type=click.Choice(['info', 'warning', 'critical']), help='Filter by severity')
@click.pass_context
def alerts(ctx: click.Context, db: str, unresolved: bool, severity: str) -> None:
    """Show and manage alerts."""
    config = ctx.obj['config']
    
    if db:
        db_path = Path(db)
    else:
        db_path = get_db_path(config)
    
    if not db_path.exists():
        raise click.ClickException(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Build query
    query = "SELECT * FROM alerts WHERE 1=1"
    params = []
    
    if unresolved:
        query += " AND resolved = 0"
    
    if severity:
        query += " AND severity = ?"
        params.append(severity)
    
    query += " ORDER BY timestamp DESC LIMIT 20"
    
    rows = conn.execute(query, params).fetchall()
    
    click.echo()
    click.echo(click.style("═══ Alerts ═══", bold=True))
    click.echo()
    
    if not rows:
        click.echo("  No alerts found")
        click.echo()
        return
    
    severity_colors = {
        'info': 'blue',
        'warning': 'yellow',
        'critical': 'red',
    }
    
    for row in rows:
        color = severity_colors.get(row['severity'], 'white')
        resolved = '✓' if row['resolved'] else '○'
        
        click.echo(f"  {resolved} [{click.style(row['severity'].upper(), fg=color)}] {row['timestamp']}")
        click.echo(f"    {row['message']}")
        if row['source']:
            click.echo(f"    Source: {row['source']}")
        click.echo()


@cli.command()
@click.pass_context
def version(ctx: click.Context) -> None:
    """Show version information."""
    click.echo("NOMADE v0.1.0")
    click.echo("NOde MAnagement DEvice")


def main() -> None:
    """Entry point for CLI."""
    cli(obj={})


if __name__ == '__main__':
    main()
