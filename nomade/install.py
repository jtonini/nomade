"""
NOMADE System Installation

Handles system-wide installation for HPC environments:
- Create nomade user/group
- Install config to /etc/nomade/
- Create data dirs in /var/lib/nomade/
- Install SLURM prolog hook
- Create systemd service
- Setup logrotate
"""

import grp
import os
import pwd
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def get_nomade_bin_path() -> str:
    """Find the installed nomade binary path."""
    # Check if nomade is in PATH
    result = subprocess.run(['which', 'nomade'], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    
    # Check common locations
    candidates = [
        Path('/usr/local/bin/nomade'),
        Path('/usr/bin/nomade'),
        Path.home() / '.local' / 'bin' / 'nomade',
        Path(sys.prefix) / 'bin' / 'nomade',
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    
    # Fallback to python -m
    return f"{sys.executable} -m nomade.cli"

# Installation paths
SYSTEM_CONFIG_DIR = Path("/etc/nomade")
SYSTEM_DATA_DIR = Path("/var/lib/nomade")
SYSTEM_LOG_DIR = Path("/var/log/nomade")
SLURM_PROLOG_DIR = Path("/etc/slurm/prolog.d")
SYSTEMD_DIR = Path("/etc/systemd/system")
LOGROTATE_DIR = Path("/etc/logrotate.d")

# Service user
NOMADE_USER = "nomade"
NOMADE_GROUP = "nomade"




def create_bin_symlink(force: bool = False) -> Optional[Path]:
    """Create /usr/local/bin/nomade symlink to actual binary."""
    target = Path('/usr/local/bin/nomade')
    
    # Find actual nomade location
    actual = None
    candidates = [
        Path('/opt/anaconda/bin/nomade'),
        Path('/opt/conda/bin/nomade'),
        Path('/usr/local/anaconda/bin/nomade'),
    ]
    
    # Also check sys.prefix (where pip installed)
    import sys
    candidates.insert(0, Path(sys.prefix) / 'bin' / 'nomade')
    
    for c in candidates:
        if c.exists():
            actual = c
            break
    
    if not actual:
        return None
    
    if target.exists() or target.is_symlink():
        if force:
            target.unlink()
        else:
            return None
    
    target.symlink_to(actual)
    return target


SYSTEMD_SERVICE = """[Unit]
Description=NOMADE HPC Monitoring Daemon
Documentation=https://github.com/jtonini/nomade
After=network.target slurmd.service

[Service]
Type=simple
User={user}
Group={group}
ExecStart={nomade_bin} collect --interval 60
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure
RestartSec=10

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=nomade

# Security
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={data_dir} {log_dir}
PrivateTmp=true

[Install]
WantedBy=multi-user.target
"""

SYSTEMD_LEARNING_SERVICE = """[Unit]
Description=NOMADE Continuous Learning Daemon
Documentation=https://github.com/jtonini/nomade
After=network.target nomade.service

[Service]
Type=simple
User={user}
Group={group}
ExecStart={nomade_bin} learn --daemon --strategy count --threshold 100
Restart=on-failure
RestartSec=60

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=nomade-learn

# Security
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={data_dir} {log_dir}
PrivateTmp=true

[Install]
WantedBy=multi-user.target
"""

SYSTEMD_TIMER = """[Unit]
Description=NOMADE ML Training Timer
Documentation=https://github.com/jtonini/nomade

[Timer]
OnBootSec=15min
OnUnitActiveSec=6h
RandomizedDelaySec=10min

[Install]
WantedBy=timers.target
"""

SYSTEMD_TRAIN_SERVICE = """[Unit]
Description=NOMADE ML Training (triggered by timer)
Documentation=https://github.com/jtonini/nomade

[Service]
Type=oneshot
User={user}
Group={group}
ExecStart={nomade_bin} learn --force --epochs 100
StandardOutput=journal
StandardError=journal
SyslogIdentifier=nomade-train
"""

LOGROTATE_CONFIG = """/var/log/nomade/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0640 {user} {group}
    sharedscripts
    postrotate
        systemctl reload nomade.service >/dev/null 2>&1 || true
    endscript
}
"""

SLURM_PROLOG_WRAPPER = """#!/bin/bash
# NOMADE SLURM Prolog Hook
# Scores jobs at submission time for risk assessment

# Only run if nomade is installed
if command -v nomade &> /dev/null; then
    /usr/bin/python3 /etc/nomade/prolog.py 2>/dev/null || true
fi

exit 0
"""


def check_root():
    """Check if running as root."""
    return os.geteuid() == 0


def user_exists(username: str) -> bool:
    """Check if user exists."""
    try:
        pwd.getpwnam(username)
        return True
    except KeyError:
        return False


def group_exists(groupname: str) -> bool:
    """Check if group exists."""
    try:
        grp.getgrnam(groupname)
        return True
    except KeyError:
        return False


def create_user_group(user: str = NOMADE_USER, group: str = NOMADE_GROUP) -> dict:
    """Create nomade user and group."""
    results = {'group_created': False, 'user_created': False}
    
    # Create group
    if not group_exists(group):
        subprocess.run(['groupadd', '--system', group], check=True)
        results['group_created'] = True
    
    # Create user
    if not user_exists(user):
        subprocess.run([
            'useradd', '--system',
            '--gid', group,
            '--home-dir', str(SYSTEM_DATA_DIR),
            '--shell', '/sbin/nologin',
            '--comment', 'NOMADE HPC Monitor',
            user
        ], check=True)
        results['user_created'] = True
    
    return results


def create_directories(user: str = NOMADE_USER, group: str = NOMADE_GROUP) -> list:
    """Create system directories with proper permissions."""
    created = []
    
    dirs = [
        (SYSTEM_CONFIG_DIR, 0o755),
        (SYSTEM_DATA_DIR, 0o750),
        (SYSTEM_DATA_DIR / 'models', 0o750),
        (SYSTEM_LOG_DIR, 0o750),
    ]
    
    for dir_path, mode in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        dir_path.chmod(mode)
        if user_exists(user) and group_exists(group):
            shutil.chown(dir_path, user=user, group=group)
        created.append(str(dir_path))
    
    return created


def install_config(source_config: Optional[Path] = None, force: bool = False) -> Path:
    """Install configuration file."""
    config_file = SYSTEM_CONFIG_DIR / 'nomade.toml'
    
    if config_file.exists() and not force:
        return config_file
    
    # Find source config
    if source_config and source_config.exists():
        shutil.copy(source_config, config_file)
    else:
        # Try package default
        try:
            from nomade.config import get_default_config_path
            default = get_default_config_path()
            if default.exists():
                shutil.copy(default, config_file)
        except ImportError:
            pass
    
    # Update paths in config
    if config_file.exists():
        content = config_file.read_text()
        content = content.replace(
            'data_dir = "~/.local/share/nomade"',
            f'data_dir = "{SYSTEM_DATA_DIR}"'
        )
        config_file.write_text(content)
        config_file.chmod(0o644)
    
    return config_file


def install_prolog_hook(force: bool = False) -> Optional[Path]:
    """Install SLURM prolog hook."""
    # Check if SLURM is available
    if not SLURM_PROLOG_DIR.parent.exists():
        return None
    
    SLURM_PROLOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy prolog.py
    prolog_dest = SYSTEM_CONFIG_DIR / 'prolog.py'
    try:
        from nomade.hooks import prolog
        prolog_src = Path(prolog.__file__)
        if prolog_src.exists():
            shutil.copy(prolog_src, prolog_dest)
            prolog_dest.chmod(0o755)
    except ImportError:
        pass
    
    # Create wrapper script
    wrapper = SLURM_PROLOG_DIR / 'nomade_score.sh'
    if not wrapper.exists() or force:
        wrapper.write_text(SLURM_PROLOG_WRAPPER)
        wrapper.chmod(0o755)
        return wrapper
    
    return None


def install_systemd_services(
    user: str = NOMADE_USER, 
    group: str = NOMADE_GROUP,
    force: bool = False
) -> list:
    """Install systemd service files."""
    installed = []
    nomade_bin = get_nomade_bin_path()
    
    services = [
        ('nomade.service', SYSTEMD_SERVICE),
        ('nomade-learn.service', SYSTEMD_LEARNING_SERVICE),
        ('nomade-train.service', SYSTEMD_TRAIN_SERVICE),
        ('nomade-train.timer', SYSTEMD_TIMER),
    ]
    
    for name, template in services:
        service_file = SYSTEMD_DIR / name
        if not service_file.exists() or force:
            content = template.format(
                user=user,
                group=group,
                data_dir=SYSTEM_DATA_DIR,
                log_dir=SYSTEM_LOG_DIR,
                nomade_bin=nomade_bin
            )
            service_file.write_text(content)
            service_file.chmod(0o644)
            installed.append(name)
    
    # Reload systemd
    if installed:
        subprocess.run(['systemctl', 'daemon-reload'], check=False)
    
    return installed


def install_logrotate(user: str = NOMADE_USER, group: str = NOMADE_GROUP) -> Optional[Path]:
    """Install logrotate configuration."""
    if not LOGROTATE_DIR.exists():
        return None
    
    logrotate_file = LOGROTATE_DIR / 'nomade'
    content = LOGROTATE_CONFIG.format(user=user, group=group)
    logrotate_file.write_text(content)
    logrotate_file.chmod(0o644)
    
    return logrotate_file


def add_slurm_group(user: str = NOMADE_USER) -> bool:
    """Add nomade user to slurm group for access to SLURM commands."""
    try:
        # Check if slurm group exists
        grp.getgrnam('slurm')
        subprocess.run(['usermod', '-aG', 'slurm', user], check=True)
        return True
    except (KeyError, subprocess.CalledProcessError):
        return False


def full_system_install(force: bool = False, verbose: bool = True) -> dict:
    """
    Complete system installation.
    
    Returns dict with installation results.
    """
    results = {
        'success': False,
        'user_created': False,
        'group_created': False,
        'directories': [],
        'config': None,
        'prolog': None,
        'services': [],
        'logrotate': None,
        'errors': []
    }
    
    if not check_root():
        results['errors'].append("Must run as root (use sudo)")
        return results
    
    try:
        # 1. Create user/group
        if verbose:
            print("Creating nomade user and group...")
        ug = create_user_group()
        results['user_created'] = ug['user_created']
        results['group_created'] = ug['group_created']
        
        # 2. Create directories
        if verbose:
            print("Creating directories...")
        results['directories'] = create_directories()
        
        # 3. Install config
        if verbose:
            print("Installing configuration...")
        results['config'] = str(install_config(force=force))
        
        # 4. Install prolog hook
        if verbose:
            print("Installing SLURM prolog hook...")
        prolog = install_prolog_hook(force=force)
        results['prolog'] = str(prolog) if prolog else None
        
        # 5. Install systemd services
        if verbose:
            print("Installing systemd services...")
        results['services'] = install_systemd_services(force=force)
        
        # 6. Install logrotate
        if verbose:
            print("Installing logrotate config...")
        logrotate = install_logrotate()
        results['logrotate'] = str(logrotate) if logrotate else None
        
        # 7. Add to slurm group
        if verbose:
            print("Adding nomade user to slurm group...")
        add_slurm_group()
        
        # 8. Create /usr/local/bin symlink if needed
        if verbose:
            print("Creating /usr/local/bin symlink...")
        symlink = create_bin_symlink(force=force)
        if symlink:
            results['symlink'] = str(symlink)
        
        results['success'] = True
        
    except Exception as e:
        results['errors'].append(str(e))
    
    return results


def print_post_install_instructions():
    """Print post-installation instructions."""
    nomade_bin = get_nomade_bin_path()
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    NOMADE System Installation                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Binary:        {nomade_bin:<47} ║
║  Configuration: /etc/nomade/nomade.toml                          ║
║  Data:          /var/lib/nomade/                                 ║
║  Logs:          /var/log/nomade/                                 ║
╚══════════════════════════════════════════════════════════════════╝

Next steps:

  1. Edit configuration:
     sudo nano /etc/nomade/nomade.toml

  2. Start the collection daemon:
     sudo systemctl enable --now nomade.service

  3. (Optional) Enable continuous learning:
     sudo systemctl enable --now nomade-train.timer
     
     OR for daemon mode:
     sudo systemctl enable --now nomade-learn.service

  4. (Optional) Enable SLURM prolog hook:
     Add to /etc/slurm/slurm.conf:
       Prolog=/etc/slurm/prolog.d/nomade_score.sh
     Then: sudo systemctl restart slurmctld

  5. Check status:
     sudo systemctl status nomade
     sudo journalctl -u nomade -f

  6. View dashboard (from head node):
     nomade dashboard --host 0.0.0.0

For module-based environments (Lmod):
  Create module file at /opt/apps/modulefiles/nomade/0.3.0
  See: https://github.com/jtonini/nomade/docs/lmod.md
""")
