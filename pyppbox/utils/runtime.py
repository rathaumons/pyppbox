# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Backend-free paths for mutable runtime files; querying paths creates nothing."""
import os
from pathlib import Path
import sys


def get_runtime_dir():
    """Return the absolute user-state directory without creating it.

    ``PYPPBOX_RUNTIME_DIR`` overrides the entire directory. Otherwise use
    LocalAppData/pyppbox on Windows, ~/Library/Application Support/pyppbox on macOS,
    or $XDG_STATE_HOME/pyppbox (default ~/.local/state/pyppbox) on other systems.
    Model, dataset, result-export, and configuration paths are unaffected.
    """
    override = os.environ.get('PYPPBOX_RUNTIME_DIR')
    if override:
        return os.path.abspath(Path(override).expanduser())
    if sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData/Local')
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library/Application Support'
    else:
        base = Path(os.environ.get('XDG_STATE_HOME') or Path.home() / '.local/state')
    return os.path.abspath(base.expanduser() / 'pyppbox')


def get_gui_state_dir():
    """Return the active GUI session directory, or the user's remembered-state path.

    Launcher children inherit ``PYPPBOX_GUI_SESSION_DIR``. Direct state helpers
    without an active session use the gui subdirectory of :func:`get_runtime_dir`.
    This function creates no files or directories.
    """
    session = os.environ.get('PYPPBOX_GUI_SESSION_DIR')
    return os.path.abspath(Path(session).expanduser()) if session else str(Path(get_runtime_dir()) / 'gui')
