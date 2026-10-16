# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


from contextlib import contextmanager
import locale
import os
from pathlib import Path
import tempfile

from pyppbox.utils.commontools import getAbsPathFDS, getBool, joinFPathFull, getGlobalRootDir
from pyppbox.utils.logtools import add_warning_log
from pyppbox.utils.runtime import get_gui_state_dir, get_runtime_dir


current_dir = os.path.dirname(__file__)
ui_tmp = str(Path(get_gui_state_dir()) / 'ui.tmp')
input_tmp = str(Path(get_gui_state_dir()) / 'input.tmp')
default_cfg = joinFPathFull(getGlobalRootDir(), 'config/cfg')
default_cfg_keyword = "NaN"

def _read_lines(path):
    try:
        with open(path, 'rb') as source:
            content = source.read(65537)
        if len(content) > 65536:
            return []
        try:
            text = content.decode('utf-8-sig')
        except UnicodeError:
            text = content.decode(locale.getpreferredencoding(False))
        return text.splitlines()
    except (OSError, UnicodeError):
        return []


def _paths(name):
    if os.environ.get('PYPPBOX_GUI_SESSION_DIR'):
        # Active launches never read another launch's remembered preferences.
        return [Path(get_gui_state_dir()) / name]
    return [Path(get_runtime_dir()) / 'gui' / name, Path(current_dir) / 'tmp' / name]


def _config(paths):
    for path in paths:
        lines = _read_lines(path)
        if lines and lines[0] and '\x00' not in lines[0]:
            return None if lines[0] == default_cfg_keyword else getAbsPathFDS(lines[0])
    return None


def _input(paths):
    for path in paths:
        lines = _read_lines(path)
        if len(lines) >= 2 and '\x00' not in lines[0] and lines[1].lower() in ('true', 'false'):
            return lines[0], getBool(lines[1])
    return '', False


def _atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         prefix='.' + path.name + '-', dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _write(name, text):
    active = Path(get_gui_state_dir()) / name
    _atomic_write(active, text)
    remembered = Path(get_runtime_dir()) / 'gui' / name
    if active != remembered:
        try:
            _atomic_write(remembered, text)
        except OSError as error:
            add_warning_log(f'GUI : Could not remember {name}: {error}')


def loadUITMP():
    """Return the launch's config directory, or None for internal/missing state.

    Without an active session, remembered user state precedes legacy gui/tmp files.
    Installed compatibility files are never written. Empty/malformed state is safe.
    """
    return _config(_paths('ui.tmp'))


def writeUITMP(cfg_dir):
    """Atomically store the config selection in writable GUI state; return None."""
    _write('ui.tmp', _config_text(cfg_dir))


def _config_text(cfg_dir):
    value = default_cfg_keyword if cfg_dir is None or os.path.abspath(cfg_dir) == os.path.abspath(default_cfg) else str(cfg_dir)
    if any(char in value for char in ('\n', '\r', '\x00')):
        raise ValueError('GUI configuration paths must fit on one line.')
    return value + '\n'


def loadInputTMP():
    """Return (video path, force-HD flag) from this launch or remembered/legacy state.

    Missing/truncated/malformed state returns an empty path and False. An active
    session reads only its own files, without falling back to shared preferences.
    """
    return _input(_paths('input.tmp'))


def writeInputTMP(input_video, force_hd):
    """Atomically store input selection and remember it for future launches."""
    value = str(input_video)
    if any(char in value for char in ('\n', '\r', '\x00')):
        raise ValueError('GUI input paths must fit on one line.')
    _write('input.tmp', value + '\n' + str(getBool(str(force_hd))) + '\n')


@contextmanager
def _gui_session(cfg_dir=None, *, env=None, reuse=False):
    """Yield an isolated child environment; clean up only this owned session.

    Reuse an inherited directory only for a launcher started by launchGUI(). The
    caller's environment is never mutated. New launches seed preferences once,
    then read their own state. Selected configuration files are not copied/moved.
    """
    child_env = dict(os.environ if env is None else env)
    if reuse and child_env.get('PYPPBOX_GUI_SESSION_DIR'):
        yield child_env
        return
    parent = Path(get_runtime_dir()) / 'sessions'
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gui-', dir=parent) as directory:
        session = Path(directory)
        if session.resolve().parent != parent.resolve():
            raise RuntimeError('GUI session directory is outside the runtime directory.')
        remembered = Path(get_runtime_dir()) / 'gui'
        legacy = Path(current_dir) / 'tmp'
        selection = cfg_dir if cfg_dir is not None else _config([remembered / 'ui.tmp', legacy / 'ui.tmp'])
        video, force_hd = _input([remembered / 'input.tmp', legacy / 'input.tmp'])
        text = _config_text(selection)
        _atomic_write(session / 'ui.tmp', text)
        _atomic_write(session / 'input.tmp', video + '\n' + str(force_hd) + '\n')
        if cfg_dir is not None:
            try:
                _atomic_write(remembered / 'ui.tmp', text)
            except OSError as error:
                add_warning_log(f'GUI : Could not remember ui.tmp: {error}')
        child_env['PYPPBOX_GUI_SESSION_DIR'] = str(session)
        yield child_env

