# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


import os
import re
import time
import logging
import sys
from threading import RLock
from uuid import uuid4

from .runtime import get_runtime_dir

__timestamp__ = str(time.strftime("%Y%m%d_%H%M%S"))
__pyppbox_root__ = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
__log_dir__ = os.path.join(get_runtime_dir(), 'logs')
__log_txt_path__ = None
__max_age__ = 86400 * 1 # 1 DAY

# Global toggles
__terminal_log__ = True
__file_log__ = False
__logger__ = logging.getLogger('pyppbox')
# Keep warnings silent when neither the host nor pyppbox has an output handler.
# Host-installed handlers, levels, propagation, and the root logger stay intact.
if not any(getattr(handler, '_pyppbox_null', False) for handler in __logger__.handlers):
    _null = logging.NullHandler()
    _null._pyppbox_null = True
    __logger__.addHandler(_null)
if not hasattr(__logger__, '_pyppbox_file_lock'):
    __logger__._pyppbox_file_lock = RLock()
_file_lock = __logger__._pyppbox_file_lock


def _after_fork():
    global _file_lock
    _file_lock = RLock()
    __logger__._pyppbox_file_lock = _file_lock


if hasattr(os, 'register_at_fork') and not getattr(__logger__, '_pyppbox_at_fork', False):
    os.register_at_fork(after_in_child=_after_fork)
    __logger__._pyppbox_at_fork = True

# Remove old logs
def cleanup_old_logs():
    """Remove pyppbox logs older than one day, skipping symlinks and known live owners."""
    global __log_dir__, __max_age__
    try:
        filenames = os.listdir(__log_dir__)
    except OSError:
        return
    for filename in filenames:
        match = re.fullmatch(r"log_\d{8}_\d{6}(?:_(\d+)_[0-9a-f]{8})?\.txt", filename)
        if not match:
            continue
        path = os.path.join(__log_dir__, filename)
        try:
            if not os.path.isfile(path) or os.path.islink(path) or os.stat(path).st_mtime >= time.time() - __max_age__:
                continue
            pid = int(match[1]) if match[1] else None
            if pid == os.getpid():
                continue
            if pid is not None and os.name == 'posix':
                try:
                    os.kill(pid, 0)  # Existence probe only; never signal a process.
                except ProcessLookupError:
                    pass
                else:
                    continue
            os.remove(path)
        except (OSError, OverflowError):
            continue


def _remove_file_handlers():
    for handler in __logger__.handlers[:]:
        if getattr(handler, '_pyppbox_file', False):
            __logger__.removeHandler(handler)
            try:
                handler.close()
            except OSError:
                pass
    if getattr(__logger__, '_pyppbox_file_level', False):
        if __logger__.level == logging.INFO:
            __logger__.setLevel(logging.NOTSET)
        __logger__._pyppbox_file_level = False


def _file_failure(error):
    global __file_log__
    with _file_lock:
        if not __file_log__:
            return
        __file_log__ = False
        _remove_file_handlers()
    message = f'PYPPBOX : File logging disabled after an I/O error: {error}'
    if __terminal_log__:
        print(message)
    __logger__.warning(message)


class _FileHandler(logging.FileHandler):
    def emit(self, record):
        if self.stream is None or self._pyppbox_pid != os.getpid():
            return
        try:
            super().emit(record)
        except OSError as error:
            self._pyppbox_error = str(error)

    def handleError(self, record):
        # Avoid acquiring configuration locks while logging holds the handler lock.
        self._pyppbox_error = str(sys.exc_info()[1])


def _ensure_file_handler():
    global __log_txt_path__
    if not __file_log__:
        return
    failure = None
    with _file_lock:
        if not __file_log__:
            return
        for handler in __logger__.handlers[:]:
            if getattr(handler, '_pyppbox_file', False):
                if (handler._pyppbox_pid == os.getpid() and handler.stream is not None
                        and os.path.dirname(handler.baseFilename) == __log_dir__):
                    __log_txt_path__ = handler.baseFilename
                    return handler
                __logger__.removeHandler(handler)
                try:
                    handler.close()
                except OSError:
                    pass
        try:
            os.makedirs(__log_dir__, mode=0o700, exist_ok=True)
            cleanup_old_logs()
            path = os.path.join(__log_dir__, f'log_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}_{uuid4().hex[:8]}.txt')
            handler = _FileHandler(path, mode='x', encoding='utf-8')
            handler._pyppbox_file = True
            handler._pyppbox_pid = os.getpid()
            handler._pyppbox_error = None
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-3s %(message)s', datefmt='%H:%M:%S'))
            __logger__.addHandler(handler)
            __log_txt_path__ = handler.baseFilename
            return handler
        except OSError as error:
            failure = str(error)
    if failure is not None:
        _file_failure(failure)


def _emit_log(level, msg):
    handler = _ensure_file_handler()
    __logger__.log(level, msg)
    if handler is not None and handler._pyppbox_error is not None:
        _file_failure(handler._pyppbox_error)

# follow the enviroment variable to disable file logging
def set_file_log_from_env():
    """Refresh the file-log flag from ``PYPPBOX_DISABLE_FILE_LOG``.

    Values 1/true/yes/on disable it and 0/false/no/off enable it, case-insensitively.
    Missing or unrecognized values leave the flag unchanged. Returns None. Enabling
    opens a uniquely named UTF-8 user log on the next message; disabling closes only
    pyppbox-owned file handlers. Root/host handlers are never configured or removed.
    Refresh after changing the runtime-directory override to select a new location.
    """
    global __file_log__, __log_dir__
    with _file_lock:
        if os.environ.get("PYPPBOX_DISABLE_FILE_LOG", "").lower() in ("1", "true", "yes", "on"):
            __file_log__ = False
        elif os.environ.get("PYPPBOX_DISABLE_FILE_LOG", "").lower() in ("0", "false", "no", "off"):
            __file_log__ = True
        __log_dir__ = os.path.join(get_runtime_dir(), 'logs')
        if __file_log__:
            if __logger__.level == logging.NOTSET:
                __logger__.setLevel(logging.INFO)
                __logger__._pyppbox_file_level = True
        else:
            _remove_file_handlers()


#############################################################################


def add_warning_log(msg, terminal_log=None, add_new_line=True):
    """
    :meta private:
    """
    global __logger__, __terminal_log__
    # Respect explicit caller override; otherwise use current global toggle
    if terminal_log is None:
        terminal_log = __terminal_log__
    if terminal_log: print(msg)
    if add_new_line: msg = ': \n' + str(msg)
    else: msg = ': ' + str(msg)
    _emit_log(logging.WARNING, msg)

def add_info_log(msg, terminal_log=None, add_new_line=False):
    """
    :meta private:
    """
    global __logger__, __terminal_log__
    if terminal_log is None:
        terminal_log = __terminal_log__
    if terminal_log: print(msg)
    if add_new_line: msg = ': \n' + str(msg)
    else: msg = ': ' + str(msg)
    _emit_log(logging.INFO, msg)

def add_error_log(msg, terminal_log=None, add_new_line=True):
    """
    :meta private:
    """
    global __logger__, __terminal_log__
    if terminal_log is None:
        terminal_log = __terminal_log__
    if terminal_log: print(msg)
    if add_new_line: msg = ': \n' + str(msg)
    else: msg = ': ' + str(msg)
    _emit_log(logging.ERROR, msg)

def ignore_this_logger(name, level=logging.ERROR):
    """
    :meta private:
    """
    logger_to_ignore = logging.getLogger(name)
    logger_to_ignore.setLevel(level)

def disable_this_logger(name, level=logging.ERROR):
    """
    :meta private:
    """
    ignore_this_logger(name=name, level=level)
    logger_to_disable = logging.getLogger(name)
    logger_to_disable.disabled = True

def disable_other_loggers():
    """
    :meta private:
    """
    for name, _ in logging.root.manager.loggerDict.items():
        wanted_list = (" yolo ultralytics pyppbox_ultralytics deepsort " + 
                       " facenet torchreid pyppbox_torchreid tensorflow ")
        if name.lower() in wanted_list:
            disable_this_logger(name=name, level=logging.ERROR)

def disable_terminal_log():
    """Disable terminal logging and pass the setting to child processes."""
    global __terminal_log__
    __terminal_log__ = False
    os.environ['PYPPBOX_DISABLE_TERMINAL_LOG'] = "1"

def enable_terminal_log():
    """Enable terminal logging and pass the setting to child processes."""
    global __terminal_log__
    __terminal_log__ = True
    os.environ['PYPPBOX_DISABLE_TERMINAL_LOG'] = "0"

def get_terminal_log_status():
    """Return whether terminal logging is enabled."""
    global __terminal_log__
    return __terminal_log__

def set_terminal_log_from_env():
    """
    Set the terminal logging status according to the environment variable
    :code:`PYPPBOX_DISABLE_TERMINAL_LOG`.
    """
    global __terminal_log__
    if os.environ.get("PYPPBOX_DISABLE_TERMINAL_LOG", "").lower() in ("1", "true", "yes", "on"):
        __terminal_log__ = False
    elif os.environ.get("PYPPBOX_DISABLE_TERMINAL_LOG", "").lower() in ("0", "false", "no", "off"):
        __terminal_log__ = True

def get_env():
    """Return a copy of the current process environment.

    Returns
    -------
    dict[str, str]
        Independent environment mapping suitable for a child process.
    """
    return os.environ.copy()


set_terminal_log_from_env()
set_file_log_from_env()
