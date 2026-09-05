# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


import os
import re
import time
import logging

__timestamp__ = str(time.strftime("%Y%m%d_%H%M%S"))
__pyppbox_root__ = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
__log_dir__ = os.path.join(__pyppbox_root__, "data/logs").replace(os.sep, '/')
__log_txt_path__ = os.path.join(__log_dir__, "log_" + __timestamp__ + ".txt")
__max_age__ = 86400 * 1 # 1 DAY

# Global toggles
__terminal_log__ = True
__file_log__ = False
__logger__ = None

# Remove old logs
def cleanup_old_logs():
    """Delete pyppbox-named regular log files whose modification time is over one day old.

    Only ``log_YYYYMMDD_HHMMSS.txt`` names in the internal log directory are eligible;
    symlinks and other filenames are skipped. Create the directory if absent.
    Returns None. Called automatically at import only when file logging is enabled;
    calling this function directly performs cleanup regardless of that toggle.
    """
    global __log_dir__, __max_age__
    if os.path.exists(__log_dir__):
        for filename in os.listdir(__log_dir__):
            if not re.fullmatch(r"log_\d{8}_\d{6}\.txt", filename):
                continue
            path = os.path.join(__log_dir__, filename)
            if not os.path.isfile(path) or os.path.islink(path):
                continue
            filestamp = os.stat(path).st_mtime
            if  filestamp < time.time() - __max_age__:
                os.remove(path)
    else: os.makedirs(__log_dir__)

# follow the enviroment variable to disable file logging
def set_file_log_from_env():
    """Refresh the file-log flag from ``PYPPBOX_DISABLE_FILE_LOG``.

    Values 1/true/yes/on disable it and 0/false/no/off enable it, case-insensitively.
    Missing or unrecognized values leave the flag unchanged. Returns None. Logger
    creation happens at module import; changing this flag later does not create or
    remove a file handler.
    """
    global __file_log__
    if os.environ.get("PYPPBOX_DISABLE_FILE_LOG", "").lower() in ("1", "true", "yes", "on"):
        __file_log__ = False
    elif os.environ.get("PYPPBOX_DISABLE_FILE_LOG", "").lower() in ("0", "false", "no", "off"):
        __file_log__ = True

set_file_log_from_env()

# Initial logger
if __file_log__:
    cleanup_old_logs()
    logging.basicConfig(
        filename=__log_txt_path__,
        filemode='a',
        format='%(asctime)s %(levelname)-3s %(message)-3s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
    with open(__log_txt_path__, 'w+') as log_txt:
        log_txt.write("-------------------------------------------------")
        log_txt.write("-------------------------------------------------\n")
        log_txt.write("#################################################")
        log_txt.write("#################################################\n")
        log_txt.write("-------------------------------------------------")
        log_txt.write("-------------------------------------------------\n")
    __logger__ = logging.getLogger(__name__)
    __logger__.info(": Here we go!")


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
    if __logger__: __logger__.warning(msg)

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
    if __logger__: __logger__.info(msg)

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
    if __logger__: __logger__.error(msg)

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
    """Disable pyppbox's terminal-log default and update the process environment.

    Set ``PYPPBOX_DISABLE_TERMINAL_LOG=1`` so subsequently launched children
    inherit the choice. Returns None. Explicit per-call terminal-log overrides,
    ordinary print calls, and messages from third-party libraries are unaffected.
    """
    global __terminal_log__
    __terminal_log__ = False
    os.environ['PYPPBOX_DISABLE_TERMINAL_LOG'] = "1"

def enable_terminal_log():
    """Enable pyppbox's terminal-log default and update the process environment.

    Set ``PYPPBOX_DISABLE_TERMINAL_LOG=0`` so subsequently launched children
    inherit the choice. Returns None. Explicit per-call terminal-log overrides,
    ordinary print calls, and messages from third-party libraries are unaffected.
    """
    global __terminal_log__
    __terminal_log__ = True
    os.environ['PYPPBOX_DISABLE_TERMINAL_LOG'] = "0"

def get_terminal_log_status():
    """Return the current default for pyppbox terminal logging.

    Returns
    -------
    bool
        True when enabled; False when disabled.
    """
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
