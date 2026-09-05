# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""

A toolbox for people detecting, tracking, and re-identifying

>>> pyppbox.launchGUI()  # Launch GUI
>>> pyppbox.docs()  # Read online documentation
>>> pyppbox.github()  # Go to our GitHub
>>> pyppbox.disable_terminal_log()  # Disable all terminal logs
>>> pyppbox.enable_terminal_log()  # Enable all terminal logs

"""



import importlib

_exports = {
    # utils
    "github": ("pyppbox.utils.commontools", "github"),
    "docs": ("pyppbox.utils.commontools", "docs"),
    # gui
    "launchGUI": ("pyppbox.gui.guitools", "launchGUI"),
    "showMainConfig": ("pyppbox.gui.guitools", "showMainConfig"),
    "showAllDTConfig": ("pyppbox.gui.guitools", "showAllDTConfig"),
    "showAllRIConfig": ("pyppbox.gui.guitools", "showAllRIConfig"),
    "showAllTKConfig": ("pyppbox.gui.guitools", "showAllTKConfig"),
    "useInternalConfigDir": ("pyppbox.gui.guitools", "useInternalConfigDir"),
    "useThisConfigDir": ("pyppbox.gui.guitools", "useThisConfigDir"),
    "resetInternalConfig": ("pyppbox.gui.guitools", "resetInternalConfig"),
    "generateConfig": ("pyppbox.gui.guitools", "generateConfig"),
    # logtools
    "disable_terminal_log": ("pyppbox.utils.logtools", "disable_terminal_log"),
    "enable_terminal_log": ("pyppbox.utils.logtools", "enable_terminal_log"),
}

def __getattr__(name):
    if name in _exports:
        mod_path, attr = _exports[name]
        mod = importlib.import_module(mod_path)
        obj = getattr(mod, attr)
        globals()[name] = obj
        return obj
    raise AttributeError(f"---PYPPBOX : Couldn't find attribute '{name}'")


__version__ = '3.14.0'
__author__ = 'Ratha SIV'
__description__ = 'Toolbox for people detecting, tracking, and re-identifying.'
__homepage__ = 'https://rathaumons.github.io/pyppbox'
__url__ = 'https://github.com/rathaumons/pyppbox.git'
