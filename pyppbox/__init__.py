"""

A toolbox for people detecting, tracking, and re-identifying

    * * * * * * * * * * * * * * * * * * * * * * * *
    *  Copyright (C) 2025 UMONS-Numediart GPLV3+  *
    * * * * * * * * * * * * * * * * * * * * * * * *

>>> pyppbox.launchGUI()  # Launch GUI
>>> pyppbox.docs()  # Read online documentation
>>> pyppbox.github()  # Go to our GitHub
>>> pyppbox.disable_terminal_log()  # Disable all terminal logs
>>> pyppbox.enable_terminal_log()  # Enable all terminal logs

"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                           #
#   pyppbox: Toolbox for people detecting, tracking, and re-identifying.    #
#   Copyright (C) 2025 UMONS-Numediart                                      #
#                                                                           #
#   This program is free software: you can redistribute it and/or modify    #
#   it under the terms of the GNU General Public License as published by    #
#   the Free Software Foundation, either version 3 of the License, or       #
#   (at your option) any later version.                                     #
#                                                                           #
#   This program is distributed in the hope that it will be useful,         #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#   GNU General Public License for more details.                            #
#                                                                           #
#   You should have received a copy of the GNU General Public License       #
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.  #
#                                                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


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


__version__ = '3.12.0'
__author__ = 'Ratha SIV'
__description__ = 'Toolbox for people detecting, tracking, and re-identifying.'
__homepage__ = 'https://rathaumons.github.io/pyppbox'
__url__ = 'https://github.com/rathaumons/pyppbox.git'
