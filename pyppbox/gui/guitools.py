# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                           #
#   pyppbox: Toolbox for people detecting, tracking, and re-identifying.    #
#   Copyright (C) 2022 UMONS-Numediart                                      #
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


import os
import sys
import subprocess as sp

from pyppbox.utils.logtools import add_warning_log, add_error_log
from pyppbox.config.configtools import PYPPBOXStructure, loadDocument, loadListDocument
from pyppbox.utils.commontools import getAbsPathFDS, joinFPathFull, isExist
from pyppbox.gui.guihub import writeUITMP

current_dir = os.path.dirname(__file__)
pyppbox_struct = PYPPBOXStructure()
__cfgdir__ =  pyppbox_struct.cfg_dir

def useThisConfigDir(cfg_dir):
    """Use your custom config directory where stores 4 required YAML files:
        - main.yaml, indicates which detector/tracker/reider is used.
        - detectors.yaml, stores all detectors' configurations.
        - trackers.yaml, stores all trackers' configurations.
        - reiders.yaml, stores all reiders' configurations.

    Parameters
    ----------
    config_dir : str
        A path of configuration directory.
    """
    global pyppbox_struct, __cfgdir__
    __cfgdir__ = getAbsPathFDS(cfg_dir)
    pyppbox_struct.setCustomCFG(cfg_dir)
    add_warning_log("FYI: This basic method only serves GUI submodule `pyppbox.gui`.")

def useInternalConfigDir():
    """Use the internal config directory, :code:`{pyppbox root}/confog/cfg` inside pyppbox package.
    """
    global pyppbox_struct
    pyppbox_struct = PYPPBOXStructure()
    add_warning_log("FYI: This basic method only serves GUI submodule `pyppbox.gui`.")

def showMainConfig():
    """Print JSON dictionary of the configurations in main.yaml.
    """
    print(loadDocument(pyppbox_struct.main_yaml))
    add_warning_log("FYI: This basic method only serves GUI submodule `pyppbox.gui`.")

def showAllDTConfig():
    """Print JSON dictionary of the configurations in detectors.yaml.
    """
    print(loadListDocument(pyppbox_struct.detectors_yaml))
    add_warning_log("FYI: This basic method only serves GUI submodule `pyppbox.gui`.")

def showAllTKConfig():
    """Print JSON dictionary of the configurations in trackers.yaml.
    """
    print(loadListDocument(pyppbox_struct.trackers_yaml))
    add_warning_log("FYI: This basic method only serves GUI submodule `pyppbox.gui`.")

def showAllRIConfig():
    """Print JSON dictionary of the configurations in reiders.yaml.
    """
    print(loadListDocument(pyppbox_struct.reiders_yaml))
    add_warning_log("FYI: This basic method only serves GUI submodule `pyppbox.gui`.")

def resetInternalConfig():
    """Reset the internal configurations.
    """
    global pyppbox_struct
    pyppbox_struct = PYPPBOXStructure()
    from pyunpack import Archive
    Archive(os.path.join(pyppbox_struct.cfg_dir, 'cfg.7z')).extractall(pyppbox_struct.cfg_dir)
    print("Reset successfully!")
    add_warning_log("FYI: This basic method only serves GUI submodule `pyppbox.gui`.")

def launchGUI():
    """Launch GUI configuration tool of pyppbox.
    """
    writeUITMP(__cfgdir__)
    p = sp.Popen([sys.executable, os.path.join(current_dir, 'ui_launcher.py')])
    stdout, stderr = p.communicate()

def generateConfig(cfg_dir, auto_launch_gui=True):
    """Generate the 4 required YAML files to a given :obj:`cfg_dir`:
        - main.yaml, indicates which detector/tracker/reider is used.
        - detectors.yaml, stores all detectors' configurations.
        - trackers.yaml, stores all trackers' configurations.
        - reiders.yaml, stores all reiders' configurations.

    Parameters
    ----------
    config_dir : str
        A path of configuration directory.
    auto_launch_gui : bool, default=True
        An indication of whether to load and launch GUI from the :obj:`config_dir`.
    """
    if isExist(cfg_dir):
        abspath = getAbsPathFDS(cfg_dir)
        global pyppbox_struct
        try:
            from pyunpack import Archive
            Archive(os.path.join(pyppbox_struct.cfg_dir, 'cfg.7z')).extractall(abspath)
            try:
                os.remove(joinFPathFull(abspath, 'strings.yaml'))
            except Exception:
                pass
            if auto_launch_gui:
                useThisConfigDir(abspath)
                launchGUI()
        except Exception as e:
            msg = "generateCFG() -> " + str(e)
            add_error_log(msg)
            raise print(msg)
