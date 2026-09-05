# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


import os
import sys
import subprocess as sp

from pyppbox.utils.logtools import add_info_log, add_warning_log, add_error_log, get_env
from pyppbox.config.configtools import PYPPBOXStructure, loadDocument, loadDocumentList
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
    cfg_dir : str
        A path of configuration directory.
    """
    global pyppbox_struct, __cfgdir__
    __cfgdir__ = getAbsPathFDS(cfg_dir)
    pyppbox_struct.setCustomCFG(cfg_dir)
    add_warning_log("FYI: This basic method only serves GUI submodule `pyppbox.gui`.")

def useInternalConfigDir():
    """Use the internal config directory, :code:`{pyppbox root}/config/cfg` inside pyppbox package.
    """
    global pyppbox_struct, __cfgdir__
    pyppbox_struct = PYPPBOXStructure()
    __cfgdir__ =  pyppbox_struct.cfg_dir
    add_warning_log("FYI: This basic method only serves GUI submodule `pyppbox.gui`.")

def showMainConfig():
    """Print the configuration mapping from main.yaml.

    Uses the GUI's selected config directory and prints Python representations,
    not serialized JSON. Returns None; this does not change the pipeline configuration.
    """
    print(loadDocument(pyppbox_struct.main_yaml))
    add_warning_log("FYI: This basic method only serves GUI submodule `pyppbox.gui`.")

def showAllDTConfig():
    """Print the list of configuration mappings from detectors.yaml.

    Uses the GUI's selected config directory and prints Python representations,
    not serialized JSON. Returns None; this does not change the pipeline configuration.
    """
    print(loadDocumentList(pyppbox_struct.detectors_yaml))
    add_warning_log("FYI: This basic method only serves GUI submodule `pyppbox.gui`.")

def showAllTKConfig():
    """Print the list of configuration mappings from trackers.yaml.

    Uses the GUI's selected config directory and prints Python representations,
    not serialized JSON. Returns None; this does not change the pipeline configuration.
    """
    print(loadDocumentList(pyppbox_struct.trackers_yaml))
    add_warning_log("FYI: This basic method only serves GUI submodule `pyppbox.gui`.")

def showAllRIConfig():
    """Print the list of configuration mappings from reiders.yaml.

    Uses the GUI's selected config directory and prints Python representations,
    not serialized JSON. Returns None; this does not change the pipeline configuration.
    """
    print(loadDocumentList(pyppbox_struct.reiders_yaml))
    add_warning_log("FYI: This basic method only serves GUI submodule `pyppbox.gui`.")

def resetInternalConfig():
    """Overwrite internal configuration files from the package's bundled ``cfg.zip``.

    This restores the archived defaults on disk, discarding edits to those files.
    It does not reconstruct existing pipeline/model instances. Returns None;
    archive/filesystem errors propagate.
    """
    global pyppbox_struct
    pyppbox_struct = PYPPBOXStructure()
    cfg_zip = os.path.join(pyppbox_struct.cfg_dir, 'cfg.zip')
    import shutil
    shutil.unpack_archive(cfg_zip, pyppbox_struct.cfg_dir)
    add_info_log("Reset successfully!")
    add_warning_log("FYI: This basic method only serves GUI submodule `pyppbox.gui`.")

def launchGUI():
    """Launch the GUI in a child Python process and wait for it to close.

    Write the selected GUI config directory to the GUI state file and pass the
    current process environment to the child. Returns None; no child exit code is
    returned. This does not initialize the standalone pipeline in the calling process.
    """
    writeUITMP(__cfgdir__)
    p = sp.Popen([sys.executable, os.path.join(current_dir, 'ui_launcher.py')], env=get_env())
    p.wait()

def generateConfig(cfg_dir, auto_launch_gui=True):
    """Extract the selected GUI config archive into an existing directory.

    Parameters
    ----------
    cfg_dir : str
        Destination directory, which must exist. Files with matching names can be
        overwritten. The archive is ``cfg.zip`` in the GUI's currently selected
        config directory; a custom source therefore needs that archive too.
    auto_launch_gui : bool
        Defaults to ``True``. Select the destination for the GUI, launch it in a
        child process, and wait for it to close.

    Notes
    -----
    Returns None. A missing destination causes no extraction. Archive and filesystem
    errors propagate. Existing standalone/MT pipeline configurations are not changed.
    """
    if isExist(cfg_dir):
        global pyppbox_struct
        abspath = getAbsPathFDS(cfg_dir)
        cfg_zip = os.path.join(pyppbox_struct.cfg_dir, 'cfg.zip')
        try:
            import shutil
            shutil.unpack_archive(cfg_zip, abspath)
            if auto_launch_gui:
                useThisConfigDir(abspath)
                launchGUI()
        except Exception as e:
            msg = "generateCFG() -> " + str(e)
            add_error_log(msg)
            print(msg)
            raise
