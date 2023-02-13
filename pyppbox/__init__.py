"""
    pyppbox: Toolbox for people detecting, tracking, and re-identifying.
    Copyright (C) 2022 UMONS-Numediart

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


from __future__ import division, print_function, absolute_import

# Info

__version__ = '1.1b3'
__author__ = 'Ratha SIV (rathaROG)'
__homepage__ = 'https://github.com/rathaumons'
__description__ = 'Python toolbox for people tracking system'
__url__ = 'https://github.com/rathaumons/pyppbox.git'


# Important

from pyppbox import utils
from pyppbox import dt_yolocv
from pyppbox import tk_centroid
from pyppbox import tk_sort
from pyppbox import tk_deepsort
from pyppbox import ri_deepreid
from pyppbox import ri_facenet
from pyppbox import tmp
from pyppbox import config
from pyppbox import ppboxmng
from pyppbox import ui_centroid
from pyppbox import ui_gt
from pyppbox import ui_deepreid
from pyppbox import ui_deepsort
from pyppbox import ui_facenet
from pyppbox import ui_sort
from pyppbox import ui_yolo
from pyppbox import uilauncher


# Extra

import os
import subprocess as sp
from pyppbox.config import MyCFGIO as GlobalCFGIO
from pyppbox.localconfig import MyCFGIO as LocalCFGIO
from pyppbox.utils.mytools import getAbsPathFDS, joinFPathFull, writeUITMP

cfgIO = GlobalCFGIO()
cfg_mode = 0
cfg_dir = ""

def switchToLocalCFG(_cfg_dir):
    global cfgIO, cfg_mode, cfg_dir
    cfg_mode = 1
    cfg_dir = getAbsPathFDS(_cfg_dir)
    cfgIO = None
    cfgIO = LocalCFGIO(cfg_dir)
    print("Switched to LOCAL mode!")

def switchToGlobalCFG():
    global cfgIO, cfg_mode
    cfg_mode = 0
    cfgIO = None
    cfgIO = GlobalCFGIO()
    print("Switched to GLOBAL mode!")

def showMainCFGTemplate():
    print(cfgIO.headers.mainHeader())

def showDetectorsCFGTemplate():
    print(cfgIO.headers.detectorHeader())

def showTrackersCFGTemplate():
    print(cfgIO.headers.trackerHeader())

def showReIDersCFGTemplate():
    print(cfgIO.headers.reiderHeader())

def showMainCFG():
    print(cfgIO.loadDocument(cfgIO.mstruct.main_yaml))

def showDetectorsCFG():
    print(cfgIO.loadAllDocuments(cfgIO.mstruct.detector_yaml))

def showTrackersCFG():
    print(cfgIO.loadAllDocuments(cfgIO.mstruct.tracker_yaml))

def showReIDersCFG():
    print(cfgIO.loadAllDocuments(cfgIO.mstruct.reider_yaml))

def resetCFG():
    if cfg_mode == 0:
        from pyunpack import Archive
        Archive(os.path.join(cfgIO.mstruct.cfg_dir, 'cfg.7z')).extractall(cfgIO.mstruct.cfg_dir)
        print("Reset successfully!")
    else:
        print("Reminder: pyppbox was set to LOCAL mode.")

def setMainCFG(input):
    cfgIO.dumpMainWithHeader(input)
    print("Saved successfully!")
    showCFCNote()

def setDetectorsCFG(input):
    cfgIO.dumpDetectorsWithHeader(input)
    print("Saved successfully!")
    showCFCNote()

def setTrackersCFG(input):
    cfgIO.dumpTrackersWithHeader(input)
    print("Saved successfully!")
    showCFCNote()

def setReIDersCFG(input):
    cfgIO.dumpReidersWithHeader(input)
    print("Saved successfully!")
    showCFCNote()

def setInputVideo(input, force_hd=False):
    from pyppbox.config import MainCFG
    mainCFG = MainCFG()
    mainCFG.set(cfgIO.loadDocument(cfgIO.mstruct.main_yaml))
    new_data = {'detector': mainCFG.detector, 
                'tracker': mainCFG.tracker, 
                'reider': mainCFG.reider, 
                'input_video': os.path.abspath(input).replace(os.sep, '/'), 
                'force_hd': force_hd}
    cfgIO.dumpMainWithHeader(new_data)
    print("Saved successfully!")
    showCFCNote()

def launchGUI():
    uitmp = joinFPathFull(cfgIO.mstruct.global_root_dir, "tmp/ui.tmp")
    if cfg_mode == 0:
        writeUITMP(uitmp, cfg_mode, cfg_dir)
        programName = joinFPathFull(cfgIO.mstruct.global_root_dir, "GUILauncher.cmd")
        p = sp.Popen([programName])
        stdout, stderr = p.communicate()
    else:
        # Option 0: Use ui.tmp file
        writeUITMP(uitmp, cfg_mode, cfg_dir)
        programName = joinFPathFull(cfgIO.mstruct.global_root_dir, "GUILauncher.cmd")
        p = sp.Popen([programName])
        stdout, stderr = p.communicate()

        # Option 1: Use Parser with the rewritten cmd file
        '''
        command = (
            "@echo off \n"
            "setlocal \n"
            "cd /d %~dp0 \n"
            "python -W ignore uilauncher.py --cfgmode 1 --cfgdir \"" + cfg_dir + "\" \n"
            "pause \n"
            )
        programName = joinFPathFull(cfgIO.mstruct.global_root_dir, "GUILauncher1.cmd")
        with open(programName, "w") as guilauncher1_cmd:
            guilauncher1_cmd.write(cfgIO.headers.copyrightCMDHeader())
            guilauncher1_cmd.write(command)
        p = sp.Popen([programName])
        stdout, stderr = p.communicate()
        '''

def showCFCNote():
    note = ("Note:\n"
            " * pyppbox does not check and verify your config\n"
            " * pyppbox can reset the global config by using 'pyppbox.resetCFG()'\n"
            " * pyppbox can also be configured through GUI: 'pyppbox.launchGUI()'")
    print(note)
