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

__version__ = '1.0b7'
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
from pyppbox.config import MyCFGIO

cfgIO = MyCFGIO()

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
    from pyunpack import Archive
    Archive(os.path.join(cfgIO.mstruct.cfg_dir, 'cfg.7z')).extractall(cfgIO.mstruct.cfg_dir)
    print("Reset successfully!")

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
    programName = os.path.join(cfgIO.mstruct.root_dir, "GUILauncher.cmd")
    p = sp.Popen([programName])
    stdout, stderr = p.communicate()

def showCFCNote():
    note = ("Note:\n"
            " * pyppbox does not check and verify your config\n"
            " * pyppbox can reset all config by using 'pyppbox.resetCFG()'\n"
            " * pyppbox can also be configured through GUI: 'pyppbox.launchGUI()'")
    print(note)

