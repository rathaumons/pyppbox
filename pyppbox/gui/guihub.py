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
from pyppbox.utils.commontools import (getAbsPathFDS, getBool, 
                                       joinFPathFull, getGlobalRootDir)


current_dir = os.path.dirname(__file__)
ui_tmp = joinFPathFull(current_dir, "tmp/ui.tmp")
input_tmp = joinFPathFull(current_dir, "tmp/input.tmp")
default_cfg = joinFPathFull(getGlobalRootDir(), 'config/cfg')
default_cfg_keyword = "NaN"

def loadUITMP():
    # Internal function
    cfg_dir = None
    with open(ui_tmp) as ui_tmp_file:
        lines = ui_tmp_file.read().splitlines()
        if lines[0] != default_cfg_keyword:
            cfg_dir = str(lines[0])
            cfg_dir = getAbsPathFDS(cfg_dir)
    return cfg_dir

def writeUITMP(cfg_dir):
    # Internal function
    if cfg_dir == default_cfg:
        cfg_dir = default_cfg_keyword
    with open(ui_tmp, "w+") as ui_tmp_file:
        ui_tmp_file.write(cfg_dir + "\n")

def loadInputTMP():
    # Internal function
    input_video = ""
    force_hd = False
    with open(input_tmp) as input_tmp_file:
        lines = input_tmp_file.read().splitlines()
        if len(lines) > 1:
            input_video = str(lines[0])
            force_hd = getBool(str(lines[1]))
    return input_video, force_hd

def writeInputTMP(input_video, force_hd):
    # Internal function
    with open(input_tmp, "w+") as input_tmp_file:
        input_tmp_file.write(str(input_video) + "\n")
        input_tmp_file.write(str(force_hd) + "\n")

