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


import os
import cv2
import numpy as np
from pathlib import Path 
from contextlib import redirect_stdout
from functools import wraps
from threading import RLock


def getVersionString():
    """
    :meta private:
    """
    version_py = joinFPathFull(getGlobalRootDir(), '__init__.py')
    with open(version_py) as version_file:
        for line in version_file.read().splitlines():
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            msg = "Unable to find version string."
            raise RuntimeError(msg)

def getCVMat(img, to_rgb=False):
    """
    :meta private:
    """
    if isinstance(img, (str, os.PathLike)):
        path = getAbsPathFDS(img)
        if not os.path.isfile(path):
            raise ValueError(f"getCVMat() -> Image file does not exist: '{path}'")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"getCVMat() -> Cannot decode image: '{path}'")
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.size == 0:
        raise ValueError("getCVMat() -> Expected a non-empty HWC image array.")
    if to_rgb:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            raise ValueError(f"getCVMat() -> Cannot convert image to RGB: {e}") from e
    return img

def replaceLine(file_name, line_num, text):
    """
    :meta private:
    """
    lines = open(file_name, 'r').readlines()
    lines[line_num-1] = text + "\n"
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

def isExist(path):
    """
    :meta private:
    """
    path = os.path.abspath(path).replace(os.sep, '/')
    return os.path.exists(path)

def getAbsPathFDS(input):
    """
    :meta private:
    """
    abspath = os.path.abspath(input).replace(os.sep, '/')
    return abspath

def getAdaptiveAbsPathFDS(from_here, input):
    """
    :meta private:
    """
    abspath = getAbsPathFDS(joinFPathFull(from_here, input))
    return abspath

def extendPathFDS(main_path, what_to_extend):
    """
    :meta private:
    """
    abspath = os.path.join(main_path, what_to_extend).replace(os.sep, '/')
    return abspath

def normalizePathFDS(main_path, what_to_normalize):
    """
    :meta private:
    """
    path = getAdaptiveAbsPathFDS(getGlobalRootDir(), what_to_normalize)
    if main_path.replace(os.sep, '/')[:2] == what_to_normalize.replace(os.sep, '/')[:2]:
        tmp = os.path.relpath(what_to_normalize, main_path).replace(os.sep, '/')
        if tmp[:2] != "..":
            path = tmp
    return path

def joinFPathFull(main, to_join):
    """
    :meta private:
    """
    return os.path.join(main, to_join).replace(os.sep, '/')

def getFileName(input):
    """
    :meta private:
    """
    return Path(input).name

def getGlobalRootDir():
    """
    :meta private:
    """
    current_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(current_dir, os.pardir)).replace(os.sep, '/')

def getParentDir(file_abs):
    """
    :meta private:
    """
    current_dir = os.path.dirname(file_abs)
    return os.path.abspath(os.path.join(current_dir, os.pardir)).replace(os.sep, '/')

def getAncestorDir(file_abs, num_of_gen=0):
    """
    :meta private:
    """
    ancestor_dir = os.path.dirname(file_abs)
    gen_count = 0
    while gen_count < int(num_of_gen):
        gen_count += 1
        ancestor_dir = os.path.abspath(os.path.join(ancestor_dir, os.pardir))
    return ancestor_dir.replace(os.sep, '/')

def getBool(input_string):
    """
    :meta private:
    """
    res = False
    if input_string.lower() == "true":
        res = True
    elif input_string.lower() == "false":
        res = False
    else:
        raise ValueError("getBool() -> Can't convert {} to a boolean.".format(input_string))
    return res

def getFloat(input_string, default_val=0.0, ignore_raise=True):
    """
    :meta private:
    """
    res = default_val
    try:
        res = float(input_string)
    except ValueError:
        msg = "The input can't be converted to float."
        if ignore_raise:
            print("IGNORE RAISE : " + msg)
        else:
            raise ValueError(msg)
    return res

def getInt(input_string, default_val=0, ignore_raise=True):
    """
    :meta private:
    """
    res = default_val
    try:
        res = int(input_string)
    except ValueError:
        msg = "The input can't be converted to int."
        if ignore_raise:
            print("IGNORE RAISE : " + msg)
        else:
            raise ValueError(msg)
    return res

def get2Dlist(input_string):
    """
    :meta private:
    """
    input_string = input_string.replace("[", "")
    input_string = input_string.replace("]", "")
    input_string = input_string.replace(" ", "")
    input_list = input_string.split(",")
    return [int(float(input_list[0])), int(float(input_list[1]))]

def to_xywh(box_xyxy):
    """
    :meta private:
    """
    box_xywh = box_xyxy.copy()
    box_xywh[2] = box_xywh[2] - box_xywh[0]
    box_xywh[3] = box_xywh[3] - box_xywh[1]
    return box_xywh

def to_xyxy(box_xywh):
    """
    :meta private:
    """
    ret = box_xywh.copy()
    ret[2:] += ret[:2]
    return ret

_silencer_lock = RLock()


def silencer(func):
    """
    :meta private:
    """
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        # Serialize nested/decorated calls; stdout redirection is still process-wide.
        # Inference code should use backend verbosity controls instead.
        with _silencer_lock, open(os.devnull, 'w') as quiet, redirect_stdout(quiet):
            return func(*args, **kwargs)
    return func_wrapper

def github():
    """
    :meta private:
    """
    import webbrowser
    webbrowser.open('https://github.com/rathaumons/pyppbox.git')

def docs():
    """
    :meta private:
    """
    import webbrowser
    webbrowser.open('https://rathaumons.github.io/pyppbox')

def getTimestamp(format="%Y%m%d_%H%M%S"):
    """
    :meta private:
    """
    import time
    timestamp = time.strftime(format)
    return str(timestamp)
