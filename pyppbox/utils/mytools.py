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


import os
from pathlib import Path 


def replaceLine(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num-1] = text + "\n"
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

def getAbsPathFDS(input):
    abspath = os.path.abspath(input).replace(os.sep, '/')
    return abspath

def extendPathFDS(main_path, what_to_extend):
    abspath = os.path.join(main_path, what_to_extend).replace(os.sep, '/')
    return abspath

def normalizePathFDS(main_path, what_to_normalize):
    path = what_to_normalize
    if main_path.replace(os.sep, '/')[:2] == what_to_normalize.replace(os.sep, '/')[:2]:
        path = os.path.relpath(what_to_normalize, main_path).replace(os.sep, '/')
    return path

def joinFPathFull(main, to_join):
    return os.path.join(main, to_join)

def getFileName(input):
    return Path(input).name

def getAncestorDir(from_file, num_of_step=0):
    ancestor_dir = Path(getAbsPathFDS(from_file)).parent
    step_count = 0
    while step_count < int(num_of_step):
        step_count += 1
        ancestor_dir = Path(ancestor_dir).parent
    return getAbsPathFDS(ancestor_dir)

def getBool(input_string):
    res = False
    if input_string.lower() == "true":
        res = True
    elif input_string.lower() == "false":
        res = False
    else:
        raise ValueError("Can't covert {} to a boolean! ---> Return False".format(input_string))
    return res

def get2Dlist(input_string):
    input_string = input_string.replace("[", "")
    input_string = input_string.replace("]", "")
    input_string = input_string.replace(" ", "")
    input_list = input_string.split(",")
    return [int(float(input_list[0])), int(float(input_list[1]))]

def customDumpSingleDoc(output_file, data, header):
    with open(output_file, 'w') as dumping:
        dumping.write(header)
        for key, value in data.items():
            dumping.write('%s: %s\n' % (key, value))

def customDumpMultiDoc(output_file, docs, header):
    with open(output_file, 'w') as dumping:
        dumping.write(header)
        sep_index = 1
        for d in docs:
            for key, value in d.items():
                dumping.write('%s: %s\n' % (key, value))
            if sep_index < len(docs):
                dumping.write("---\n")
                sep_index += 1

def loadUITMP(uitmp):
    cfg_mode = 0
    cfg_dir = ""
    with open(uitmp) as uitmp_file:
        lines = uitmp_file.read().splitlines()
        if len(lines) > 1:
            cfg_mode = int(lines[0])
            cfg_dir = getAbsPathFDS(str(lines[1]))
    return cfg_mode, cfg_dir

def writeUITMP(uitmp, cfg_mode, cfg_dir):
    with open(uitmp, "w") as uitmp_file:
        uitmp_file.write(str(cfg_mode) + "\n")
        uitmp_file.write(cfg_dir + "\n")
