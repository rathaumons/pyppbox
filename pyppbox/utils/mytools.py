from __future__ import division, print_function, absolute_import

import os

from posixpath import abspath
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
