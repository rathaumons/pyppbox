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


import argparse

from pyppbox.utils.gttools import convertStringToNPL
from pyppbox.utils.commontools import to_xyxy
from pyppbox.utils.persontools import findRepspoint, findRepspointBB

# step 0: filter conf
def filter_conf(input_line, splitter=',', min_conf=0.25, max_conf=1.00):
    """
    :meta private:
    """
    res = False
    spl_indexs = [i for i, char in enumerate(input_line) if char == splitter]
    conf = float(input_line[spl_indexs[5] + 1:spl_indexs[6]])
    if conf >= min_conf and conf <= max_conf:
        res = True
    return res

# step 1: remove the last 4 columns (from coma 6th)
def remove_from_splitter(input_line, splitter=',', from_spl=6, with_newline=False):
    """
    :meta private:
    """
    spl_indexs = [i for i, char in enumerate(input_line) if char == splitter]
    res = input_line[:spl_indexs[from_spl - 1]]
    if with_newline: res = res + "\n"
    return res

# step 2: replace 2nd colum with repspoint and realID
def replace_column_with(input_line, splitter=',', column_index=2, with_str="(0, 0)	XXX", use_numid=True, with_newline=False):
    """
    :meta private:
    """
    spl_indexs = [i for i, char in enumerate(input_line) if char == splitter]
    if use_numid:
        num_id = input_line[spl_indexs[0] + 1:spl_indexs[1]]
        with_str = with_str.replace("XXX", str(num_id))
        # print(f"with_str = {with_str}")
    res = input_line[0:spl_indexs[column_index - 2] + 1] + with_str + input_line[spl_indexs[column_index - 1]:]
    if with_newline: res = res + "\n"
    return res

# step 3: finalize
def finalize(input_line, splitter=',', find_repspoint=True, alt_repspoint=False, alt_repspoint_top=True, with_newline=False):
    """
    :meta private:
    """
    spl_indexs = [i for i, char in enumerate(input_line) if char == splitter]
    input_line = input_line[0:spl_indexs[0]] + '\t' + input_line[spl_indexs[0] + 1:]
    bbox = input_line[spl_indexs[2] + 1:]
    bbox = bbox.replace(',', ' ')
    bbox_xyxy_np = to_xyxy(convertStringToNPL(bbox))
    res = input_line[0:spl_indexs[2]] + "\t[" + bbox + ']\t' + str(bbox_xyxy_np)
    if find_repspoint:
        repspoint = "(0, 0)"
        if alt_repspoint: repspoint = findRepspointBB(bbox_xyxy_np, prefer_top=alt_repspoint_top)
        else: repspoint = findRepspoint(bbox_xyxy_np, calibrate_weight=0.25)
        res = res.replace("(0, 0)", str(repspoint))
    res = res.replace("    ", " ")
    res = res.replace("   ", " ")
    res = res.replace("  ", " ")
    res = res.replace("[ ", "[")
    if with_newline: res = res + "\n"
    return res

# AIO
def convert(input_file="gt.txt", 
            output_file="gt_pyppbox.txt", 
            splitter=',',
            use_numid=True,  
            min_conf=0.0, 
            max_conf=1.0, 
            find_repspoint=True, 
            alt_repspoint=False, 
            alt_repspoint_top=True):
    """Convert ground-truth of MOT Challenge text file to pyppbox format. 

    Parameters
    ----------
    input_file : str, default="gt.txt"
        Input MOT Challenge text file.
    output_file : str, default="gt_pyppbox.txt"
        Output pyppbox text file.
    splitter : str, default=','
        Column or data splitter.
    use_numid : bool, default=False
        Whether to use number id as real id.
    min_conf : float, default=0.0
        Filter for minimum conf.
    max_conf : float, default=1.0
        Filter for maximum conf.
    find_repspoint : bool, default=True
        Whether to calculate repspoint.
    alt_repspoint : bool, default=False
        Whether to use alternative repspoint.
    alt_repspoint_top : bool, default=True
        Whether y is the top for alternative repspoint.
    """
    
    with open(output_file, 'w') as pyppboxtxt:
        with open(input_file, 'r') as mottxt:
            lines = mottxt.readlines()
            for line in lines:
                if filter_conf(line, splitter=splitter, min_conf=min_conf, max_conf=max_conf):
                    line = line.replace("\r", "")
                    line = line.replace("\n", "")
                    line = remove_from_splitter(line, splitter=splitter)
                    line = replace_column_with(line, splitter=splitter, use_numid=use_numid)
                    line = finalize(line, 
                                    splitter=splitter, 
                                    find_repspoint=find_repspoint, 
                                    alt_repspoint=alt_repspoint, 
                                    alt_repspoint_top=alt_repspoint_top, 
                                    with_newline=True)
                    pyppboxtxt.write(line)


###################################################################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert ground-truth of MOT Challenge text file to pyppbox format.")

    parser.add_argument("input_file", type=str, default="gt.txt", help="Input MOT Challenge text file")
    parser.add_argument("output_file", type=str, default="gt_pyppbox.txt", help="Output pyppbox text file")
    parser.add_argument("--splitter", type=str, default=',', help="Column or data splitter")
    parser.add_argument("--use-numid", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=1, help="Whether to use number id as real id")
    parser.add_argument("--min-conf", type=float, default='0.0', help="Filter for minimum conf")
    parser.add_argument("--max-conf", type=float, default='1.0', help="Filter for maximum conf")
    parser.add_argument("--find-repspoint", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=1, help="Whether to calculate repspoint")
    parser.add_argument("--alt-repspoint", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=0, help="Whether to use alternative repspoint")
    parser.add_argument("--alt-repspoint-top", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=1, help="Whether y is the top for alternative repspoint")

    args = parser.parse_args()
    convert(input_file=args.input_file, 
            output_file=args.output_file, 
            splitter=args.splitter, 
            use_numid=args.use_numid, 
            min_conf=args.min_conf, 
            max_conf=args.max_conf, 
            find_repspoint=args.find_repspoint, 
            alt_repspoint=args.alt_repspoint, 
            alt_repspoint_top=args.alt_repspoint_top)

