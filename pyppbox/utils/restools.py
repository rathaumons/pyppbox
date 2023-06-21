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


import numpy as np

from .persontools import Person
from .logtools import add_info_log, add_warning_log
from .commontools import (joinFPathFull, getGlobalRootDir, isExist, 
                          getAbsPathFDS, getTimestamp)


default_dump_dir = joinFPathFull(getGlobalRootDir(), "data/res/")
default_dump_file = joinFPathFull(default_dump_dir, "res.txt")
default_dumpall_file = joinFPathFull(default_dump_dir, "res_all.txt")

class ResIO(object):

    """
    A class used to generate and dump results into text file.

    Attributes
    ----------
    frames : list[str, ...]
        a list of frame indexes.
    people : list[Person, ...]
        A list of object :class:`Person`.
    sorted_people : list[Person, ...]
        A list used to store the sorted list of :attr:`people`.
    """

    def __init__(self):
        self.frames = []
        self.people = []

    def addPerson(self, frame, person):
        """Add a :obj:`frame` index and a :class:`Person` object in that 
        :attr:`frame` index into :attr:`frames` and :attr:`people`.
        
        Parameters
        ----------
        frame : int
            A frame index.
        person : Person
            An object of :class:`Person` class.
        """
        if isinstance(person, Person):
            self.frames.append(str(frame))
            self.people.append(person)
        else:
            raise ValueError("RESIO : addPerson() -> Input 'person' is not valid.")
    
    def addPeople(self, frame, people):
        """Add a frame index and a list of :class:`Person` object in that :obj:`frame` 
        index into :attr:`frames` and :attr:`people`.
        
        Parameters
        ----------
        frame : int
            A frame index.
        people : list[Person, ...]
            A list of object :class:`Person`.
        """
        if isinstance(people, list):
            if len(people) > 0:
                if isinstance(people[0], Person):
                    for person in people:
                        self.frames.append(str(frame))
                        self.people.append(person)
                else:
                    raise ValueError("RESIO : addPeople() -> Input 'people' is not valid.")
    
    def dump(self, dump_dir=default_dump_dir, dump_mode=3, id_mode="deepid"):
        """Dump the result as a text file in a directory with a choice of :code:`"deepid"` or 
        :code:`"faceid"`. Each line represents frame index and a person's details separated by '\\t'.

        Parameters
        ----------
        dump_dir : str, default='{pyppbox root}/data/res'
            A directory where to dump the result text file.
        dump_mode : int, default=3
            Set 1 to dump: frame index, repspoint, deepid/faceid.
            Set 2 to dump: frame index, repspoint, deepid/faceid, box_xywh.
            Set 3 to dump: frame index, repspoint, deepid/faceid, box_xywh, box_xyxy.
        id_mode : str, defualt="deepid"
            Set choice between :code:`"deepid"` and :code:`"faceid"`.
        """
        dump_file = self.__generateFileName__(dump_dir)
        if id_mode != "deepid":
            if str(id_mode).lower() in " deepid faceid ":
                id_mode = str(id_mode).lower()
                add_info_log("-----RESIO : Set id_mode='" + str(id_mode) + "'")
            else :
                add_warning_log("-----RESIO : id_mode='" + str(id_mode) + "' is not recognized.")
                add_warning_log("-----RESIO : Overwite id_mode='" + str(id_mode) + "'.")
        self.__sort_people_by_x__()
        dump_mode = int(dump_mode)
        if dump_mode < 1 and dump_mode > 3:
            add_warning_log("-----RESIO : 'dump_mode' is out of range -> Overwite 'dump_mode=3'")
        with open(dump_file, 'w') as dumpfile:
            for f, p in zip(self.frames, self.sorted_people):
                dump_str = ""
                if dump_mode == 1:
                    if id_mode == "deepid": 
                        tmp_deepid = p.deepid
                        if '%' in tmp_deepid: tmp_deepid = tmp_deepid[:-4]
                        dump_str = f + "\t" + str(p.repspoint) + "\t" + tmp_deepid + "\n"
                    elif id_mode == "faceid": 
                        tmp_faceid = p.faceid
                        if '%' in tmp_faceid: tmp_faceid = tmp_faceid[:-4]
                        dump_str = f + "\t" + str(p.repspoint) + "\t" + tmp_faceid + "\n"
                elif dump_mode == 2:
                    if id_mode == "deepid": 
                        tmp_deepid = p.deepid
                        if '%' in tmp_deepid: tmp_deepid = tmp_deepid[:-4]
                        dump_str = (f + "\t" + str(p.repspoint) + "\t" + tmp_deepid + "\t" + 
                                    str(p.box_xywh) + "\n")
                    elif id_mode == "faceid": 
                        tmp_faceid = p.faceid
                        if '%' in tmp_faceid: tmp_faceid = tmp_faceid[:-4]
                        dump_str = (f + "\t" + str(p.repspoint) + "\t" + tmp_faceid + "\t" + 
                                    str(p.box_xywh) + "\n")
                else:
                    if id_mode == "deepid": 
                        tmp_deepid = p.deepid
                        if '%' in tmp_deepid: tmp_deepid = tmp_deepid[:-4]
                        dump_str = (f + "\t" + str(p.repspoint) + "\t" + tmp_deepid + "\t" + 
                                    str(p.box_xywh) + "\t" + str(p.box_xyxy) + "\n")
                    elif id_mode == "faceid": 
                        tmp_faceid = p.faceid
                        if '%' in tmp_faceid: tmp_faceid = tmp_faceid[:-4]
                        dump_str = (f + "\t" + str(p.repspoint) + "\t" + tmp_faceid + "\t" + 
                                    str(p.box_xywh) + "\t" + str(p.box_xyxy) + "\n")
                dumpfile.write(dump_str)
        add_info_log("-----RESIO : Successfully dump to '" + dump_file + "'")
    
    def dumpAll(self, dump_dir=default_dump_dir, dump_mode=3):
        """Dump the result as a text file in a directory with both deepid and faceid. 
        Each line represents frame index and a person's details separated by '\\t'.

        Parameters
        ----------
        dump_dir : str, default='{pyppbox root}/data/res'
            A directory of where to dump the result text file.
        dump_mode : int, default=3
            Set 1 to dump: frame index, repspoint, deepid, faceid.
            Set 2 to dump: frame index, repspoint, deepid, faceid, box_xywh.
            Set 3 to dump: frame index, repspoint, deepid, faceid, box_xywh, box_xyxy.
        """
        dump_file = self.__generateFileName__(dump_dir)
        dump_mode = int(dump_mode)
        self.__sort_people_by_x__()
        if dump_mode < 1 and dump_mode > 3:
            add_warning_log("-----RESIO : 'dump_mode' is out of range -> Overwite 'dump_mode=3'")
        with open(dump_file, 'w') as dumpfile:
            for f, p in zip(self.frames, self.sorted_people):
                dump_str = ""
                if dump_mode == 1:
                    tmp_deepid = p.deepid
                    tmp_faceid = p.faceid
                    if '%' in tmp_deepid: tmp_deepid = tmp_deepid[:-4]
                    if '%' in tmp_faceid: tmp_faceid = tmp_faceid[:-4]
                    dump_str = (f + "\t" + str(p.repspoint) + "\t" + tmp_deepid + "\t" + 
                                tmp_faceid + "\n")
                elif dump_mode == 2:
                    tmp_deepid = p.deepid
                    tmp_faceid = p.faceid
                    if '%' in tmp_deepid: tmp_deepid = tmp_deepid[:-4]
                    if '%' in tmp_faceid: tmp_faceid = tmp_faceid[:-4]
                    dump_str = (f + "\t" + str(p.repspoint) + "\t" + tmp_deepid + "\t" + 
                                tmp_faceid + "\t" + str(p.box_xywh) + "\n")
                else:
                    tmp_deepid = p.deepid
                    tmp_faceid = p.faceid
                    if '%' in tmp_deepid: tmp_deepid = tmp_deepid[:-4]
                    if '%' in tmp_faceid: tmp_faceid = tmp_faceid[:-4]
                    dump_str = (f + "\t" + str(p.repspoint) + "\t" + tmp_deepid + "\t" + 
                                tmp_faceid + "\t" + str(p.box_xywh) + "\t" + 
                                str(p.box_xyxy) + "\n")
                dumpfile.write(dump_str)
        add_info_log("-----RESIO : Successfully dump to '" + dump_file + "'")

    def __generateFileName__(self, dump_dir=default_dump_dir):
        timestamp = getTimestamp()
        dump_file_name = "res_" + str(timestamp) + "_full.txt"
        if isExist(dump_dir):
            dump_dir = getAbsPathFDS(dump_dir)
        else:
            add_warning_log("-----RESIO : dump_dir='" + str(dump_dir) + "' does not exist!")
            dump_dir = default_dump_dir
            add_warning_log("-----RESIO : Overwrite dump_dir='" + str(dump_dir) + "'.")
        dump_file_name = joinFPathFull(dump_dir, dump_file_name)
        return dump_file_name

    def __sort_people_by_x__(self):
        self.sorted_people = []
        tmp_x = []
        tmp_pp = []
        len_frames = len(self.frames)
        if len_frames > 0:
            current_frame = int(self.frames[0])
            previous_frame = int(self.frames[0])
            for frame, person in zip(self.frames, self.people):
                (x, _) = person.repspoint
                current_frame = int(frame)
                if current_frame == previous_frame:
                    tmp_pp.append(person)
                    tmp_x.append(x)
                elif current_frame > previous_frame:
                    previous_frame = current_frame
                    sorted_idx = np.argsort(tmp_x)
                    tmp_pp_np = np.array(tmp_pp)[sorted_idx]
                    self.sorted_people = self.sorted_people + tmp_pp_np.tolist()
                    tmp_pp = []
                    tmp_x = []
                    tmp_pp.append(person)
                    tmp_x.append(x)
            sorted_idx = np.argsort(tmp_x)
            tmp_pp_np = np.array(tmp_pp)[sorted_idx]
            self.sorted_people = self.sorted_people + tmp_pp_np.tolist()
