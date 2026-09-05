# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


import numpy as np

from copy import deepcopy

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
        A list of object :class:`pyppbox.utils.persontools.Person`.
    sorted_people : list[Person, ...]
        A list used to store the sorted list of :attr:`people`.
    """

    def __init__(self):
        self.frames = []
        self.people = []

    def addPerson(self, frame, person):
        """Add a ``frame`` index and a :class:`pyppbox.utils.persontools.Person`
        object in that ``frame`` index into :attr:`frames` and :attr:`people`.

        Parameters
        ----------
        frame : int
            A frame index.
        person : Person
            An object of :class:`pyppbox.utils.persontools.Person` class.

        Notes
        -----
        Stores deep copies, so later mutations of the supplied people do not
        change saved frames. Add frames in nondecreasing order and keep each frame's
        people together; dumping sorts people within these frame groups, not the
        global frame sequence. Returns None.
        """
        if isinstance(person, Person):
            self.frames.append(str(frame))
            self.people.append(deepcopy(person))
        else:
            raise ValueError("RESIO : addPerson() -> Input 'person' is not valid.")

    def addPeople(self, frame, people):
        """Add a frame index and a list of :class:`pyppbox.utils.persontools.Person`
        object in that ``frame`` index into :attr:`frames` and :attr:`people`.

        Parameters
        ----------
        frame : int
            A frame index.
        people : list[Person, ...]
            A list of object :class:`pyppbox.utils.persontools.Person`.

        Notes
        -----
        Stores deep copies, so later mutations of the supplied people do not
        change saved frames. Add frames in nondecreasing order and keep each frame's
        people together; dumping sorts people within these frame groups, not the
        global frame sequence. Returns None.
        """
        if isinstance(people, list):
            if len(people) > 0:
                if isinstance(people[0], Person):
                    for person in people:
                        self.frames.append(str(frame))
                        self.people.append(deepcopy(person))
                else:
                    raise ValueError("RESIO : addPeople() -> Input 'people' is not valid.")

    def dump(self, dump_dir=default_dump_dir, dump_mode=3, id_mode="deepid", include_misc=False, max_misc=5):
        """Dump the result as a text file in a directory with a choice of :code:`"deepid"` or
        :code:`"faceid"`. Each line represents frame index and a person's details separated by '\t'.

        Parameters
        ----------
        dump_dir : str
            Defaults to ``'{pyppbox root}/data/res'``.
            A directory where to dump the result text file.
        dump_mode : int
            Defaults to ``3``.
            Set 1 to dump: frame index, repspoint, deepid/faceid.
            Set 2 to dump: frame index, repspoint, deepid/faceid, box_xywh.
            Set 3 to dump: frame index, repspoint, deepid/faceid, box_xywh, box_xyxy.
        id_mode : str
            Defaults to ``"deepid"``.
            Set choice between :code:`"deepid"` and :code:`"faceid"`.
        include_misc : bool
            Defaults to ``False``.
            Set whether to include misc (Miscellaneous items).
        max_misc : int
            Defaults to ``5``.
            Set the maximum number of miscellaneous items to include.

        Notes
        -----
        Writes a timestamped ``res_<timestamp>_full.txt`` and returns None;
        the path is reported through logging. The directory must already exist; a
        missing requested directory falls back to the package's ``data/res`` directory.
        People are sorted by representative-point x within each consecutive frame group.
        Stored records are retained after writing. Calls within the same timestamp
        second can overwrite the same output file.
        """
        dump_file = self.__generateFileName__(dump_dir)
        if id_mode != "deepid":
            if str(id_mode).lower() in " deepid faceid ":
                id_mode = str(id_mode).lower()
                add_info_log(f"-----RESIO : Set id_mode='{id_mode}'")
            else :
                add_warning_log(f"-----RESIO : id_mode='{id_mode}' is not recognized.")
                add_warning_log(f"-----RESIO : Override id_mode='{id_mode}'.")
        self.__sort_people_by_x__()
        dump_mode = int(dump_mode)
        if dump_mode < 1 and dump_mode > 3:
            add_warning_log("-----RESIO : 'dump_mode' is out of range -> Override 'dump_mode=3'")
        with open(dump_file, 'w') as dumpfile:
            for f, p in zip(self.frames, self.sorted_people):
                dump_str = ""
                if dump_mode == 1:
                    if id_mode == "deepid": 
                        tmp_deepid = p.deepid
                        if '%' in tmp_deepid: tmp_deepid = tmp_deepid[:-4]
                        dump_str = f"{f}\t{p.repspoint}\t{tmp_deepid}\n"
                    elif id_mode == "faceid": 
                        tmp_faceid = p.faceid
                        if '%' in tmp_faceid: tmp_faceid = tmp_faceid[:-4]
                        dump_str = f"{f}\t{p.repspoint}\t{tmp_faceid}\n"
                elif dump_mode == 2:
                    if id_mode == "deepid": 
                        tmp_deepid = p.deepid
                        if '%' in tmp_deepid: tmp_deepid = tmp_deepid[:-4]
                        dump_str = f"{f}\t{p.repspoint}\t{tmp_deepid}\t{p.box_xywh}\n"
                    elif id_mode == "faceid": 
                        tmp_faceid = p.faceid
                        if '%' in tmp_faceid: tmp_faceid = tmp_faceid[:-4]
                        dump_str = f"{f}\t{p.repspoint}\t{tmp_faceid}\t{p.box_xywh}\n"
                else:
                    if id_mode == "deepid": 
                        tmp_deepid = p.deepid
                        if '%' in tmp_deepid: tmp_deepid = tmp_deepid[:-4]
                        dump_str = f"{f}\t{p.repspoint}\t{tmp_deepid}\t{p.box_xywh}\t{p.box_xyxy}\n"
                    elif id_mode == "faceid": 
                        tmp_faceid = p.faceid
                        if '%' in tmp_faceid: tmp_faceid = tmp_faceid[:-4]
                        dump_str = f"{f}\t{p.repspoint}\t{tmp_faceid}\t{p.box_xywh}\t{p.box_xyxy}\n"
                if include_misc:
                    if p.misc:
                        misc = p.misc[:max_misc]
                        dump_str = dump_str[:-1] + '\t' + '\t'.join(str(m) for m in misc) + '\n'
                dumpfile.write(dump_str)
        add_info_log(f"-----RESIO : Successfully dump to '{dump_file}'")

    def dumpAll(self, dump_dir=default_dump_dir, dump_mode=3, include_misc=False, max_misc=5):
        """Dump the result as a text file in a directory with both :code:`"deepid"` and :code:`"faceid"`.
        Each line represents frame index and a person's details separated by '\t'.

        Parameters
        ----------
        dump_dir : str
            Defaults to ``'{pyppbox root}/data/res'``.
            A directory of where to dump the result text file.
        dump_mode : int
            Defaults to ``3``.
            Set 1 to dump: frame index, repspoint, deepid, faceid.
            Set 2 to dump: frame index, repspoint, deepid, faceid, box_xywh.
            Set 3 to dump: frame index, repspoint, deepid, faceid, box_xywh, box_xyxy.
        include_misc : bool
            Defaults to ``False``.
            Set whether to include misc (Miscellaneous items).
        max_misc : int
            Defaults to ``5``.
            Set the maximum number of miscellaneous items to include.

        Notes
        -----
        Writes a timestamped ``res_<timestamp>_full.txt`` and returns None;
        the path is reported through logging. The directory must already exist; a
        missing requested directory falls back to the package's ``data/res`` directory.
        People are sorted by representative-point x within each consecutive frame group.
        Stored records are retained after writing. Calls within the same timestamp
        second can overwrite the same output file.
        """
        dump_file = self.__generateFileName__(dump_dir)
        dump_mode = int(dump_mode)
        self.__sort_people_by_x__()
        if dump_mode < 1 and dump_mode > 3:
            add_warning_log("-----RESIO : 'dump_mode' is out of range -> Override 'dump_mode=3'")
        with open(dump_file, 'w') as dumpfile:
            for f, p in zip(self.frames, self.sorted_people):
                dump_str = ""
                tmp_deepid = p.deepid
                tmp_faceid = p.faceid
                if '%' in tmp_deepid: tmp_deepid = tmp_deepid[:-4]
                if '%' in tmp_faceid: tmp_faceid = tmp_faceid[:-4]
                if dump_mode == 1:
                    dump_str = f"{f}\t{p.repspoint}\t{tmp_deepid}\t{tmp_faceid}\n"
                elif dump_mode == 2:
                    dump_str = f"{f}\t{p.repspoint}\t{tmp_deepid}\t{tmp_faceid}\t{p.box_xywh}\n"
                else:
                    dump_str = f"{f}\t{p.repspoint}\t{tmp_deepid}\t{tmp_faceid}\t{p.box_xywh}\t{p.box_xyxy}\n"
                if include_misc:
                    if p.misc:
                        misc = p.misc[:max_misc]
                        dump_str = dump_str[:-1] + '\t' + '\t'.join(str(m) for m in misc) + '\n'
                dumpfile.write(dump_str)
        add_info_log(f"-----RESIO : Successfully dump to '{dump_file}'")

    def __generateFileName__(self, dump_dir=default_dump_dir):
        timestamp = getTimestamp()
        dump_file_name = f"res_{timestamp}_full.txt"
        if isExist(dump_dir):
            dump_dir = getAbsPathFDS(dump_dir)
        else:
            add_warning_log(f"-----RESIO : dump_dir='{dump_dir}' does not exist!")
            dump_dir = default_dump_dir
            add_warning_log(f"-----RESIO : Override dump_dir='{dump_dir}'.")
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
