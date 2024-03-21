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


import timeit

from .gttools import GTInterpreter, convertStringToNPL
from .commontools import joinFPathFull, getAbsPathFDS, isExist, getAncestorDir
from .logtools import add_info_log, add_warning_log, add_error_log


class MyEVA(object):

    """
    A class used to generate the evaluation of the supported datasets.

    Attributes
    ----------
    reid_count : int
        Total number of ReID count.
    diff_count : int
        Total number of wrong IDs.
    missed_detect : int
        Total number of missed detection(s).
    fault_detect : int
        Total number of fault detection(s).
    current_frame : int
        Current frame index.
    frame_to_check : int
        Frame index which is used for the validation.
    score : float
        Score of the evaluation, between 0 and 1.
    id_mode : str, default="deepid"
        Indication of whether :code:`"deepid"` or :code:`"faceid"` is used for the evaluation.
    gt_interpreter : GTInterpreter, auto
        GT (Ground-truth) loader, :class:`GTInterpreter` object.
    gt_file_name : str, auto
        GT (Ground-truth) file name.
    gt_file : str, auto
        GT (Ground-truth) file path.
    """

    def __init__(self):
        self.reid_count = 0
        self.diff_count = 0
        self.missed_detect = 0
        self.fault_detect = 0
        self.current_frame = 0
        self.frame_to_check = 0
        self.id_mode = "deepid"
        self.no_gt = False

    def setGTByGTMap(self, gt_map_txt, input_video, id_mode="deepid"):
        """Set a GT (Ground-truth) text file by give the GT mapping file :obj:`gt_map_txt` and 
        :obj:`input_video` of a supported dataset.

        Parameters
        ----------
        gt_map_file : str
            A file path of a Video=GT mapping text file for GT (Ground-truth).
        input_video : str
            A video file path.
        id_mode : bool, default="deepid"
            An indication of whether :code:`"deepid"` or :code:`"faceid"` is used for the evaluation.
        """
        if id_mode != "deepid":
            if str(id_mode).lower() in " deepid faceid ":
                self.id_mode = str(id_mode).lower()
                add_info_log("-------EVA : Set id_mode='" + str(self.id_mode) + "'")
            else :
                add_warning_log("-------EVA : id_mode='" + str(id_mode) + "' is not recognized.")
                add_warning_log("-------EVA : Overwite id_mode='" + str(self.id_mode) + "'.")
        self.gt_interpreter = GTInterpreter()
        self.gt_interpreter.gtIO.loadInputGTMap(gt_map_txt)
        self.gt_file_name = self.gt_interpreter.gtIO.getGTFileName(input_video)
        self.gt_file = ""
        if self.gt_file_name != "":
            self.gt_file = joinFPathFull(getAncestorDir(gt_map_txt), self.gt_file_name)
            self.gt_interpreter.setGT(self.gt_file)
        else:
            msg = ("MyEVA : setGTByGTMap() -> There is no GT file for the input '" + 
                   str(input_video) + "'")
            # add_error_log(msg)
            # raise ValueError(msg)
            add_warning_log(msg)
            self.no_gt = True

    
    def setGTByKnownGTFile(self, gt_file, id_mode="deepid"):
        """Set a GT (Ground-truth) text file and :obj:`id_mode` which is used to compare.

        Parameters
        ----------
        gt_file : str
            A file path of a GT (Ground-truth).
        id_mode : bool, default="deepid"
            An indication of whether :code:`"deepid"` or :code:`"faceid"` is used for the evaluation.
        """
        if id_mode != "deepid":
            if str(id_mode).lower() in " deepid faceid ":
                self.id_mode = str(id_mode).lower()
                add_info_log("-------EVA : Set id_mode='" + str(self.id_mode) + "'")
            else :
                add_warning_log("-------EVA : id_mode='" + str(id_mode) + "' is not recognized")
                add_warning_log("-------EVA : Overwite id_mode='" + str(self.id_mode) + "'")
        self.gt_interpreter = GTInterpreter()
        self.gt_file = gt_file
        if isExist(gt_file):
            self.gt_interpreter.setGT(getAbsPathFDS(self.gt_file))
            add_info_log("-------EVA : Custom gt_file='" + str(gt_file) + "'")
        else:
            msg = ("MyEVA : setGTByKnownGTFile() -> The input gt_file='" + 
                   str(gt_file) + "' does not exist")
            # add_error_log(msg)
            # raise ValueError(msg)
            add_warning_log(msg)
            self.no_gt = True

    def __checkInReID__(self):
        self.reid_count += 1

    def setReIDcount(self, total_count):
        """Set the total :attr:`reid_count` according to :obj:`total_count`.

        Parameters
        ----------
        total_count : int
            Total number of ReID count.
        """
        self.reid_count = total_count

    def __compareIDList2GT__(self, id_list_gt, id_list_dt):
        diff_count = 0
        missed_detect = 0
        fault_detect = 0
        tmp = len(id_list_gt) - len(id_list_dt)
        i = 0
        if tmp >= 0:
            missed_detect = tmp
            for i in range(0, len(id_list_dt)):
                if '%' in id_list_dt[i]: id_list_dt[i] = id_list_dt[i][:-4]
                if id_list_gt[i] != id_list_dt[i]:
                    msg = ("-------EVA : ------------------------------>   " + 
                           str(id_list_gt[i]) + "\t  --vs--    " + str(id_list_dt[i]) + 
                           "\t@" + str(self.current_frame))
                    add_info_log(msg)
                    diff_count += 1
        else:
            fault_detect = len(id_list_dt) - len(id_list_gt)
            for i in range(0, len(id_list_gt)):
                if '%' in id_list_dt[i]: id_list_dt[i] = id_list_dt[i][:-4]
                if id_list_gt[i] != id_list_dt[i]:
                    msg = ("-------EVA : ------------------------------>   " + 
                           str(id_list_gt[i]) + "\t  --vs--    " + str(id_list_dt[i]) + 
                           "\t@" + str(self.current_frame))
                    add_info_log(msg)
                    diff_count += 1
        return diff_count, missed_detect, fault_detect

    def __compareDeepID__(self, gt_frame, people_dt):
        diff_count = 0
        missed_detect = 0
        fault_detect = 0
        tmp = len(gt_frame) - len(people_dt)
        i = 0
        if tmp >= 0:
            missed_detect = tmp
            for i in range(0, len(people_dt)):
                person_index_to_compare = findPersonIndexGTFrame(gt_frame, people_dt[i].box_xyxy, 
                                                                 max_spread_limit=16)
                if person_index_to_compare >= 0:
                    tmp_deepid = people_dt[i].deepid
                    if '%' in tmp_deepid: tmp_deepid = tmp_deepid[:-4]
                    if gt_frame[person_index_to_compare][2].lower() != tmp_deepid.lower():
                        msg = ("-------EVA : Frame \t@" + str(self.current_frame) + 
                               "\t   ---------->   (GT) " + str(gt_frame[person_index_to_compare][2]) + 
                               "\t -v.s- \t " + str(tmp_deepid))
                        add_info_log(msg)
                        diff_count += 1
                else: fault_detect += 1
        else:
            fault_detect = abs(tmp)
            for i in range(0, len(gt_frame)):
                person_index_to_compare = findPersonIndexGTFrame(gt_frame, people_dt[i].box_xyxy, 
                                                                 max_spread_limit=16)
                if person_index_to_compare >= 0:
                    tmp_deepid = people_dt[i].deepid
                    if '%' in tmp_deepid: tmp_deepid = tmp_deepid[:-4]
                    if gt_frame[person_index_to_compare][2].lower() != tmp_deepid.lower():
                        msg = ("-------EVA : Frame \t@" + str(self.current_frame) + 
                               "\t   ---------->   (GT) " + str(gt_frame[person_index_to_compare][2]) + 
                               "\t -v.s- \t " + str(tmp_deepid))
                        add_info_log(msg)
                        diff_count += 1
                else: fault_detect += 1
        return diff_count, missed_detect, fault_detect
    
    def __compareFaceID__(self, gt_frame, people_dt):
        diff_count = 0
        missed_detect = 0
        fault_detect = 0
        tmp = len(gt_frame) - len(people_dt)
        i = 0
        if tmp >= 0:
            missed_detect = tmp
            for i in range(0, len(people_dt)):
                person_index_to_compare = findPersonIndexGTFrame(gt_frame, people_dt[i].box_xyxy, 
                                                                 max_spread_limit=16)
                if person_index_to_compare >= 0:
                    tmp_faceid = people_dt[i].faceid
                    if '%' in tmp_faceid: tmp_faceid = tmp_faceid[:-4]
                    if gt_frame[person_index_to_compare][2].lower() != tmp_faceid.lower():
                        msg = ("-------EVA : Frame \t@" + str(self.current_frame) + 
                               "\t   ---------->   (GT) " + str(gt_frame[person_index_to_compare][2]) + 
                               "\t -v.s- \t " + str(tmp_faceid))
                        add_info_log(msg)
                        diff_count += 1
                else: fault_detect += 1
        else:
            fault_detect = abs(tmp)
            for i in range(0, len(gt_frame)):
                person_index_to_compare = findPersonIndexGTFrame(gt_frame, people_dt[i].box_xyxy, 
                                                                 max_spread_limit=16)
                if person_index_to_compare >= 0:
                    tmp_faceid = people_dt[i].faceid
                    if '%' in tmp_faceid: tmp_faceid = tmp_faceid[:-4]
                    if gt_frame[person_index_to_compare][2].lower() != tmp_faceid.lower():
                        msg = ("-------EVA : Frame \t@" + str(self.current_frame) + 
                               "\t   ---------->   (GT) " + str(gt_frame[person_index_to_compare][2]) + 
                               "\t -v.s- \t " + str(tmp_faceid))
                        add_info_log(msg)
                        diff_count += 1
                else: fault_detect += 1
        return diff_count, missed_detect, fault_detect

    def validate(self, people, frame_id=-1):
        """Validate a frame by comparing the given list of Person to the GT (Ground-truth) file. 
        :class:`MyEVA` has an internal frame tracker :obj:`current_frame` which helps the :meth:`validate()` 
        keep track which frame to validate especially when it is called inside a loop of video frames; 
        however, when the parameter :obj:`frame_id` is used and set to greater than :code:`-1`, 
        :meth:`validate()` will refer to this :obj:`frame_id` as the frame index for the validation.

        Parameters
        ----------
        people : list[Person, ...]
            A list of :class:`Person` object.
        frame_id : int, default=-1, optional
            An indication of which frame index is used for the validation.
        """
        if frame_id >= 0: self.frame_to_check = frame_id
        else : self.frame_to_check = self.current_frame
        if self.gt_file != "":
            diff_c = 0
            missed_d = 0
            fault_d = 0
            gt_frame = self.gt_interpreter.findGTFrame(self.frame_to_check)
            if self.id_mode == "deepid":
                diff_c, missed_d, fault_d = self.__compareDeepID__(gt_frame=gt_frame, people_dt=people)
            elif self.id_mode == "faceid":
                diff_c, missed_d, fault_d = self.__compareFaceID__(gt_frame=gt_frame, people_dt=people)
            self.diff_count = self.diff_count + diff_c
            self.missed_detect = self.missed_detect + missed_d
            self.fault_detect = self.fault_detect + fault_d
        self.current_frame += 1

    def getSummary(self, print_summary=True):
        """Generate a summary of the evaluation.

        Parameters
        ----------
        print_summary : bool, default=True
            An indication of whether to print a summary text in the termianl.

        Returns
        -------
        diff_count : int
            Total number of wrong ID(s).
        missed_detect : int
            Total number of missed detection(s).
        fault_detect : int
            Total number of fault detection(s).
        reid_count : int
            Total number of ReID count.
        gt_interpreter.total_detections : int
            Total number of detection(s) or all ID(s) in all frame(s) in the GT (Ground-truth).
        score : float
            Score of the evaluation, between 0 and 1 -> (Total ID - Wrong ID - Missed Detection) / Total ID.
        """
        if self.no_gt:
            self.score = 0.0
        else:
            self.score = float((self.gt_interpreter.total_detections - self.diff_count - 
                                self.missed_detect) / self.gt_interpreter.total_detections)
        if print_summary:
            msg = ("\n#####################################################################\n\n" +
                   "  Summary: \n\n" +
                   "  -----------------------------------------------------------------  \n" +
                   "                  ReID count  =  " + str(self.reid_count) + "\n" +
                   "      Missed detection count  =  " + str(self.missed_detect) + "\n" +
                   "       Fault detection count  =  " + str(self.fault_detect) + "\n" +
                   "              Wrong ID count  =  " + (str(self.diff_count) + " / " + 
                                                        str(self.gt_interpreter.total_detections) + "\n") +
                   "  -----------------------------------------------------------------  \n\n" +
                   "               * Final score  =  " + str(self.score) + "\n\n" +
                   "     [(Total ID) - (Wrong ID) - (Missed Detection)] / (Total ID)     \n\n" +
                   "#####################################################################\n")
            add_info_log(msg, add_new_line=True)
        return (self.diff_count, self.missed_detect, self.fault_detect, self.reid_count, 
                self.gt_interpreter.total_detections, self.score)


class NothingDetecter(object):
    """
    A class acted as a detector which does not perform any detection. 
    """
    def __init__(self):
        pass
    def detectFrame(self):
        return []


class NothingTracker(object):
    """
    A class acted as a tracker which does not perform any tracking. 
    """
    def __init__(self):
        pass
    def update(self, pp, img=None):
        return pp


class NothingReider(object):
    """
    A class acted as a reider which does not perform any re-identifying. 
    """
    def __init__(self):
        pass
    def recognize(self, res):
        return res


class TKOReider(object):

    """
    A class acted as a reider which can return random IDs. 
    """

    def __init__(self, static=False, static_ids=[], string_length=5):
        self.is_static = static
        self.string_length = string_length
        self.static_index = 0
        self.setStaticIDs(static_ids)

    def recognize(self, _):
        res = ""
        if self.is_static:
            if self.static_index >= self.static_ids_len:
                res = self.generateStaticID()
            else:
                res = self.generateID(self.string_length)
        else:
            res = self.generateID(self.string_length)
        return res

    def generateID(self, string_length):
        import uuid
        random = str(uuid.uuid4()) # Convert UUID format to a Python string.
        random = random.upper() # Make all characters uppercase.
        random = random.replace("-","") # Remove the UUID '-'.
        res = random[0:string_length]
        return res

    def generateStaticID(self):
        res = self.static_ids[self.static_index]
        self.static_index += 1
        return res

    def setStaticIDs(self, static_ids, plus_random=1000):
        self.static_ids = static_ids
        self.static_ids_len = len(self.static_ids)
        if self.is_static and self.static_ids_len > 0:
            self.static_ids = static_ids
        elif self.is_static and self.static_ids_len <= 0:
            self.static_ids.append("Lester")
            self.static_ids.append("Michael")
            self.static_ids.append("Franklin")
            self.static_ids.append("Trevor")
            self.static_ids.append("Amanda")
            self.static_ids.append("MCU-Vision")
            self.static_ids.append("MCU-Thor")
            self.static_ids.append("MCU-Hulk")
            self.static_ids.append("MCU-Loki")
            self.static_ids.append("MCU-Thanos")
            self.static_ids.append("DC-Batman")
            self.static_ids.append("DC-Superman")
            self.static_ids.append("DC-Aquaman")
            self.static_ids.append("DC-Shazam")
            self.static_ids.append("DC-Cyborg")
            for _ in range(0, plus_random):
                self.static_ids.append(self.generateID(self.string_length))


###############################################################################################


def findPersonIndexGTFrame(gt_frame, box_xyxy, box_xyxy_index=4, max_spread_limit=16):
    """
    :meta private:
    """
    box_list = box_xyxy.tolist()
    min_box_spread = 8192
    index = -1
    i = 0
    for p in gt_frame:
        pbbox_list = convertStringToNPL(p[box_xyxy_index]).tolist()
        max_ss = max([abs(box_list[j] - pbbox_list[j]) for j in range(0, 4)])
        if max_ss < min_box_spread:
            min_box_spread = max_ss
            index = i
        i += 1
    if min_box_spread > max_spread_limit: index = -1
    return index

def compareRes2Ref(res_txt, ref_txt, res_box_xyxy_index=5, ref_box_xyxy_index=4, 
                   res_compare_index=2, ref_compare_index=2, box_max_spread=5):
    """Compare the result text file generated by :class:`ResIO` to any reference 
    or GT (Ground-truth) text file, ideally used for comparing the strings of 
    :obj:`deepid` or :obj:`faceid` in result to the reference.

    Parameters
    ----------
    res_txt : str
        A path of the result text file.
    ref_txt : str
        A path of the reference text file.
    res_box_xyxy_index : int, default=5
        Index of bounding box :code:`[X1, Y1, X2, Y2]` in the result text file.
    ref_box_xyxy_index : int, default=4
        Index of bounding box :code:`[X1, Y1, X2, Y2]` in the reference text file.
    res_compare_index : int, default=2
        Index of what to compare in the result text file.
    ref_compare_index : int, default=2
        Index of what to compare in the reference text file.
    box_max_spread : int, default=10
        Max spread or max margin used to decide whether 2 bounding boxes are the same 
        by comparing the differences between the elements in the result's bounding 
        box and the coressponding elements in the reference's bounding box.
    
    Returns
    -------
    int
        Total number of wrong ID count.
    int
        Total number of missed detection count.
    int
        Total number of fault detection count.
    int
        Total number of all detection or ID count in all frame(s) in the reference text file.
    float
        Score of the evaluation, between 0 and 1 -> (Total ID - Wrong ID - Missed detection) / Total ID.
    """

    diff_count = 0
    missed_detect = 0
    fault_detect = 0
    score = 0
    total_detections = 0
    
    if isExist(str(ref_txt)):
        ref_interpreter = GTInterpreter()
        ref_interpreter.setGT(getAbsPathFDS(ref_txt))
        
        if isExist(str(res_txt)):
            res_interpreter = GTInterpreter()
            res_interpreter.setGT(getAbsPathFDS(res_txt))
            init_frame = ref_interpreter.init_frame
            last_frame = init_frame + ref_interpreter.gt_frames_dict[-1] + 1
            
            for frame in range(init_frame, last_frame):
                ref_frame = ref_interpreter.findGTFrame(frame)
                res_frame = res_interpreter.findGTFrame(frame)
                tmp = len(ref_frame) - len(res_frame)
                i = 0
                if tmp >= 0:
                    missed_detect = missed_detect + tmp
                    for i in range(0, len(res_frame)):
                        person_index_to_compare = findPersonIndexGTFrame(
                            ref_frame, 
                            convertStringToNPL(res_frame[i][res_box_xyxy_index]), 
                            box_xyxy_index=ref_box_xyxy_index, 
                            max_spread_limit=box_max_spread
                        )
                        if person_index_to_compare >= 0:
                            if (ref_frame[person_index_to_compare][ref_compare_index].lower() != 
                                res_frame[i][res_compare_index].lower()):
                                add_info_log(
                                    "compareRes2Ref() -> Frame \t@" + 
                                    str(frame) + "\t   ---------->   (Ref) " + 
                                    str(ref_frame[person_index_to_compare][ref_compare_index]) + 
                                    "\t -v.s- \t(Res) " + str(res_frame[i][res_compare_index].lower())
                                )
                                diff_count += 1
                        else: fault_detect += 1
                else:
                    fault_detect = fault_detect + abs(tmp)
                    for i in range(0, len(ref_frame)):
                        person_index_to_compare = findPersonIndexGTFrame(
                            ref_frame, 
                            convertStringToNPL(res_frame[i][res_box_xyxy_index]), 
                            box_xyxy_index=ref_box_xyxy_index, 
                            max_spread_limit=box_max_spread
                        )
                        if person_index_to_compare >= 0:
                            if (ref_frame[person_index_to_compare][ref_compare_index].lower() != 
                                res_frame[i][res_compare_index].lower()):
                                add_info_log(
                                    "compareRes2Ref() -> Frame \t@" +
                                    str(frame) + "\t   ---------->   (Ref) " + 
                                    str(ref_frame[person_index_to_compare][ref_compare_index]) + 
                                    "\t -v.s- \t(Res) " + str(res_frame[i][res_compare_index].lower())
                                )
                                diff_count += 1
                        else: fault_detect += 1
    
            total_detections = ref_interpreter.total_detections
            score = float((total_detections - diff_count - missed_detect)/total_detections)

            msg = ("\n#####################################################################\n\n" +
                   "  Summary: \n\n" +
                   "  -----------------------------------------------------------------  \n" +
                   "      Missed detection count  =  " + str(missed_detect) + "\n" +
                   "       Fault detection count  =  " + str(fault_detect) + "\n" +
                   "              Wrong ID count  =  " + (str(diff_count) + " / " + 
                                                          str(total_detections) + "\n") +
                   "  -----------------------------------------------------------------  \n\n" +
                   "               * Final score  =  " + str(score) + "\n\n" +
                   "     [(Total ID) - (Wrong ID) - (Missed Detection)] / (Total ID)     \n\n" +
                   "#####################################################################\n")
            add_info_log(msg, add_new_line=True)

        else:
            msg = "compareRes2Ref() -> Input 'res_txt' does not exist. "
            add_error_log(msg)
            raise ValueError(msg)
    else:
        msg = "compareRes2Ref() -> Input 'ref_txt' does not exist. "
        add_error_log(msg)
        raise ValueError(msg)
    
    return diff_count, missed_detect, fault_detect, total_detections, score
