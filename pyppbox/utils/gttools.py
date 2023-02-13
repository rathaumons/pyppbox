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

import re

from .person import Person
from .mytools import getFileName


def convert2DStringToPoint(input):
    input = input.replace("(", "")
    input = input.replace(")", "")
    input = input.replace(" ", "")
    input_list = input.split(",")
    return (int(float(input_list[0])), int(float(input_list[1])))


class GTLoader(object):

    def __init__(self):
        self.current_frame = 0
        self.realid_mode = True
        self.total_detections = 0

    def loadGT(self, gt_file_txt):
        if "real" in gt_file_txt.lower():
            self.loadGTRealID(gt_file_txt)
            print("GT: GTLoader mode = RealID")
        elif "number" in gt_file_txt.lower():
            self.realid_mode = False
            self.loadGTNumberID(gt_file_txt)
            print("GT: GTLoader mode = NumberID")

    def nextFrame(self):
        if self.realid_mode:
            return self.nextFrameRealID()
        else:
            return self.nextFrameNumberID()

    def loadGTNumberID(self, gt_file_txt):
        self.gt_frames = []
        with open(gt_file_txt, 'r') as gt_file:
            self.gt_lines = gt_file.readlines()
            same_frame = 0
            gt_frame = []
            for line in self.gt_lines:
                self.total_detections += 1
                line = line.replace("\n", "")
                line = re.split(r'\t+', line)
                line[1] = convert2DStringToPoint(line[1])
                line[0] = int(line[0])
                line[2] = int(line[2]) if line[2].isdigit() else None
                if line[0] == same_frame:
                    gt_frame.append(line)
                else:
                    same_frame += 1
                    self.gt_frames.append(gt_frame)
                    gt_frame = []
                    gt_frame.append(line)
            self.gt_frames.append(gt_frame)
        print("GT: Loaded <-- " + str(gt_file_txt))
        print("GT: Found " + str(len(self.gt_frames)) + " frames")


    def loadGTRealID(self, gt_file_txt):
        self.gt_frames = []
        with open(gt_file_txt, 'r') as gt_file:
            self.gt_lines = gt_file.readlines()
            same_frame = 0
            gt_frame = []
            for line in self.gt_lines:
                self.total_detections += 1
                line = line.replace("\n", "")
                line = re.split(r'\t+', line)
                line[1] = convert2DStringToPoint(line[1])
                line[0] = int(line[0])
                if line[0] == same_frame:
                    gt_frame.append(line)
                else:
                    same_frame += 1
                    self.gt_frames.append(gt_frame)
                    gt_frame = []
                    gt_frame.append(line)
            self.gt_frames.append(gt_frame)
        print("GT: Loaded <-- " + str(gt_file_txt))
        print("GT: Found " + str(len(self.gt_frames)) + " frames")


    def nextFrameRealID(self):
        pp = []
        for gt in self.gt_frames[self.current_frame]:
            pp.append(Person(0, 0, gt[2], gt[2], gt[1]))
        self.current_frame += 1
        return pp


    def nextFrameNumberID(self):
        pp = []
        for gt in self.gt_frames[self.current_frame]:
            pp.append(Person(gt[2], gt[2], "-", "-", gt[1]))
        self.current_frame += 1
        return pp


def sort_pp_by_x(pp):
    sorted_pp = []
    tmp_x = []

    for p in pp:
        (_x, _) = p.repspoint
        tmp_x.append(int(float(_x)))
    tmp_x = sorted(tmp_x, key=int)

    for x in tmp_x:
        for p in pp:
            (_x, _) = p.repspoint
            if x == int(float(_x)):
                sorted_pp.append(p)
                continue
    
    return sorted_pp


def get_sorted_realid_by_x_for_gt(pp):
    sorted_realid = []
    tmp_x = []

    for p in pp:
        (_x, _) = p[1]
        tmp_x.append(int(float(_x)))
    tmp_x = sorted(tmp_x, key=int)

    for x in tmp_x:
        for p in pp:
            (_x, _) = p[1]
            if x == int(float(_x)):
                sorted_realid.append(p[2])
                continue
    
    return sorted_realid


def get_sorted_faceid_by_x(pp):
    sorted_faceid = []
    tmp_x = []

    for p in pp:
        (_x, _) = p.repspoint
        tmp_x.append(int(float(_x)))
    tmp_x = sorted(tmp_x, key=int)

    for x in tmp_x:
        for p in pp:
            (_x, _) = p.repspoint
            if x == int(float(_x)):
                sorted_faceid.append(p.faceid)
                continue
    
    return sorted_faceid


def get_sorted_deepid_by_x(pp):
    sorted_deepid = []
    tmp_x = []

    for p in pp:
        (_x, _) = p.repspoint
        tmp_x.append(int(float(_x)))
    tmp_x = sorted(tmp_x, key=int)

    for x in tmp_x:
        for p in pp:
            (_x, _) = p.repspoint
            if x == int(float(_x)):
                sorted_deepid.append(p.deepid)
                continue
    
    return sorted_deepid


def removeConfidence(pp):
    i = 0
    for i in range(0, len(pp)):
        pp[i] = pp[i][:-4]
    return pp


def countDiff(gt, dt, frame):
    diff_count = 0
    missed_detect = 0
    fault_detect = 0
    tmp = len(gt) - len(dt)
    i = 0
    if tmp >= 0:
        missed_detect = tmp
        for i in range(0, len(dt)):
            # print(str(gt[i]) + "  --vs--  " + str(dt[i]))
            if gt[i] != dt[i]:
                print(" - EVA ------------------------------------>   " + str(gt[i]) + "\t  --vs--    " + str(dt[i]) + "\t" + str(frame))
                diff_count += 1
    else:
        fault_detect = len(dt) - len(gt)
        for i in range(0, len(gt)):
            # print(str(gt[i]) + "  --vs--  " + str(dt[i]))
            if gt[i] != dt[i]:
                print(" - EVA ------------------------------------>   " + str(gt[i]) + "\t  --vs--    " + str(dt[i]) + "\t" + str(frame))
                diff_count += 1
    return diff_count, missed_detect, fault_detect


class MyEvalEmpty(object):
    def __init__(self):
        pass
    def checkInReID(self):
        pass
    def checkFrame(self, pp, id_mode="deepid"):
        pass
    def getSummary(self):
        print("------------------------------------------------------------------------------")
        print(" - EVA: EVA does not evaluate in \"None/None/None\" or \"DT only\" mode.")
        print(" - EVA: EVA cannot do real-time EVA for DT+TK (Without RI) mode due to the ")
        print("        randomized IDs given by the tracker ---> Use the offline EVA tool.")
        print("------------------------------------------------------------------------------")


class MyEval(object):

    def __init__(self, gt_file_txt):
        self.reid_count = 0
        self.diff_count = 0
        self.missed_detect = 0
        self.fault_detect = 0
        self.current_frame = 0
        self.id_mode = ""
        self.gt_loader = GTLoader()
        self.gt_loader.loadGT(gt_file_txt)

    def checkInReID(self):
        self.reid_count += 1
    
    def checkFrame(self, pp, id_mode="deepid"):
        self.id_mode = id_mode
        if id_mode.lower() in " deepid faceid gt_real ":

            sorted_realid = []
            sorted_realid_gt = get_sorted_realid_by_x_for_gt(self.gt_loader.gt_frames[self.current_frame])
            
            if id_mode.lower() == "deepid":
                sorted_realid = removeConfidence(get_sorted_deepid_by_x(pp))
            elif id_mode.lower() == "faceid":
                sorted_realid = removeConfidence(get_sorted_faceid_by_x(pp))
            elif id_mode.lower() == "gt_real":
                sorted_realid = get_sorted_faceid_by_x(pp)
            
            diff_c, missed_d, fault_d = countDiff(sorted_realid_gt, sorted_realid, self.current_frame)
            self.diff_count = self.diff_count + diff_c
            self.missed_detect = self.missed_detect + missed_d
            self.fault_detect = self.fault_detect + fault_d

            self.current_frame += 1

    def getSummary(self):
        if self.id_mode.lower() == "":
            print("------------------------------------------------------------------------------")
            print(" - EVA: EVA does not evaluate in \"None/None/None\" or \"DT only\" mode.")
            print(" - EVA: EVA cannot do real-time EVA for DT+TK (Without RI) mode due to the ")
            print("        randomized IDs given by the tracker ---> Use the offline EVA tool.")
            print("------------------------------------------------------------------------------")
        else:
            # self.score = float(((self.gt_loader.total_detections - self.missed_detect - self.fault_detect - self.diff_count)/self.gt_loader.total_detections)*100)
            self.score = float(((self.gt_loader.total_detections - self.diff_count)/self.gt_loader.total_detections)*100)
            print("------------------------------------------------------------------------------")
            print(" ")
            print(" Summary: ")
            print(" ")
            print("  - ReID count     =  " + str(self.reid_count))
            print("  - Missed detect  =  " + str(self.missed_detect))
            print("  - Fault detect   =  " + str(self.fault_detect))
            print("  - Wrong ID       =  " + str(self.diff_count) + " / " + str(self.gt_loader.total_detections))
            print(" ")
            print("  ---> Final score = " + str(self.score))
            print(" ")
            print("------------------------------------------------------------------------------")
            return self.gt_loader.total_detections, self.diff_count, self.missed_detect, self.fault_detect, self.reid_count, self.score


class EvalIO(object):

    def __init__(self, mode="deepid"):
        self.storage = []
        self.map_list = []
        self.mode = mode

    def add_person(self, frame, person):
        if self.mode.lower() == "faceid":
            self.storage.append([frame, person.repspoint, person.faceid])
        else:
            self.storage.append([frame, person.repspoint, person.deepid])
    
    def dump(self, file_path):
        with open(file_path, 'w') as dump_file:
            for [frame, point, deepid] in self.storage:
                dump_file.write(str(frame) + "\t" + str(point) + "\t" + str(deepid) + "\n")

    def loadInputGTMap(self, input_gt_map_txt):
        try:
            with open(input_gt_map_txt, 'r') as map_file:
                self.map_lines = map_file.readlines()
                for line in self.map_lines:
                    line = line.replace("\n", "")
                    line = line.split('=')
                    self.map_list.append(line)
        except Exception as e:
            print("EvalIO: " + str(e))

    def getGTFileName(self, input_video):
        gt_txt = ""
        for pair in self.map_list:
            if getFileName(input_video) == pair[0]:
                gt_txt = pair[1]
        return gt_txt


class EmptyDetecter(object):
    def __init__(self):
        pass
    def detectFrame(self):
        return []


class EmptyTracker(object):
    def __init__(self):
        pass
    def update(self, _1, _2):
        return []


class EmptyReider(object):
    def __init__(self):
        pass
    def recognize(self):
        pass


class NothingTracker(object):
    def __init__(self):
        pass
    def update(self, _, pp):
        return pp


class NothingReider(object):
    def __init__(self):
        pass
    def recognize(self, res):
        return res


class TackingOnlyReider(object):

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

