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


import re
import cv2
import numpy as np

from .persontools import Person
from .commontools import getFileName
from .logtools import add_info_log, add_error_log


class GTIO(object):

    """
    A class used for performing I/O opterations on the GT (Ground-truth) text file of 
    the supported datasets.

    Attributes
    ----------
    map_list : list[list[str, str], ...]
        A map list of :code:`[[video file name, GT (Ground-truth) text file name], ...]`.
    """

    def __init__(self):
        self.map_list = []

    def loadInputGTMap(self, gt_map_txt, splitter=':'):
        """Load an input of GT (Ground-truth) map text and set :attr:`map_list`. 
        A map text must have consistent format with the same spliiter in every 
        line like. The default format is :code:`video_file_name:gt_file_name`; for example, 
        the first line in the map file looks like :code:`video.mp4:gt.txt`. Most suitable 
        :obj:`splitter` is colon :code:`:` because it can't be used in a file name. Other 
        symbol/char like :code:`=` should be fine as long as it is not used in the file 
        names of both video and GT text file. 

        Parameters
        ----------
        gt_map_txt : str
            A file path of GT (Ground-truth) map text.
        splitter : str, default=':'
            A string with length of :code:`1` used as a splitter of the mapping.
        """
        if len(str(splitter)) > 1:
            msg = ("GTIO : loadInputGTMap() -> splitter='" + str(splitter) + 
                   "' is not acceptable")
            add_error_log(msg)
            raise ValueError(msg)
        try:
            with open(gt_map_txt, 'r') as map_file:
                self.map_lines = map_file.readlines()
                for line in self.map_lines:
                    line = formatLineInGTMap(line, splitter=splitter)
                    line = line.split(splitter)
                    self.map_list.append(line)
        except Exception as e:
            msg = "GTIO : loadInputGTMap() -> " + str(e)
            add_error_log(msg)
            raise ValueError(msg)

    def getGTFileName(self, input_video):
        """Return the video file name corresponding to the :obj:`input_video`.

        Parameters
        ----------
        input_video : str
            File path of input video.
        
        Returns
        -------
        str
            A file name of the coresponding GT (Ground-truth) text.
        """
        gt_txt = ""
        for pair in self.map_list:
            if getFileName(input_video) == pair[0]:
                gt_txt = pair[1]
        return gt_txt

    def loadGT(self, gt_file_txt):
        """Read an input of GT (Ground-truth) text file, and return the :obj:`gt_frames`, 
        :obj:`gt_frames_dict`, :obj:`total_detections`, and :obj:`init_frame`.

        Parameters
        ----------
        gt_file_txt : str
            A file path of GT (Ground-truth) text.
        
        Returns
        -------
        list[list[list[int, tuple(int, int), str, [int, int, int, int], [int, int, int, int]], ...], ...]
            A list of GT frames, each frame is a list of GT-format person, and each 
            GT-format person is a list carrying info of a person in a frame like 
            :code:`[1, (637, 308), "Franklin", [593 241  89 270], [593 241 682 511]]`.
        list[int, ...]
            A list of frame indexes of the GT (Ground-truth).
        int
            Total number of all detection or ID count in all frame(s) in the 
            GT (Ground-truth).
        int
            Initial frame index.
        """
        gt_frames = []
        gt_frames_dict = []
        total_detections = 0
        init_frame = 0
        try:
            with open(gt_file_txt, 'r') as gt_file:
                gt_lines = gt_file.readlines()
                same_frame = 0
                gt_frame = []
                set_init_frame = False
                loop_id = 0
                loop_len = len(gt_lines)
                for line in gt_lines:
                    total_detections += 1
                    line = line.replace("\n", "")
                    line = re.split(r'\t+', line)
                    line[1] = convert2DStringToPoint(line[1])
                    line[0] = int(line[0])
                    if not set_init_frame:
                        init_frame = line[0]
                        same_frame = line[0]
                        set_init_frame = True
                    if line[0] <= same_frame:
                        gt_frame.append(line)
                    elif line[0] > same_frame or loop_id == loop_len - 1:
                        gt_frames.append(gt_frame)
                        gt_frames_dict.append(same_frame)
                        same_frame = line[0]
                        gt_frame = []
                        gt_frame.append(line)
                    loop_id += 1
                gt_frames.append(gt_frame)
                gt_frames_dict.append(same_frame)
            add_info_log("------GTIO : Loaded <- " + getFileName(gt_file_txt))
            add_info_log("------GTIO : Found " + str(len(gt_frames)) + 
                         " nonempty frame(s) and the initial frame is " + 
                         str(init_frame) + ".")
        except Exception as e:
            msg = "GTIO : loadGT() -> " + str(e)
            add_error_log(msg)
            raise ValueError(msg)
        return gt_frames, gt_frames_dict, total_detections, init_frame


class GTInterpreter(object):

    """
    A class used for interpreting GT (Ground-truth) of supported datasets.

    Attributes
    ----------
    init_frame : int
        Initial frame index.
    current_frame : int
        Current frame index.
    total_detections : int
        Total number of all detection or ID count in all frame(s) in the 
        GT (Ground-truth).
    detect_only : bool
        Indication of whether using 'Detect Only' mode or full GT mode with real ID.
    gtIO : GTIO
        GT (Ground-truth) IO, :class:`GTIO` object.
    unknownFID : str
        A string for setting unknown :obj:`faceid` of a :class:`Person` object.
    unknownDID : str
        A string for setting unknown :obj:`deepid` of a :class:`Person` object.
    gt_frames : list[list[list[int, tuple(int, int), str, [int, int, int, int], [int, int, int, int]], ...], ...]
        A list of GT frames, each frame is a list of GT-format person, and 
        each GT-format person is a list carrying info of a person in a frame like 
        :code:`[1, (637, 308), "Franklin", [593 241  89 270], [593 241 682 511]]`.
    gt_frames_dict : list[int, ...]
        A list of frame indexes of GT (Ground-truth).
    """

    def __init__(self):
        self.init_frame = 0
        self.current_frame = 0
        self.total_detections = 0
        self.detect_only = False
        self.gtIO = GTIO()

    def setDetectOnly(self, unknownFID="Unknown", unknownDID="Unknown", detect_only=True):
        """Set whether to use 'Detect Only' mode (Set unkown faceid and deepid) or full GT 
        mode with real ID as in the GT (Ground-truth).
        
        Parameters
        ----------
        unknownFID : str, default="Unknown"
            Set the :attr:`unknownFID`.
        unknownFID : str, default="Unknown"
            Set the :attr:`unknownFID`.
        detect_only : bool, default=True
            Set :code:`detect_only=True` to tell :func:`getPeople()` to return people 
            with unkown :obj:`faceid` and :obj:`deepid`.
            Set :code:`detect_only=False` to tell :func:`getPeople()` to return people 
            with real IDs as in the GT (Ground-truth).
        """
        self.detect_only = detect_only
        self.unknownFID = unknownFID
        self.unknownDID = unknownDID

    def __createStaticCID__(self, realID):
        if realID == "Lester":
            return 0
        elif realID== "Michael":
            return 1
        elif realID == "Franklin":
            return 2
        elif realID == "Trevor":
            return 3
        elif realID == "Amanda":
            return 4
        else:
            return -1

    def setGT(self, gt_file_txt):
        """Set a GT (Ground-truth) file.

        Parameters
        ----------
        gt_file_txt : str
            A file path of GT (Ground-truth) text.
        """
        (self.gt_frames, 
         self.gt_frames_dict, 
         self.total_detections, 
         self.init_frame) = self.gtIO.loadGT(gt_file_txt=gt_file_txt)

    def findGTFrame(self, frame_index):
        """Return a gt_frame, list of GT-format person in the given :obj:`frame_index`.

        Parameters
        ----------
        frame_index : int
            A frame index.
        
        Returns
        -------
        list[list[str, ...], ...]
            A list of all GT-format person in the given :obj:`frame_index`; for example, 
            :code:`[[1, (637, 308), "Franklin", [593 241  89 270], [593 241 682 511]], ...]`.
        """
        gt_frame = []
        if int(frame_index) in self.gt_frames_dict:
            found_index = self.gt_frames_dict.index(frame_index)
            gt_frame = self.gt_frames[found_index]
        return gt_frame

    def getPeople(self, img, visual=False):
        """Return a :code:`list[Person, ...]` and a cv :obj:`Mat`, similar to function 
        :func:`detectPeople()` in :py:mod:`pyppbox.standalone`.

        Parameters
        ----------
        img : Mat
            A cv :obj:`Mat` image.
        visual : 
            An indication of whether to draw bounding boxes on the return :obj:`img`.

        Returns
        -------
        list[Person, ...]
            A list of :class:`Person` object.
        Mat
            A cv :obj:`Mat` image.
        """
        people = []
        tmp_id = 0
        for gt in self.findGTFrame(self.current_frame):
            box_xywh = convertStringToNPL(gt[3])
            box_xyxy = convertStringToNPL(gt[4])
            if self.detect_only:
                people.append(Person(tmp_id, tmp_id, box_xywh=box_xywh, box_xyxy=box_xyxy, 
                                     repspoint=gt[1], faceid=self.unknownFID, deepid=self.unknownDID))
                tmp_id += 1
            else:
                tmp_sttcid = self.__createStaticCID__(gt[2])
                people.append(Person(tmp_sttcid, tmp_sttcid, box_xywh=box_xywh, box_xyxy=box_xyxy, 
                                     repspoint=gt[1], faceid=gt[2], deepid=gt[2]))
            if visual:
                bxyxy = box_xyxy.tolist()
                cv2.circle(img, gt[1], radius=5, color=(0, 0, 255), thickness=-1)
                cv2.rectangle(img, (bxyxy[0], bxyxy[1]), (bxyxy[2], bxyxy[3]), (255, 255, 0), 2)
        self.current_frame += 1
        return people, img


#############################################################################################################


def convert2DStringToPoint(input):
    """
    :meta private:
    """
    input = input.replace("(", "")
    input = input.replace(")", "")
    input = input.replace(" ", "")
    input_list = input.split(",")
    return (int(float(input_list[0])), int(float(input_list[1])))

def convertStringToNPL(input):
    """
    :meta private:
    """
    input = input.replace("#", "")
    input = input.replace("[", "")
    input = input.replace("]", "")
    input = " ".join(input.split())
    input_list = list(map(float, input.split()))
    return np.array(input_list).astype(int)

def formatLineInGTMap(line, splitter=':'):
    """
    :meta private:
    """
    line = str(line)
    line = line.replace("\n", "")
    line = line.replace("\r", "")
    s_splitter = " " + splitter
    splitter_s = splitter + " "
    while s_splitter in line or splitter_s in line:
        line = line.replace(s_splitter, splitter)
        line = line.replace(splitter_s, splitter)
    while line[0] == ' ':
        line = line[1:]
    while line[-1] == ' ':
        line = line[:-1]
    return line
