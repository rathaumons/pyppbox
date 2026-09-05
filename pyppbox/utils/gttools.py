# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                           #
#   pyppbox: Toolbox for people detecting, tracking, and re-identifying.    #
#   Copyright (C) 2025 UMONS-Numediart                                      #
#                                                                           #
#   This program is free software: you can redistribute it and/or modify    #
#   it under the terms of the GNU Affero General Public License as          #
#   published by the Free Software Foundation, either version 3 of the      #
#   License, or (at your option) any later version.                         #
#                                                                           #
#   This program is distributed in the hope that it will be useful,         #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#   GNU Affero General Public License for more details.                     #
#                                                                           #
#   You should have received a copy of the GNU Affero General Public License#
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
    map_dict : Dict[str, str]
        A map dictionary of :code:`{video file name: GT (Ground-truth) text file name, ...}`.
    """

    def __init__(self):
        self.map_list = []
        self.map_dict = {}

    def loadInputGTMap(self, gt_map_txt, splitter=':'):
        """Append video-to-GT filename pairs from a text file.

        Parameters
        ----------
        gt_map_txt : str
            Mapping file with one ``video_filename:gt_filename`` pair per line when
            using the default separator. Use basenames, without blank rows.
        splitter : str
            Defaults to ``':'``. One nonempty character absent from both filenames.

        Raises
        ------
        ValueError
            If the separator is invalid or the file cannot be read/processed.

        Notes
        -----
        Returns None. Existing mappings are retained, and the last loaded value wins
        for duplicate video keys in ``map_dict``. Rows with the wrong number of fields
        are logged and skipped; this is not an atomic load, so earlier rows remain if
        a later row fails. ``map_list`` retains all successfully parsed pairs.
        """
        if len(str(splitter)) > 1:
            msg = f"GTIO : loadInputGTMap() -> splitter='{splitter}' is not acceptable"
            add_error_log(msg)
            raise ValueError(msg)
        try:
            with open(gt_map_txt, 'r') as map_file:
                self.map_lines = map_file.readlines()
                for line in self.map_lines:
                    line = formatLineInGTMap(line, splitter=splitter)
                    parts = line.split(splitter)
                    if len(parts) == 2:
                        self.map_list.append(parts)
                        self.map_dict[parts[0]] = parts[1]  # <-- Add this line
                    else:
                        msg = f"GTIO : loadInputGTMap() -> Invalid line format: {line}"
                        add_error_log(msg)
        except Exception as e:
            msg = f"GTIO : loadInputGTMap() -> {e}"
            add_error_log(msg)
            raise ValueError(msg)


    def getGTFileName(self, input_video):
        """Return the mapped GT filename for a video's basename.

        Parameters
        ----------
        input_video : str
            Video path or filename. Only the basename is used for a case-sensitive lookup.

        Returns
        -------
        str
            Mapped GT text filename, or an empty string when no mapping exists.
            The GT file is not opened or checked by this lookup.
        """

        # gt_txt = ""
        # for pair in self.map_list:
        #     if getFileName(input_video) == pair[0]:
        #         gt_txt = pair[1]
        # return gt_txt

        video_name = getFileName(input_video)
        return self.map_dict.get(video_name, "")

    def loadGT(self, gt_file_txt):
        """Read tab-separated GT rows grouped in nondecreasing frame-index order.

        Parameters
        ----------
        gt_file_txt : str
            Input file with no blank rows. Columns begin with frame index, representative
            point, identity, xywh box, and xyxy box. Additional columns remain available
            as text, which also allows the offline evaluator to read result files.

        Returns
        -------
        list[list[list]]
            Groups of rows. In each row, column 0 is an int and column 1 is an (x, y)
            tuple. All later columns, including box coordinates, remain strings.
        list[int]
            Frame indices corresponding to the groups, in file order.
        int
            Number of rows read.
        int
            First frame index, or 0 for an empty file.

        Raises
        ------
        ValueError
            If the file cannot be read or a row's initial fields cannot be parsed.

        Notes
        -----
        This does not sort or validate box columns. Empty input currently returns
        one empty group with frame-index list ``[0]`` and a row count of zero.
        """
        gt_frames = []
        gt_frames_list = []
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
                        gt_frames_list.append(same_frame)
                        same_frame = line[0]
                        gt_frame = []
                        gt_frame.append(line)
                    loop_id += 1
                gt_frames.append(gt_frame)
                gt_frames_list.append(same_frame)
            add_info_log(f"------GTIO : Loaded <- {getFileName(gt_file_txt)}")
            add_info_log(f"------GTIO : Found {len(gt_frames)} nonempty frame(s) and the initial frame is {init_frame}.")
        except Exception as e:
            msg = "GTIO : loadGT() -> " + str(e)
            add_error_log(msg)
            raise ValueError(msg)
        return gt_frames, gt_frames_list, total_detections, init_frame


class GTInterpreter(object):

    """A class used for interpreting GT (Ground-truth) of supported datasets.

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
        A string for setting unknown ``faceid`` of a :class:`pyppbox.utils.persontools.Person` object.
    unknownDID : str
        A string for setting unknown ``deepid`` of a :class:`pyppbox.utils.persontools.Person` object.
    gt_frames : list[list[list]]
        Row groups returned by ``GTIO.loadGT()``. Frame IDs and points are parsed;
        identities and box columns remain strings until ``getPeople()`` converts them.
    gt_frames_list : list[int, ...]
        A list of frame indexes of GT (Ground-truth).
    """

    def __init__(self):
        self.init_frame = 0
        self.current_frame = 0
        self.total_detections = 0
        self.detect_only = False
        self.gtIO = GTIO()

    def setDetectOnly(self, unknownFID="Unknown", unknownDID="Unknown", detect_only=True):
        """Set whether to use 'Detect Only' mode (Set unknown faceid and deepid) or full GT 
        mode with real ID as in the GT (Ground-truth).
        
        Parameters
        ----------
        unknownFID : str
            Defaults to ``"Unknown"``.
            Set the :attr:`unknownFID`.
        unknownDID : str
            Defaults to ``"Unknown"``.
            Set the :attr:`unknownDID`.
        detect_only : bool
            Defaults to ``True``.
            Set :code:`detect_only=True` to tell :func:`getPeople()` to return people 
            with unknown ``faceid`` and ``deepid``.
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
        """Load a GT file and replace its row groups and frame metadata.

        Parameters
        ----------
        gt_file_txt : str
            Input path accepted by ``GTIO.loadGT()``.

        Notes
        -----
        Returns None. This does not reset ``current_frame``. Set that cursor explicitly
        when reusing an interpreter, or to ``init_frame`` to start at a nonzero first GT frame.
        """
        (self.gt_frames, 
         self.gt_frames_list, 
         self.total_detections, 
         self.init_frame) = self.gtIO.loadGT(gt_file_txt=gt_file_txt)

    def findGTFrame(self, frame_index):
        """Find raw GT rows for an integer frame index without advancing the cursor.

        Parameters
        ----------
        frame_index : int
            Requested frame index.

        Returns
        -------
        list[list]
            The stored row group, or an empty list if the frame is absent. Rows are
            not copied; box columns remain text as returned by ``GTIO.loadGT()``.
        """
        gt_frame = []
        if int(frame_index) in self.gt_frames_list:
            found_index = self.gt_frames_list.index(frame_index)
            gt_frame = self.gt_frames[found_index]
        return gt_frame

    def getPeople(self, img, visual=False):
        """Convert the current GT frame to people and advance the cursor by one.

        Parameters
        ----------
        img : ``numpy.ndarray``
            BGR frame array. Passed through, and modified in place when drawing.
        visual : bool
            Defaults to ``False``. Draw representative points and bounding boxes.

        Returns
        -------
        list[pyppbox.utils.persontools.Person]
            Fresh people with integer box arrays. A missing frame returns an empty list.
            In detection-only mode, IDs start at zero within each frame and identity
            names are unknown. Full-GT mode copies names to both identity fields; numeric
            CIDs use the built-in GTA identity mapping and are -1 for other names.
        ``numpy.ndarray``
            The input array, with drawings if requested.

        Notes
        -----
        Call ``setGT()`` first. Cursor advancement also happens on frames with no rows.
        This does not infer the frame index from the image or its filename.
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
