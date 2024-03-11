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


# Logging
from pyppbox.utils.logtools import add_info_log, add_warning_log, add_error_log

# Common
import cv2
from collections import Counter

# Configurations
from pyppbox.config.configtools import isDictString, getCFGDict
from pyppbox.config.myconfig import (
    MyConfigurator, NoneCFG,
    DCFGYOLOCLS, DCFGYOLOULT, DCFGGT, 
    TCFGCentroid, TCFGSORT, TCFGDeepSORT, 
    RCFGFaceNet, RCFGTorchreid, 
)

# Classes & tools
from pyppbox.utils.persontools import Person
from pyppbox.utils.gttools import GTInterpreter
from pyppbox.utils.evatools import NothingDetecter, NothingTracker, NothingReider, TKOReider
from pyppbox.utils.commontools import getAbsPathFDS, isExist, getCVMat, getAncestorDir


__none_cfg__ = NoneCFG()
__none_cfg__.set("Fiat Moneey")

class MT(object):

    """An all-in-one class designed for easy detect, track, and reid people in a single threading 
    or multithreading application.

    Example:
    
    >>> import cv2
    >>> import threading
    >>> from pyppbox.utils.visualizetools import visualizePeople
    >>> from pyppbox.standalone import MT
    >>> 
    >>> def ppb_task(input, main_configs, name="Task"):
    >>>     ppbmt = MT() # Use `MT` for multithreading
    >>>     ppbmt.setMainModules(main_yaml=main_configs)
    >>>     cap = cv2.VideoCapture(input)
    >>>     while cap.isOpened():
    >>>         hasFrame, frame = cap.read()
    >>>         if hasFrame:
    >>>             detected_people, _ = ppbmt.detectPeople(frame, img_is_mat=True, visual=False)
    >>>             tracked_people = ppbmt.trackPeople(frame, detected_people, img_is_mat=True)
    >>>             reidentified_people, reid_count = ppbmt.reidPeople(
    >>>                 frame,
    >>>                 tracked_people,
    >>>                 img_is_mat=True
    >>>             )
    >>>             visualized_mat = visualizePeople(
    >>>                 frame,
    >>>                 reidentified_people,
    >>>                 show_reid=reid_count
    >>>             )
    >>>             cv2.imshow("Multithreading (" + name + ")", visualized_mat)
    >>>             if cv2.waitKey(1) & 0xFF == ord('q'):
    >>>                 break
    >>>         else:
    >>>             break
    >>>     cap.release()
    >>> 
    >>> if __name__ == '__main__':
    >>>     input_one = "data/gta.mp4"
    >>>     input_two = "data/gta.mp4"
    >>>     main_configs_one = {'detector': 'YOLO_Classic',
    >>>                         'tracker': 'SORT',
    >>>                         'reider': 'Torchreid'}
    >>>     main_configs_two = {'detector': 'YOLO_Classic',
    >>>                         'tracker': 'Centroid',
    >>>                         'reider': 'FaceNet'}
    >>>     thread_one = threading.Thread(target=ppb_task, args=(input_one, main_configs_one, "Task 1"))
    >>>     thread_two = threading.Thread(target=ppb_task, args=(input_two, main_configs_two, "Task 2"))
    >>>     thread_one.start()
    >>>     thread_two.start()
    >>>     thread_one.join()
    >>>     thread_two.join()
    >>> 

    """

    def __init__(self):
        # config
        self.__cfg__ = MyConfigurator()
        self.__unistrings__ = self.__cfg__.unified_strings
        self.__cfg_is_set__ = False
        # detector
        self.__dt_is_set__ = False
        self.__dt_cfg__ = []
        self.__dt__ = []
        # tracker
        self.__tk_is_set__ = False
        self.__tk_cfg__ = []
        self.__tk__ = []
        # reider
        self.__ri_is_set__ = False
        self.__ri_cfg__ = []
        self.__ri__ = []
        self.__deepidlistTMP__ = []
        self.__faceidlistTMP__ = []


    ###########################################
    # Configurator
    ###########################################

    def __setInternalCFGDir__(self, load_all):
        self.__cfg__.__init__()
        self.__cfg__.setMainModules()
        if load_all:
            add_info_log("---PYPPBOX : DT='" + 
                        self.__cfg__.mcfg.detector + "', TK='" + 
                        self.__cfg__.mcfg.tracker + "', RI='" + 
                        self.__cfg__.mcfg.reider + "'")
            self.__loadDefaultDetector__()
            self.__loadDefaultTracker__()
            self.__loadDefaultReIDer__()

    def setConfigDir(self, config_dir=None, load_all=False):
        """Set configurations by a pointing to a config directory :obj:`config_dir`, 
        where stores 4 required YAML files:
        (1) main.yaml, tells what main detector/tracker/reider are chosen.
        (2) detectors.yaml, stores all detectors' configurations.
        (3) trackers.yaml, stores all trackers' configurations.
        (4) reiders.yaml, stores all reiders' configurations.

        Note: JSON file (.josn) is also supported.

        Parameters
        ----------
        config_dir : str, default=None
            A path of a config directory.
            Set :code:`config_dir=None` to use the internal config directory 
            :code:'{pyppbox root}/config/cfg'.
        load_all : bool, default=False
            Set :code:`load_all=True` to set and load the selected detector/tracker/reider 
            according to the main configurations. 
            Set :code:`load_all=False` to select and load a detector/tracker/reider manually later.
        """
        self.__cfg_is_set__ = False
        if config_dir == None:
            add_info_log("---PYPPBOX : Now use the internal config directory")
            self.__cfg_is_set__ = True
            self.__setInternalCFGDir__(load_all=load_all)
        elif isinstance(config_dir, str):
            if isExist(config_dir):
                add_info_log("---PYPPBOX : Now use custom config dir, config_dir='" 
                             + str(config_dir) + "'")
                self.__cfg__.setCustomCFG(cfg_dir=config_dir)
                self.__cfg__.setMainModules()
                self.__cfg_is_set__ = True
                if load_all:
                    add_info_log("---PYPPBOX : DT='" + 
                                 self.__cfg__.mcfg.detector + "', TK='" + 
                                 self.__cfg__.mcfg.tracker + "', RI='" + 
                                 self.__cfg__.mcfg.reider + "'")
                    self.__loadDefaultDetector__()
                    self.__loadDefaultTracker__()
                    self.__loadDefaultReIDer__()
            else:
                add_warning_log("---PYPPBOX : config_dir='" + str(config_dir) + "' does not exist")
                add_warning_log("---PYPPBOX : Switched to internal config directory")
                self.__cfg_is_set__ = True
                self.__setInternalCFGDir__(load_all=load_all)
        else:
            msg = "PYPPBOX : setConfigDir() -> config_dir='" + str(config_dir) + "' is not valid."
            add_error_log(msg)
            raise ValueError(msg)

    def setMainModules(self, main_yaml=None, load_all=True):
        """Load and set the main detector, the main tracker, and the main reider all at once 
        according to the given main configurations, :obj:`main_yaml`. If the :func:`setConfigDir()` 
        has not yet been called, internal config directory will be used.

        Parameters
        ----------
        main_yaml : str or dict, default=None
            A YAML/JSON file path, or a raw/ready dictionary of the main configurations or main.yaml. 
            This :obj:`main_yaml` helps overwrite the original one configured earlier. 
            Leave it as default :obj:`main_yaml=None`, to load and set according to the configurations 
            in the config directory.
        load_all : bool, default=True
            Set :code:`load_all=True` to set and load the selected detector/tracker/reider according 
            to the main configurations, which it is meant for using this :func:`setMainModules()` method.
            Set :code:`load_all=False` to select and load a detector/tracker/reider manually later.
        """
        if not self.__cfg_is_set__: self.setConfigDir()
        self.__cfg__.setMainModules(main_yaml=main_yaml)
        if load_all: 
            add_info_log("---PYPPBOX : DT='" + 
                         self.__cfg__.mcfg.detector + "', TK='" + 
                         self.__cfg__.mcfg.tracker + "', RI='" + 
                         self.__cfg__.mcfg.reider + "'")
            self.__loadDefaultDetector__()
            self.__loadDefaultTracker__()
            self.__loadDefaultReIDer__()

    def getConfig(self):
        """
        Get the current :class:`MyConfigurator` object.
        
        Returns
        -------
        MyConfigurator
            A :class:`MyConfigurator` object used to store and manage all the configurations of pyppbox.
        """
        return self.__cfg__


    ###########################################
    # Detector
    ###########################################

    def __setGTDTOnly__(self):
        if self.__dt_is_set__:
            if self.__dt_cfg__.dt_name.lower() == self.__unistrings__.gt:
                if self.__tk_is_set__ or self.__ri_is_set__:
                    self.__dt__.setDetectOnly(self.__unistrings__.unk_fid, self.__unistrings__.unk_did)
                    msg = ("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                        "---PYPPBOX : For DT='GT', if (TK!='None' or RI!='None')\n"
                        "---PYPPBOX : -> Set detect_only=True for GT, DETECT ONLY mode.\n"
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                    add_info_log(msg, add_new_line=True)

    def __revokeGTDTOnly__(self):
        if self.__dt_is_set__:
            if self.__dt_cfg__.dt_name.lower() == self.__unistrings__.gt:
                if self.__tk_is_set__ and self.__ri_is_set__:
                    if (self.__tk_cfg__.tk_name.lower() == self.__unistrings__.none and 
                        self.__ri_cfg__.ri_name.lower() == self.__unistrings__.none):
                        self.__dt__.setDetectOnly(self.__unistrings__.unk_fid, self.__unistrings__.unk_did, 
                                                  detect_only=False)
                        msg = ("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                            "---PYPPBOX : TK='None' & RI='None'\n"
                            "---PYPPBOX : -> Overwrite detect_only=False for GT, FULL GT mode."
                            "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                        add_info_log(msg, add_new_line=True)

    def forceFullGTMode(self):
        """Normally when :code:`DT='GT'`, pyppbox can automatically decide the GT mode based on the 
        name of the tracker and/or the name of reider; however, if the decision is not satisfied 
        (Should not happen), calling this :func:`forceFUllGTMode()` will overwrite :code:`detect_only=False`.
        """
        success = False
        if self.__dt_is_set__:
            if self.__dt_cfg__.dt_name.lower() == self.__unistrings__.gt:
                self.__dt__.setDetectOnly(self.__unistrings__.unk_fid, self.__unistrings__.unk_did, 
                                          detect_only=False)
                success = True
                msg = ("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    "---PYPPBOX : Overwrite detect_only=False for GT, FULL GT mode."
                    "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                add_info_log(msg, add_new_line=True)

        if not success:
            msg = ("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "---PYPPBOX : DT!='GT' -> Failed to overwrite detect_only=False"
                "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            add_warning_log(msg, add_new_line=True)
            

    def __loadDefaultDetector__(self):
        if self.__cfg_is_set__:
            if self.__cfg__.mcfg.detector.lower() == self.__unistrings__.yolo_cls:
                from pyppbox.modules.detectors.yolocls import MyYOLOCLS
                self.__dt_cfg__ = self.__cfg__.dcfg_yolocs
                self.__dt__ = MyYOLOCLS(self.__dt_cfg__)
                self.__dt_is_set__ = True
            elif self.__cfg__.mcfg.detector.lower() == self.__unistrings__.yolo_ult:
                from pyppbox.modules.detectors.yoloult import MyYOLOULT
                self.__dt_cfg__ = self.__cfg__.dcfg_yolout
                self.__dt__ = MyYOLOULT(self.__dt_cfg__)
                self.__dt_is_set__ = True
            elif self.__cfg__.mcfg.detector.lower() == self.__unistrings__.gt:
                self.__dt_cfg__ = self.__cfg__.dcfg_gt
                self.__dt__ = GTInterpreter()
                self.__dt__.setGT(self.__dt_cfg__.gt_file)
                self.__dt_is_set__ = True
            elif self.__cfg__.mcfg.detector.lower() == self.__unistrings__.none:
                self.__dt_cfg__ = __none_cfg__
                self.__dt__ = NothingDetecter()
                self.__dt_is_set__ = True
            else: 
                add_info_log("---PYPPBOX : The input detecor is not recognized.")
                self.__dt_is_set__ = False

    def __setCustomDetector__(self, detector_dict):
        if detector_dict:
            if detector_dict['dt_name'].lower() == self.__unistrings__.yolo_cls:
                from pyppbox.modules.detectors.yolocls import MyYOLOCLS
                self.__dt_cfg__ = DCFGYOLOCLS()
                self.__dt_cfg__.set(detector_dict)
                self.__dt__ = MyYOLOCLS(self.__dt_cfg__)
                self.__dt_is_set__ = True
                add_info_log("---PYPPBOX : Set detector='" + self.__dt_cfg__.dt_name + "'")
            elif detector_dict['dt_name'].lower() == self.__unistrings__.yolo_ult:
                from pyppbox.modules.detectors.yoloult import MyYOLOULT
                self.__dt_cfg__ = DCFGYOLOULT()
                self.__dt_cfg__.set(detector_dict)
                self.__dt__ = MyYOLOULT(self.__dt_cfg__)
                self.__dt_is_set__ = True
                add_info_log("---PYPPBOX : Set detector='" + self.__dt_cfg__.dt_name + "'")
            elif detector_dict['dt_name'].lower() == self.__unistrings__.gt:
                self.__dt_cfg__ = DCFGGT()
                self.__dt_cfg__.set(detector_dict)
                self.__dt__ = GTInterpreter()
                self.__dt__.setGT(self.__dt_cfg__.gt_file)
                self.__dt_is_set__ = True
                add_info_log("---PYPPBOX : Set detector='" + self.__dt_cfg__.dt_name + "'")
            elif detector_dict['dt_name'].lower() == self.__unistrings__.none:
                self.__dt_cfg__ = __none_cfg__
                self.__dt__ = NothingDetecter()
                self.__dt_is_set__ = True
                add_info_log("---PYPPBOX : Set detector='" + self.__dt_cfg__.dt_name + "'")
            else: 
                self.__dt_is_set__ = False
                add_warning_log("---PYPPBOX : detector='" + detector_dict['dt_name'] + 
                                "' is not recognized.")

    def setMainDetector(self, detector=""):
        """Set the main detector by a supported name, a raw/ready dictionary, or a YAML/JSON file. 
        Calling :func:`setConfigDir()` before :func:`setMainTracker()` is optional. Different from 
        the rest, setting the main detector by its name results in loading the configurations from 
        the config directory set by the last :func:`setConfigDir()`. If :func:`setConfigDir()` has 
        not been called before, setting the main detector by a supported name results in referencing 
        the internal config directory in order to load the corresponding configurations. 

        Parameters
        ----------
        detector : str or dict, default=""
            (1) Set :code:`detector=""` to set the main detector according to the main configurations 
            or main.yaml and load its configurations from the detectors.yaml. 
            (2) Set :code:`detector="YOLO_Classic"`.etc, to set YOLO Classic as the main detector and 
            load its configurations from detectors.yaml.
            (3) Set a raw string or ready dictionary is also possible; for example, 
            :code:`detector="[{'dt_name': 'YOLO_Ultralytics', 'conf': 0.5, 'iou': 0.7, 'imgsz': 416, 
            'show_boxes': True, 'device': 0, 'max_det': 100, 'line_width': 500, 
            'model_file': 'data/modules/yolo_ultralytics/yolov8l-pose.pt', 
            'repspoint_calibration': 0.25}]"`.
            (4) Set :code:`detector="a_supported_detector.yaml"` or :code:`detector="a_supported_detector.json"` 
            to set the main detector and its configuration from a YAML file. 
        """
        self.__dt_is_set__ = False
        self.__dt__ = []
        if isinstance(detector, dict):
            self.__setCustomDetector__(detector)
        elif isinstance(detector, str):
            if (isDictString(detector) or "yaml" in detector.lower() or 
                "json" in detector.lower()):
                self.__setCustomDetector__(getCFGDict(detector))
            elif detector.lower() == "":
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__loadDefaultDetector__()
                add_info_log("---PYPPBOX : Use detector according to the \"main.yaml\"")
            elif detector.lower() == self.__unistrings__.yolo_cls:
                from pyppbox.modules.detectors.yolocls import MyYOLOCLS
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllDCFG()
                self.__dt_cfg__ = self.__cfg__.dcfg_yolocs
                self.__dt__ = MyYOLOCLS(self.__dt_cfg__)
                self.__dt_is_set__ = True
                add_info_log("---PYPPBOX : Set detector='" + str(detector) + "'")
            elif detector.lower() == self.__unistrings__.yolo_ult:
                from pyppbox.modules.detectors.yoloult import MyYOLOULT
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllDCFG()
                self.__dt_cfg__ = self.__cfg__.dcfg_yolout
                self.__dt__ = MyYOLOULT(self.__dt_cfg__)
                self.__dt_is_set__ = True
                add_info_log("---PYPPBOX : Set detector='" + str(detector) + "'")
            elif detector.lower() == self.__unistrings__.gt:
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllDCFG()
                self.__dt_cfg__ = self.__cfg__.dcfg_gt
                self.__dt__ = GTInterpreter()
                self.__dt__.setGT(self.__dt_cfg__.gt_file)
                self.__dt_is_set__ = True
                add_info_log("---PYPPBOX : Set detector='" + str(detector) + "'")
            elif detector.lower() == self.__unistrings__.none:
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllDCFG()
                self.__dt_cfg__ = __none_cfg__
                self.__dt__ = NothingDetecter()
                self.__dt_is_set__ = True
                add_info_log("---PYPPBOX : Set detector='" + str(detector) + "'")
        else:
            add_warning_log("---PYPPBOX : detector='" + str(detector) + "' is not recognized.")

    def detectPeople(self, 
                     img, 
                     img_is_mat=False, 
                     visual=False, 
                     save=False, 
                     save_file="", 
                     min_width_filter=35,
                     alt_repspoint=False, 
                     alt_repspoint_top=True): 
        """Detect people by giving an image. :func:`setConfigDir()` or :func:`setMainDetector()` must 
        be called in advance.

        Parameters
        ----------
        img : str or Mat
            Set an image file or a cv :obj:`Mat`.
        img_is_mat : bool, default=False
            Speed up the function by telling whether the :obj:`img` is cv :obj:`Mat`.
        visual : bool, default=False
            Decide whether to visualize like drawing bounding boxes and keypoints to the 
            return :obj:`img`.
        save : bool, default=False
            Decide whether to save the return :obj:`img` to a JPG file.
        save_file : str, default=""
            Indicate a path of where to save the processed image.
        min_width_filter : int, default=35
            Mininum width filter of a detected person.
        alt_repspoint : bool, default=False
            An indication of whether to use the alternative :meth:`findRepspointBB`.
        alt_repspoint_top : bool, default=True
            A parameter passed to :obj:`prefer_top` of :meth:`findRepspointBB`.
        
        Returns
        -------
        list[Person, ...]
            A  list of :class:`Person` object.
        Mat
            A cv :obj:`Mat` image.
        """ 
        people = []
        if self.__dt_is_set__: 
            if not isinstance(self.__dt__, NothingDetecter):
                if not img_is_mat: img = getCVMat(img)
                if (self.__dt_cfg__.dt_name.lower() == self.__unistrings__.yolo_cls or 
                    self.__dt_cfg__.dt_name.lower() == self.__unistrings__.yolo_ult):
                    people, img = self.__dt__.detectPeople(img, 
                                                           visual=visual, 
                                                           min_width_filter=min_width_filter, 
                                                           alt_repspoint=alt_repspoint, 
                                                           alt_repspoint_top=alt_repspoint_top)
                elif self.__dt_cfg__.dt_name.lower() == self.__unistrings__.gt:
                    people, img = self.__dt__.getPeople(img, visual=visual)
                if save:
                    if isExist(getAncestorDir(str(save_file))):
                        filename = getAbsPathFDS(str(save_file))
                        cv2.imwrite(filename=filename, img=img)
                    else:
                        msg = ("PYPPBOX : detectPeople() -> save_file='" + 
                               str(save_file) + "' is not valid.")
                        add_error_log(msg)
                        raise ValueError(msg)
        else:
            add_warning_log("---PYPPBOX : detectPeople() -> The main detector is not set.")
        return people, img


    ###########################################
    # Tracker
    ###########################################

    def __loadDefaultTracker__(self):
        if self.__cfg_is_set__:
            if self.__cfg__.mcfg.tracker.lower() == self.__unistrings__.centroid:
                from pyppbox.modules.trackers.centroid import MyCentroid
                self.__tk_cfg__ = self.__cfg__.tcfg_centroid
                self.__tk__ = MyCentroid(self.__tk_cfg__)
                self.__tk_is_set__ = True
                self.__setGTDTOnly__()
            elif self.__cfg__.mcfg.tracker.lower() == self.__unistrings__.sort:
                from pyppbox.modules.trackers.sort import MySORT
                self.__tk_cfg__ = self.__cfg__.tcfg_sort
                self.__tk__ = MySORT(self.__tk_cfg__)
                self.__tk_is_set__ = True
                self.__setGTDTOnly__()
            elif self.__cfg__.mcfg.tracker.lower() == self.__unistrings__.deepsort:
                from pyppbox.modules.trackers.deepsort import MyDeepSORT
                self.__tk_cfg__ = self.__cfg__.tcfg_deepsort
                self.__tk__ = MyDeepSORT(self.__tk_cfg__)
                self.__tk_is_set__ = True
                self.__setGTDTOnly__()
            elif self.__cfg__.mcfg.tracker.lower() == self.__unistrings__.none:
                self.__tk_cfg__ = __none_cfg__
                self.__tk__ = NothingTracker()
                self.__tk_is_set__ = True
            else: 
                add_warning_log("---PYPPBOX : The input tracker is not recognized.")
                self.__tk_is_set__ = False

    def __setCustomTracker__(self, tracker_dict):
        if tracker_dict:
            if tracker_dict['tk_name'].lower() == self.__unistrings__.centroid:
                from pyppbox.modules.trackers.centroid import MyCentroid
                self.__tk_cfg__ = TCFGCentroid()
                self.__tk_cfg__.set(tracker_dict)
                self.__tk__ = MyCentroid(self.__tk_cfg__)
                self.__tk_is_set__ = True
                self.__setGTDTOnly__()
                add_info_log("---PYPPBOX : Set tracker='" + self.__tk_cfg__.tk_name + "'")
            elif tracker_dict['tk_name'].lower() == self.__unistrings__.sort:
                from pyppbox.modules.trackers.sort import MySORT
                self.__tk_cfg__ = TCFGSORT()
                self.__tk_cfg__.set(tracker_dict)
                self.__tk__ = MySORT(self.__tk_cfg__)
                self.__tk_is_set__ = True
                self.__setGTDTOnly__()
                add_info_log("---PYPPBOX : Set tracker='" + self.__tk_cfg__.tk_name + "'")
            elif tracker_dict['tk_name'].lower() == self.__unistrings__.deepsort:
                from pyppbox.modules.trackers.deepsort import MyDeepSORT
                self.__tk_cfg__ = TCFGDeepSORT()
                self.__tk_cfg__.set(tracker_dict)
                self.__tk__ = MyDeepSORT(self.__tk_cfg__)
                self.__tk_is_set__ = True
                self.__setGTDTOnly__()
                add_info_log("---PYPPBOX : Set tracker='" + self.__tk_cfg__.tk_name + "'")
            elif tracker_dict['tk_name'].lower() == self.__unistrings__.none:
                self.__tk_cfg__ = __none_cfg__
                self.__tk__ = NothingTracker()
                self.__tk_is_set__ = True
                add_info_log("---PYPPBOX : Set tracker='" + self.__tk_cfg__.tk_name + "'")
            else:
                self.__tk_is_set__ = False
                add_warning_log("---PYPPBOX : tracker='" + tracker_dict['tk_name'] + 
                                "' is not recognized")

    def setMainTracker(self, tracker=""):
        """Set the main tracker by a supported name, a raw/ready dictionary, or a YAML/JSON file. 
        Calling :func:`setConfigDir()` before :func:`setMainTracker()` is optional. Different from 
        the rest, setting the main tracker by its name results in loading the configurations from 
        the config directory set by the last :func:`setConfigDir()`. If :func:`setConfigDir()` has 
        not been called before, setting the main tracker by a supported name results in referencing 
        the internal config directory in order to load the corresponding configurations. 

        Parameters
        ----------
        tracker : str or dict, default=""
            (1) Set :code:`tracker=""` to set the main tracker according to the main configurations or 
            main.yaml and load its configurations from the trackers.yaml. 
            (2) Set :code:`tracker="Centroid"`.etc, to set Centroid as the main tracker and load its 
            configurations from trackers.yaml.
            (3) Set a raw string or ready dictionary is also possible; for example, 
            :code:`tracker="[{'tk_name': 'SORT', 'max_age': 1, 'min_hits': 3, 'iou_threshold': 0.3}]"`.
            (4) Set :code:`tracker="a_supported_tracker.yaml"` or :code:`tracker="a_supported_tracker.json"` 
            to set the main tracker and its configuration from a YAML file. 
        """
        self.__tk_is_set__ = False
        self.__tk__ = []
        if isinstance(tracker, dict):
            self.__setCustomTracker__(tracker)
        elif isinstance(tracker, str):
            if (isDictString(tracker) or "yaml" in tracker.lower() or 
                "json" in tracker.lower()):
                self.__setCustomTracker__(getCFGDict(tracker))
            elif tracker.lower() == "default":
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__loadDefaultTracker__()
                add_info_log("---PYPPBOX : Use tracker according to the \"main.yaml\"")
            elif tracker.lower() == self.__unistrings__.centroid:
                from pyppbox.modules.trackers.centroid import MyCentroid
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllTCFG()
                self.__tk_cfg__ = self.__cfg__.tcfg_centroid
                self.__tk__ = MyCentroid(self.__tk_cfg__)
                self.__tk_is_set__ = True
                self.__setGTDTOnly__()
                add_info_log("---PYPPBOX : Set tracker='" + str(tracker) + "'")
            elif tracker.lower() == self.__unistrings__.sort:
                from pyppbox.modules.trackers.sort import MySORT
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllTCFG()
                self.__tk_cfg__ = self.__cfg__.tcfg_sort
                self.__tk__ = MySORT(self.__tk_cfg__)
                self.__tk_is_set__ = True
                self.__setGTDTOnly__()
                add_info_log("---PYPPBOX : Set tracker='" + str(tracker) + "'")
            elif tracker.lower() == self.__unistrings__.deepsort:
                from pyppbox.modules.trackers.deepsort import MyDeepSORT
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllTCFG()
                self.__tk_cfg__ = self.__cfg__.tcfg_deepsort
                self.__tk__ = MyDeepSORT(self.__tk_cfg__)
                self.__tk_is_set__ = True
                self.__setGTDTOnly__()
                add_info_log("---PYPPBOX : Set tracker='" + str(tracker) + "'")
            elif tracker.lower() == self.__unistrings__.none:
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllTCFG()
                self.__tk_cfg__ = __none_cfg__
                self.__tk__ = NothingTracker()
                self.__tk_is_set__ = True
                add_info_log("---PYPPBOX : Set tracker='" + str(tracker) + "'")
        else:
            add_warning_log("---PYPPBOX : tracker='" + str(tracker) + "' is not valid")

    def trackPeople(self, img, people, img_is_mat=False):
        """Track people by giving an image and a list of detected people. :func:`setConfigDir()` 
        or :func:`setMainTracker()` must be called in advance.

        Parameters
        ----------
        img : str or Mat
            Set an image file or a cv :obj:`Mat`.
        people : list[Person, ...]
            Set a list of :class:`Person` object which stores the detected people in the input :obj:`img`.
        img_is_mat : bool, default=False
            Speed up the function by telling whether the :obj:`img` is cv :obj:`Mat`.
        
        Returns
        -------
        list[Person, ...]
            A list of :class:`Person` object which stores people with updated IDs.
        """
        res = []
        if self.__tk_is_set__: 
            if isinstance(people, list):
                if not img_is_mat: img = getCVMat(img)
                res = self.__tk__.update(people, img=img)
            else:
                msg = "PYPPBOX : trackPeople() -> Input 'people' is not correct."
                add_error_log(msg)
                raise ValueError(msg)
        else:
            add_warning_log("---PYPPBOX : trackPeople() -> The main tracker is not set.")
        return res


    ###########################################
    # REIDer
    ###########################################

    def __loadDefaultReIDer__(self, auto_load=True):
        if self.__cfg_is_set__:
            if self.__cfg__.mcfg.reider.lower() == self.__unistrings__.facenet:
                from pyppbox.modules.reiders.facenet import MyFaceNet
                self.__ri_cfg__ = self.__cfg__.rcfg_facenet
                self.__ri__ = MyFaceNet(self.__ri_cfg__, auto_load=auto_load)
                self.__ri_is_set__ = True
                self.__setGTDTOnly__()
            elif self.__cfg__.mcfg.reider.lower() == self.__unistrings__.torchreid:
                from pyppbox.modules.reiders.torchreid import MyTorchreid
                self.__ri_cfg__ = self.__cfg__.rcfg_torchreid
                self.__ri__ = MyTorchreid(self.__ri_cfg__, auto_load=auto_load)
                self.__ri_is_set__ = True
                self.__setGTDTOnly__()
            elif self.__cfg__.mcfg.reider.lower() == self.__unistrings__.none:
                if (self.__dt_cfg__.dt_name.lower() != self.__unistrings__.none and 
                    self.__tk_cfg__.tk_name.lower() != self.__unistrings__.none):
                    self.__ri__ = TKOReider(static=True)
                else:
                    self.__ri__ = NothingReider()
                self.__ri_cfg__ = __none_cfg__
                self.__ri_is_set__ = True
                self.__revokeGTDTOnly__()
            else:
                add_warning_log("---PYPPBOX : The input reider is not recognized.")
                self.__ri_is_set__ = False

    def __setCustomReIDer__(self, reider_dict, auto_load=True):
        if reider_dict:
            if reider_dict['ri_name'].lower() == self.__unistrings__.facenet:
                from pyppbox.modules.reiders.facenet import MyFaceNet
                self.__ri_cfg__ = RCFGFaceNet()
                self.__ri_cfg__.set(reider_dict)
                self.__ri__ = MyFaceNet(self.__ri_cfg__, auto_load=auto_load)
                self.__ri_is_set__ = True
                add_info_log("---PYPPBOX : Set reider='" + self.__ri_cfg__.ri_name + "'")
                self.__setGTDTOnly__()
            elif reider_dict['ri_name'].lower() == self.__unistrings__.torchreid:
                from pyppbox.modules.reiders.torchreid import MyTorchreid
                self.__ri_cfg__ = RCFGTorchreid()
                self.__ri_cfg__.set(reider_dict)
                self.__ri__ = MyTorchreid(self.__ri_cfg__, auto_load=auto_load)
                self.__ri_is_set__ = True
                add_info_log("---PYPPBOX : Set reider='" + self.__ri_cfg__.ri_name + "'")
                self.__setGTDTOnly__()
            elif reider_dict['ri_name'].lower() == self.__unistrings__.none:
                if (self.__dt_cfg__.dt_name.lower() != self.__unistrings__.none and 
                    self.__tk_cfg__.tk_name.lower() != self.__unistrings__.none):
                    self.__ri__ = TKOReider(static=True)
                else:
                    self.__ri__ = NothingReider()
                self.__ri_cfg__ = __none_cfg__
                self.__ri_is_set__ = True
                add_info_log("---PYPPBOX : Set reider='" + self.__ri_cfg__.ri_name + "'")
                self.__revokeGTDTOnly__()
            else :
                self.__ri_is_set__ = False
                add_warning_log("---PYPPBOX : reider='" + reider_dict['ri_name'] + 
                                "' is not recognized")

    def setMainReIDer(self, reider="", auto_load=True):
        """Set the main reider by a supported name, a raw/ready dictionary, or a YAML/JSON file. 
        Calling :func:`setConfigDir()` before :func:`setMainTracker()` is optional. Different from 
        the rest, setting the main reider by its name results in loading the configurations from the 
        config directory set by the last :func:`setConfigDir()`. If :func:`setConfigDir()` has not 
        been called before, setting the main reider by a supported name results in referencing the 
        internal config directory in order to load the corresponding configurations. 

        Parameters
        ----------
        reider : str or dict, default=""
            (1) Set :code:`reider=""` to set the main reider according to the main configurations 
            or main.yaml and load its configurations from the reiders.yaml. 
            (2) Set :code:`reider="FaceNet"`.etc, to set FaceNet as a reider and load its 
            configurations from reiders.yaml.
            (3) Set a raw string or ready dictionary is also possible; for example, 
            :code:`reider="[{'ri_name': 'Torchreid', 
            'classifier_pkl': 'data/modules/torchreid/classifier/gta5_osnet_ain_ms_d_c.pkl', 
            'train_data': 'data/datasets/GTA_V_DATASET/body_128x256', 'model_name': 'osnet_ain_x1_0', 
            'model_path': 'data/modules/torchreid/models/torchreid/osnet_ain_ms_d_c.pth.tar', 
            'min_confidence': 0.35, 'device': 'cuda'}]"`.
            (4) Set :code:`reider="a_supported_reider.yaml"` or :code:`reider="a_supported_reider.json"` 
            to set a the main reider and its configurations from a YAML file. 
        auto_load : bool, default=True
            All supported reiders are Pytorch or Tensorflow based module, thus they need to 
            initial and load their models/weights. :obj:`auto_load` is used to decide whether to 
            load the reider automatically once the reider is set. Keep the :code:`auto_load=True` 
            if it is meant for using :func:`reidPeople()`; however, even if the :code:`auto_load=False`, 
            the :func:`reidPeople()` will initial and load the reider by itself, but it requires 
            some time to do so.
        """
        self.__ri_is_set__ = False
        self.__ri__ = []
        if isinstance(reider, dict):
            self.__setCustomReIDer__(reider, auto_load)
        elif isinstance(reider, str):
            if (isDictString(reider) or "yaml" in reider.lower() or 
                "json" in reider.lower()):
                self.__setCustomReIDer__(getCFGDict(reider), auto_load)
            elif reider.lower() == "default":
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__loadDefaultReIDer__(auto_load=auto_load)
                add_info_log("---PYPPBOX : Use reider according to the \"main.yaml\"")
            elif reider.lower() == self.__unistrings__.facenet:
                from pyppbox.modules.reiders.facenet import MyFaceNet
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllRCFG()
                self.__ri_cfg__ = self.__cfg__.rcfg_facenet
                self.__ri__ = MyFaceNet(self.__ri_cfg__, auto_load=auto_load)
                self.__ri_is_set__ = True
                add_info_log("---PYPPBOX : Set reider='" + str(reider) + "'")
                self.__setGTDTOnly__()
            elif reider.lower() == self.__unistrings__.torchreid:
                from pyppbox.modules.reiders.torchreid import MyTorchreid
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllRCFG()
                self.__ri_cfg__ = self.__cfg__.rcfg_torchreid
                self.__ri__ = MyTorchreid(self.__ri_cfg__, auto_load=auto_load)
                self.__ri_is_set__ = True
                add_info_log("---PYPPBOX : Set reider='" + str(reider) + "'")
                self.__setGTDTOnly__()
            elif reider.lower() == self.__unistrings__.none:
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllRCFG()
                if (self.__dt_cfg__.dt_name.lower() != self.__unistrings__.none and 
                    self.__tk_cfg__.tk_name.lower() != self.__unistrings__.none):
                    self.__ri__ = TKOReider(static=True)
                else:
                    self.__ri__ = NothingReider()
                self.__ri_cfg__ = __none_cfg__
                self.__ri_is_set__ = True
                add_info_log("---PYPPBOX : Set reider='" + str(reider) + "'")
                self.__revokeGTDTOnly__()
        else:
            add_warning_log("---PYPPBOX : reider='" + str(reider) + "' is not valid")

    def reidPeople(self, img, people, deduplicate=True, img_is_mat=False):
        """Re-identify people by giving an image and a list of detected or tracked people. 
        :func:`setConfigDir()` or :func:`setMainReIDer()` must be called in advance.

        Parameters
        ----------
        img : str or Mat
            Set an image file or a cv :obj:`Mat`.
        people : list[Person, ...]
            Set a list of :class:`Person` object which stores the detected or tracked people in 
            the input :obj:`img`.
        deduplicate : bool, default=True
            Indicate whether to re-reid people who have the same face ids or deep ids.
        img_is_mat : bool, default=False
            Speed up the function by telling whether the :obj:`img` is cv :obj:`Mat`.
        
        Returns
        -------
        list[Person, ...]
            A list of :class:`Person` object which stores people with the updated IDs.
        tuple(int, int)
            A tuple of (ReID count, ReID deduplicate count).
        """
        res = []
        reid_count = [0, 0]
        if self.__ri_is_set__:
            if self.__ri_cfg__.ri_name.lower() != self.__unistrings__.none:
                if not self.__ri__.auto_load:
                    self.__ri__.load_classifier()
                    self.__ri__.auto_load = True
            if isinstance(people, list):
                if len(people) > 0:
                    if isinstance(people[0], Person):
                        if not img_is_mat: img = getCVMat(img)
                        res, reid_count[0] = self.__reidNormal__(img, people)
                        if deduplicate: res, reid_count[1] = self.__reidDupkiller__(img, res)
                    else:
                        msg = "PYPPBOX : reidPeople() -> Input 'people' has unsupported element."
                        add_error_log(msg)
                        raise ValueError(msg)
            else:
                msg = "PYPPBOX : reidPeople() -> The input 'people' is invalid."
                add_error_log(msg)
                raise ValueError(msg)
        else:
            add_warning_log("---PYPPBOX : reidPeople() -> The main ReIDer is not set.")
        return res, tuple(reid_count)

    def __reidNormal__(self, img, people):
        if self.__ri_cfg__.ri_name.lower() == self.__unistrings__.facenet:
            return self.__reidFaceNormal__(img, people)
        elif self.__ri_cfg__.ri_name.lower() == self.__unistrings__.torchreid:
            return self.__reidDeepNormal__(img, people)
        else:
            return self.__reidEmpty__(img, people)

    def __reidDupkiller__(self, img, people):
        if self.__ri_cfg__.ri_name.lower() == self.__unistrings__.facenet:
            return self.__reidDupFacekiller__(img, people)
        elif self.__ri_cfg__.ri_name.lower() == self.__unistrings__.torchreid:
            return self.__reidDupDeepkiller__(img, people)
        else:
            return self.__reidEmpty__(img, people)

    def __reidEmpty__(self, _, people):
        index = 0
        for person in people:
            deepid = str(person.deepid)
            if self.__unistrings__.err_did in deepid or self.__unistrings__.unk_did in deepid:
                people[index].deepid = self.__ri__.recognize(self.__unistrings__.unk_did)
            index += 1
        return people, 0

    def __reidDeepNormal__(self, img, people):
        reid_count = 0
        index = 0
        self.__deepidlistTMP__ = []
        for person in people:
            deepid = person.deepid
            if self.__unistrings__.err_did in deepid or self.__unistrings__.unk_did in deepid:
                miniframe = img.copy()
                try:
                    [x1, y1, x2, y2] = person.box_xyxy
                    miniframe = miniframe[y1:y2, x1:x2]
                    people[index].deepid, people[index].deepid_conf = self.__ri__.recognize(
                        cv2.resize(miniframe, self.__ri_cfg__.model_wh), 
                        is_bgr=True
                    )
                    reid_count += 1
                except Exception as e:
                    add_warning_log("---PYPPBOX : __reidDeepNormal__() -> " + str(e))
            self.__deepidlistTMP__.append(deepid)
            index += 1
        return people, reid_count

    def __reidDupDeepkiller__(self, img, people):
        reid_count = 0
        if len(self.__deepidlistTMP__) != len(set(self.__deepidlistTMP__)):
            ddeepids = [k for k, v in Counter(self.__deepidlistTMP__).items() if v > 1]
            for ddeepid in ddeepids:
                index = 0
                for person in people:
                    try:
                        if person.deepid == ddeepid:
                            [x1, y1, x2, y2] = person.box_xyxy
                            miniframe = img.copy()
                            miniframe = miniframe[y1:y2, x1:x2]
                            people[index].deepid, people[index].deepid_conf = self.__ri__.recognize(
                                cv2.resize(miniframe, self.__ri_cfg__.model_wh), 
                                is_bgr=True
                            )
                            reid_count += 1
                    except Exception as e:
                        add_warning_log("---PYPPBOX : __reidDupDeepkiller__() -> " + str(e))
                    index += 1
        return people, reid_count

    def __reidFaceNormal__(self, img, people):
        reid_count = 0
        index = 0
        self.__faceidlistTMP__ = []
        for person in people:
            faceid = person.faceid
            if self.__unistrings__.err_fid in faceid or self.__unistrings__.unk_fid in faceid:
                (x, y) = person.repspoint
                miniframe = img.copy()
                try:
                    miniframe = miniframe[
                        y + int(self.__cfg__.rcfg_facenet.yl_h_calibration[0]):
                        y + int(self.__cfg__.rcfg_facenet.yl_h_calibration[1]), 
                        x + int(self.__cfg__.rcfg_facenet.yl_w_calibration[0]):
                        x + int(self.__cfg__.rcfg_facenet.yl_w_calibration[1])
                    ]
                    people[index].faceid, people[index].faceid_conf = self.__ri__.recognize(
                        miniframe, 
                        is_bgr=True
                    )
                    reid_count += 1
                except Exception as e:
                    add_warning_log("---PYPPBOX : __reidFaceNormal__() -> " + str(e))
            self.__faceidlistTMP__.append(faceid)
            index += 1
        return people, reid_count

    def __reidDupFacekiller__(self, img, people):
        reid_count = 0
        if len(self.__faceidlistTMP__) != len(set(self.__faceidlistTMP__)):
            dfaceids = [k for k, v in Counter(self.__faceidlistTMP__).items() if v > 1]
            for dfaceid in dfaceids:
                index = 0
                for person in people:
                    try:
                        if person.faceid == dfaceid:
                            (x, y) = person.repspoint
                            miniframe = img.copy()
                            miniframe = miniframe[
                                y + int(self.__cfg__.rcfg_facenet.yl_h_calibration[0]):
                                y + int(self.__cfg__.rcfg_facenet.yl_h_calibration[1]), 
                                x + int(self.__cfg__.rcfg_facenet.yl_w_calibration[0]):
                                x + int(self.__cfg__.rcfg_facenet.yl_w_calibration[1])
                            ]
                            people[index].faceid, people[index].faceid_conf = self.__ri__.recognize(
                                miniframe, 
                                is_bgr=True
                            )
                            reid_count += 1
                    except Exception as e:
                        add_warning_log("---PYPPBOX : __reidDupFacekiller__() -> " + str(e))
                    index += 1
        return people, reid_count

    def trainReIDClassifier(self, reider="Default", train_data="", classifier_pkl=""):
        """Train classifier of a reider by pointing to a data directory. Calling 
        :func:`setConfigDir()` or :func:`setMainReIDer()` in advance is not required.

        Parameters
        ----------
        reider : str or dict, default="Default" 
            A supported name, a raw/ready dictionary, or a YAML/JSON file which is passed to 
            :func:`setMainReIDer(reider=reider, auto_load=False)`.
        train_data : str, default=""
            A path of data to train, where consists of 2 or more sub-folders which classify 
            2 or more people. Set :code:`train_data=""` or keep default to use the configured 
            :obj:`train_data` according to the input :code:`reider`. All images in this the sub-folders 
            must be 128x256 for Torchreid and 182x182 for FaceNet.
        classifier_pkl : str, default=""
            A file path for the classifier PKL file. Set :code:`classifier_pkl=""` or keep default 
            to use the configured :obj:`classifier_pkl` in the input :obj:`reider`.
        """
        self.setMainReIDer(reider=reider, auto_load=False)
        if self.__ri_is_set__:
            valid_train_data = True
            valid_pkl = True
            if train_data != "":
                if isExist(train_data): 
                    self.__ri_cfg__.train_data = getAbsPathFDS(train_data)
                else: 
                    valid_train_data = False
                    add_error_log("---PYPPBOX : train_data='" + str(train_data) + 
                                  "' does not exist.")
            if classifier_pkl != "":
                if not isExist(getAncestorDir(classifier_pkl)):
                    valid_pkl = False
                    add_error_log("---PYPPBOX : classifier_pkl='" + str(classifier_pkl) + 
                                  "' is not valid.")
                if valid_pkl: self.__ri_cfg__.classifier_pkl = getAbsPathFDS(classifier_pkl)
            if valid_train_data and valid_pkl:
                from pyppbox.modules.reiders.facenet import MyFaceNet
                from pyppbox.modules.reiders.torchreid import MyTorchreid
                if isinstance(self.__ri__, MyFaceNet):
                    self.__ri__ = MyFaceNet(self.__ri_cfg__, auto_load=False)
                    add_info_log("------------- FaceNet --------------")
                elif isinstance(self.__ri__, MyTorchreid):
                    self.__ri__ = MyTorchreid(self.__ri_cfg__, auto_load=False)
                    add_info_log("------------ Torchreid -------------")
                add_info_log("---PYPPBOX : train_data='" + str(self.__ri_cfg__.train_data) + "'")
                add_info_log("---PYPPBOX : classifier_pkl='" + str(self.__ri_cfg__.classifier_pkl) + "'")
                self.__ri__.train_classifier()
