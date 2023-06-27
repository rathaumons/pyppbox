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
from pyppbox.config.configtools import isRawYAMLString, getCFGDict
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


###########################################
# Configurator
###########################################

__cfg__ = MyConfigurator()
__unistrings__ = __cfg__.unified_strings
__cfg_is_set__ = False
__none_cfg__ = NoneCFG()
__none_cfg__.set("Fiat Moneey")

def __setInternalCFGDir__(load_all):
    global __cfg__
    __cfg__.__init__()
    __cfg__.setMainModules()
    if load_all:
        add_info_log("---PYPPBOX : DT='" + 
                     __cfg__.mcfg.detector + "', TK='" + 
                     __cfg__.mcfg.tracker + "', RI='" + 
                     __cfg__.mcfg.reider + "'")
        __loadDefaultDetector__()
        __loadDefaultTracker__()
        __loadDefaultReIDer__()

def setConfigDir(config_dir=None, load_all=False):
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
    global __cfg__, __cfg_is_set__, __cfg__
    __cfg_is_set__ = False
    if config_dir == None:
        add_info_log("---PYPPBOX : Now use the internal config directory")
        __cfg_is_set__ = True
        __setInternalCFGDir__(load_all=load_all)
    elif isinstance(config_dir, str):
        if isExist(config_dir):
            add_info_log("---PYPPBOX : Now use custom config dir, config_dir='" 
                  + str(config_dir) + "'")
            __cfg__.setCustomCFG(cfg_dir=config_dir)
            __cfg__.setMainModules()
            __cfg_is_set__ = True
            if load_all:
                add_info_log("---PYPPBOX : DT='" + 
                             __cfg__.mcfg.detector + "', TK='" + 
                             __cfg__.mcfg.tracker + "', RI='" + 
                             __cfg__.mcfg.reider + "'")
                __loadDefaultDetector__()
                __loadDefaultTracker__()
                __loadDefaultReIDer__()
        else:
            add_warning_log("---PYPPBOX : config_dir='" + str(config_dir) + "' does not exist")
            add_warning_log("---PYPPBOX : Switched to internal config directory")
            __cfg_is_set__ = True
            __setInternalCFGDir__(load_all=load_all)
    else:
        msg = "PYPPBOX : setConfigDir() -> config_dir='" + str(config_dir) + "' is not valid."
        add_error_log(msg)
        raise ValueError(msg)

def setMainModules(main_yaml=None, load_all=True):
    """Load and set the main detector, the main tracker, and the main reider all at once 
    according to the given main configurations, :obj:`main_yaml`. If the :func:`setConfigDir()` 
    has not yet been called, internal config directory will be used.

    Parameters
    ----------
    main_yaml : str or dict, default=None
        A YAML/JSON file path, a raw YAML/JSON string, or ready YAML/JSON dictionary of the 
        main configurations or main.yaml. This :obj:`main_yaml` helps overwrite the original one 
        configured earlier. Leave it as default :obj:`main_yaml=None`, to load and set according 
        to the configurations in the config directory.
    load_all : bool, default=True
        Set :code:`load_all=True` to set and load the selected detector/tracker/reider according 
        to the main configurations, which it is meant for using this :func:`setMainModules()` method.
        Set :code:`load_all=False` to select and load a detector/tracker/reider manually later.
    """
    global __cfg__, __cfg_is_set__
    if not __cfg_is_set__: setConfigDir()
    __cfg__.setMainModules(main_yaml=main_yaml)
    if load_all: 
        add_info_log("---PYPPBOX : DT='" + 
                     __cfg__.mcfg.detector + "', TK='" + 
                     __cfg__.mcfg.tracker + "', RI='" + 
                     __cfg__.mcfg.reider + "'")
        __loadDefaultDetector__()
        __loadDefaultTracker__()
        __loadDefaultReIDer__()

def getConfig():
    """
    Get the current :class:`MyConfigurator` object.
    
    Returns
    -------
    MyConfigurator
        A :class:`MyConfigurator` object used to store and manage all the configurations of pyppbox.
    """
    global __cfg__
    return __cfg__


###########################################
# Detector
###########################################

__dt_is_set__ = False
__dt_cfg__ = []
__dt__ = []

def __setGTDTOnly__():
    global __dt_is_set__, __tk_is_set__, __ri_is_set__, __dt__, __dt_cfg__, __unistrings__
    if __dt_is_set__:
        if __dt_cfg__.dt_name.lower() == __unistrings__.gt:
            if __tk_is_set__ or __ri_is_set__:
                __dt__.setDetectOnly(__unistrings__.unk_fid, __unistrings__.unk_did)
                msg = ("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                       "---PYPPBOX : For DT='GT', if (TK!='None' or RI!='None')\n"
                       "---PYPPBOX : -> Set detect_only=True for GT, DETECT ONLY mode.\n"
                       "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                add_info_log(msg, add_new_line=True)

def __revokeGTDTOnly__():
    global __dt_is_set__, __tk_is_set__, __ri_is_set__
    global __dt__, __tk_cfg__, __ri_cfg__, __dt_cfg__, __unistrings__
    if __dt_is_set__:
        if __dt_cfg__.dt_name.lower() == __unistrings__.gt:
            if __tk_is_set__ and __ri_is_set__:
                if (__tk_cfg__.tk_name.lower() == __unistrings__.none and 
                    __ri_cfg__.ri_name.lower() == __unistrings__.none):
                    __dt__.setDetectOnly(__unistrings__.unk_fid, __unistrings__.unk_did, 
                                         detect_only=False)
                    msg = ("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                           "---PYPPBOX : TK='None' & RI='None'\n"
                           "---PYPPBOX : -> Overwrite detect_only=False for GT, FULL GT mode."
                           "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                    add_info_log(msg, add_new_line=True)

def forceFullGTMode():
    """Normally when :code:`DT='GT'`, pyppbox can automatically decide the GT mode based on the 
    name of the tracker and/or the name of reider; however, if the decision is not satisfied 
    (Should not happen), calling this :func:`forceFUllGTMode()` will overwrite :code:`detect_only=False`.
    """
    global __dt_is_set__, __dt__, __dt_cfg__, __unistrings__
    success = False
    if __dt_is_set__:
        if __dt_cfg__.dt_name.lower() == __unistrings__.gt:
            __dt__.setDetectOnly(__unistrings__.unk_fid, __unistrings__.unk_did, 
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
        

def __loadDefaultDetector__():
    global __unistrings__, __cfg__, __cfg_is_set__
    global __dt__, __dt_cfg__, __dt_is_set__, __none_cfg__
    if __cfg_is_set__:
        if __cfg__.mcfg.detector.lower() == __unistrings__.yolo_cls:
            from pyppbox.modules.detectors.yolocls import MyYOLOCLS
            __dt_cfg__ = __cfg__.dcfg_yolocs
            __dt__ = MyYOLOCLS(__dt_cfg__)
            __dt_is_set__ = True
        elif __cfg__.mcfg.detector.lower() == __unistrings__.yolo_ult:
            from pyppbox.modules.detectors.yoloult import MyYOLOULT
            __dt_cfg__ = __cfg__.dcfg_yolout
            __dt__ = MyYOLOULT(__dt_cfg__)
            __dt_is_set__ = True
        elif __cfg__.mcfg.detector.lower() == __unistrings__.gt:
            __dt_cfg__ = __cfg__.dcfg_gt
            __dt__ = GTInterpreter()
            __dt__.setGT(__dt_cfg__.gt_file)
            __dt_is_set__ = True
        elif __cfg__.mcfg.detector.lower() == __unistrings__.none:
            __dt_cfg__ = __none_cfg__
            __dt__ = NothingDetecter()
            __dt_is_set__ = True
        else: 
            add_info_log("---PYPPBOX : The input detecor is not recognized.")
            __dt_is_set__ = False

def __setCustomDetector__(detector_dict):
    global __unistrings__, __dt__, __dt_cfg__, __dt_is_set__, __none_cfg__
    if detector_dict:
        if detector_dict['dt_name'].lower() == __unistrings__.yolo_cls:
            from pyppbox.modules.detectors.yolocls import MyYOLOCLS
            __dt_cfg__ = DCFGYOLOCLS()
            __dt_cfg__.set(detector_dict)
            __dt__ = MyYOLOCLS(__dt_cfg__)
            __dt_is_set__ = True
            add_info_log("---PYPPBOX : Set detector='" + __dt_cfg__.dt_name + "'")
        elif detector_dict['dt_name'].lower() == __unistrings__.yolo_ult:
            from pyppbox.modules.detectors.yoloult import MyYOLOULT
            __dt_cfg__ = DCFGYOLOULT()
            __dt_cfg__.set(detector_dict)
            __dt__ = MyYOLOULT(__dt_cfg__)
            __dt_is_set__ = True
            add_info_log("---PYPPBOX : Set detector='" + __dt_cfg__.dt_name + "'")
        elif detector_dict['dt_name'].lower() == __unistrings__.gt:
            __dt_cfg__ = DCFGGT()
            __dt_cfg__.set(detector_dict)
            __dt__ = GTInterpreter()
            __dt__.setGT(__dt_cfg__.gt_file)
            __dt_is_set__ = True
            add_info_log("---PYPPBOX : Set detector='" + __dt_cfg__.dt_name + "'")
        elif detector_dict['dt_name'].lower() == __unistrings__.none:
            __dt_cfg__ = __none_cfg__
            __dt__ = NothingDetecter()
            __dt_is_set__ = True
            add_info_log("---PYPPBOX : Set detector='" + __dt_cfg__.dt_name + "'")
        else: 
            __dt_is_set__ = False
            add_warning_log("---PYPPBOX : detector='" + detector_dict['dt_name'] + 
                            "' is not recognized.")

def setMainDetector(detector=""):
    """Set the main detector by a supported name, a raw YAML/JSON string, a ready YAML/JSON 
    dictionary, or a YAML/JSON file. Calling :func:`setConfigDir()` before :func:`setMainTracker()` 
    is optional. Different from the rest, setting the main detector by its name results 
    in loading the configurations from the config directory set by the last :func:`setConfigDir()`. 
    If :func:`setConfigDir()` has not been called before, setting the main detector by a supported 
    name results in referencing the internal config directory in order to load the 
    corresponding configurations. 

    Parameters
    ----------
    detector : str or dict, default=""
        (1) Set :code:`detector=""` to set the main detector according to the main configurations 
        or main.yaml and load its configurations from the detectors.yaml. 
        (2) Set :code:`detector="YOLO_Classic"`.etc, to set YOLO Classic as the main detector and 
        load its configurations from detectors.yaml.
        (3) Set a raw JSON string or a ready JSON dictionary is also possible; for example, 
        :code:`detector="[{'dt_name': 'YOLO_Ultralytics', 'conf': 0.5, 'iou': 0.7, 'imgsz': 416, 
        'boxes': True, 'device': 0, 'max_det': 100, 'line_width': 500, 
        'model_file': 'data/modules/yolo_ultralytics/yolov8l-pose.pt', 
        'repspoint_calibration': 0.25}]"`.
        (4) Set :code:`detector="a_suported_detector.yaml"` or :code:`detector="a_suported_detector.json"` 
        to set the main detector and its configuration from a YAML file. 
    """
    global __unistrings__, __cfg__, __cfg_is_set__, __dt__
    global __dt_cfg__, __dt_is_set__, __none_cfg__
    __dt_is_set__ = False
    __dt__ = []
    if isinstance(detector, dict):
        __setCustomDetector__(detector)
    elif isinstance(detector, str):
        if (isRawYAMLString(detector) or "yaml" in detector.lower() or 
            "json" in detector.lower()):
            __setCustomDetector__(getCFGDict(detector))
        elif detector.lower() == "":
            if not __cfg_is_set__: setConfigDir()
            __loadDefaultDetector__()
            add_info_log("---PYPPBOX : Use detector according to the \"main.yaml\"")
        elif detector.lower() == __unistrings__.yolo_cls:
            from pyppbox.modules.detectors.yolocls import MyYOLOCLS
            if not __cfg_is_set__: setConfigDir()
            __cfg__.setAllDCFG()
            __dt_cfg__ = __cfg__.dcfg_yolocs
            __dt__ = MyYOLOCLS(__dt_cfg__)
            __dt_is_set__ = True
            add_info_log("---PYPPBOX : Set detector='" + str(detector) + "'")
        elif detector.lower() == __unistrings__.yolo_ult:
            from pyppbox.modules.detectors.yoloult import MyYOLOULT
            if not __cfg_is_set__: setConfigDir()
            __cfg__.setAllDCFG()
            __dt_cfg__ = __cfg__.dcfg_yolout
            __dt__ = MyYOLOULT(__dt_cfg__)
            __dt_is_set__ = True
            add_info_log("---PYPPBOX : Set detector='" + str(detector) + "'")
        elif detector.lower() == __unistrings__.gt:
            if not __cfg_is_set__: setConfigDir()
            __cfg__.setAllDCFG()
            __dt_cfg__ = __cfg__.dcfg_gt
            __dt__ = GTInterpreter()
            __dt__.setGT(__dt_cfg__.gt_file)
            __dt_is_set__ = True
            add_info_log("---PYPPBOX : Set detector='" + str(detector) + "'")
        elif detector.lower() == __unistrings__.none:
            if not __cfg_is_set__: setConfigDir()
            __cfg__.setAllDCFG()
            __dt_cfg__ = __none_cfg__
            __dt__ = NothingDetecter()
            __dt_is_set__ = True
            add_info_log("---PYPPBOX : Set detector='" + str(detector) + "'")
    else:
        add_warning_log("---PYPPBOX : detector='" + str(detector) + "' is not recognized.")

def detectPeople(img, img_is_mat=False, visual=False, save=False, save_file=""): 
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
    
    Returns
    -------
    list[Person, ...]
        A  list of :class:`Person` object.
    Mat
        A cv :obj:`Mat` image.
    """ 
    global __unistrings__, __dt__, __dt_cfg__, __dt_is_set__
    people = []
    if __dt_is_set__: 
        if not isinstance(__dt__, NothingDetecter):
            if not img_is_mat: img = getCVMat(img)
            if (__dt_cfg__.dt_name.lower() == __unistrings__.yolo_cls or 
                __dt_cfg__.dt_name.lower() == __unistrings__.yolo_ult):
                people, img = __dt__.detectPeople(img, visual=visual)
            elif __dt_cfg__.dt_name.lower() == __unistrings__.gt:
                people, img = __dt__.getPeople(img, visual=visual)
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

__tk_is_set__ = False
__tk_cfg__ = []
__tk__ = []

def __loadDefaultTracker__():
    global __unistrings__, __cfg__, __cfg_is_set__, __tk__
    global __tk_cfg__, __tk_is_set__, __none_cfg__
    if __cfg_is_set__:
        if __cfg__.mcfg.tracker.lower() == __unistrings__.centroid:
            from pyppbox.modules.trackers.centroid import MyCentroid
            __tk_cfg__ = __cfg__.tcfg_centroid
            __tk__ = MyCentroid(__tk_cfg__)
            __tk_is_set__ = True
            __setGTDTOnly__()
        elif __cfg__.mcfg.tracker.lower() == __unistrings__.sort:
            from pyppbox.modules.trackers.sort import MySORT
            __tk_cfg__ = __cfg__.tcfg_sort
            __tk__ = MySORT(__tk_cfg__)
            __tk_is_set__ = True
            __setGTDTOnly__()
        elif __cfg__.mcfg.tracker.lower() == __unistrings__.deepsort:
            from pyppbox.modules.trackers.deepsort import MyDeepSORT
            __tk_cfg__ = __cfg__.tcfg_deepsort
            __tk__ = MyDeepSORT(__tk_cfg__)
            __tk_is_set__ = True
            __setGTDTOnly__()
        elif __cfg__.mcfg.tracker.lower() == __unistrings__.none:
            __tk_cfg__ = __none_cfg__
            __tk__ = NothingTracker()
            __tk_is_set__ = True
        else: 
            add_warning_log("---PYPPBOX : The input tracker is not recognized.")
            __tk_is_set__ = False

def __setCustomTracker__(tracker_dict):
    global __unistrings__, __tk__, __tk_cfg__, __tk_is_set__, __none_cfg__
    if tracker_dict:
        if tracker_dict['tk_name'].lower() == __unistrings__.centroid:
            from pyppbox.modules.trackers.centroid import MyCentroid
            __tk_cfg__ = TCFGCentroid()
            __tk_cfg__.set(tracker_dict)
            __tk__ = MyCentroid(__tk_cfg__)
            __tk_is_set__ = True
            __setGTDTOnly__()
            add_info_log("---PYPPBOX : Set tracker='" + __tk_cfg__.tk_name + "'")
        elif tracker_dict['tk_name'].lower() == __unistrings__.sort:
            from pyppbox.modules.trackers.sort import MySORT
            __tk_cfg__ = TCFGSORT()
            __tk_cfg__.set(tracker_dict)
            __tk__ = MySORT(__tk_cfg__)
            __tk_is_set__ = True
            __setGTDTOnly__()
            add_info_log("---PYPPBOX : Set tracker='" + __tk_cfg__.tk_name + "'")
        elif tracker_dict['tk_name'].lower() == __unistrings__.deepsort:
            from pyppbox.modules.trackers.deepsort import MyDeepSORT
            __tk_cfg__ = TCFGDeepSORT()
            __tk_cfg__.set(tracker_dict)
            __tk__ = MyDeepSORT(__tk_cfg__)
            __tk_is_set__ = True
            __setGTDTOnly__()
            add_info_log("---PYPPBOX : Set tracker='" + __tk_cfg__.tk_name + "'")
        elif tracker_dict['tk_name'].lower() == __unistrings__.none:
            __tk_cfg__ = __none_cfg__
            __tk__ = NothingTracker()
            __tk_is_set__ = True
            add_info_log("---PYPPBOX : Set tracker='" + __tk_cfg__.tk_name + "'")
        else:
            __tk_is_set__ = False
            add_warning_log("---PYPPBOX : tracker='" + tracker_dict['tk_name'] + 
                  "' is not recognized")

def setMainTracker(tracker=""):
    """Set the main tracker by a supported name, a raw YAML/JSON string, a ready YAML/JSON 
    dictionary, or a YAML/JSON file. Calling :func:`setConfigDir()` before :func:`setMainTracker()` 
    is optional. Different from the rest, setting the main tracker by its name results 
    in loading the configurations from the config directory set by the last :func:`setConfigDir()`. 
    If :func:`setConfigDir()` has not been called before, setting the main tracker by a supported 
    name results in referencing the internal config directory in order to load the 
    corresponding configurations. 

    Parameters
    ----------
    tracker : str or dict, default=""
        (1) Set :code:`tracker=""` to set the main tracker according to the main configurations or 
        main.yaml and load its configurations from the trackers.yaml. 
        (2) Set :code:`tracker="Centroid"`.etc, to set Centroid as the main tracker and load its 
        configurations from trackers.yaml.
        (3) Set a raw JSON string or a ready JSON dictionary is also possible; for example, 
        :code:`tracker="[{'tk_name': 'SORT', 'max_age': 1, 'min_hits': 3, 'iou_threshold': 0.3}]"`.
        (4) Set :code:`tracker="a_suported_tracker.yaml"` or :code:`tracker="a_suported_tracker.json"` 
        to set the main tracker and its configuration from a YAML file. 
    """
    global __unistrings__, __cfg__, __cfg_is_set__, __tk__
    global __tk_cfg__, __tk_is_set__, __none_cfg__
    __tk_is_set__ = False
    __tk__ = []
    if isinstance(tracker, dict):
        __setCustomTracker__(tracker)
    elif isinstance(tracker, str):
        if (isRawYAMLString(tracker) or "yaml" in tracker.lower() or 
            "json" in tracker.lower()):
            __setCustomTracker__(getCFGDict(tracker))
        elif tracker.lower() == "default":
            if not __cfg_is_set__: setConfigDir()
            __loadDefaultTracker__()
            add_info_log("---PYPPBOX : Use tracker according to the \"main.yaml\"")
        elif tracker.lower() == __unistrings__.centroid:
            from pyppbox.modules.trackers.centroid import MyCentroid
            if not __cfg_is_set__: setConfigDir()
            __cfg__.setAllTCFG()
            __tk_cfg__ = __cfg__.tcfg_centroid
            __tk__ = MyCentroid(__tk_cfg__)
            __tk_is_set__ = True
            __setGTDTOnly__()
            add_info_log("---PYPPBOX : Set tracker='" + str(tracker) + "'")
        elif tracker.lower() == __unistrings__.sort:
            from pyppbox.modules.trackers.sort import MySORT
            if not __cfg_is_set__: setConfigDir()
            __cfg__.setAllTCFG()
            __tk_cfg__ = __cfg__.tcfg_sort
            __tk__ = MySORT(__tk_cfg__)
            __tk_is_set__ = True
            __setGTDTOnly__()
            add_info_log("---PYPPBOX : Set tracker='" + str(tracker) + "'")
        elif tracker.lower() == __unistrings__.deepsort:
            from pyppbox.modules.trackers.deepsort import MyDeepSORT
            if not __cfg_is_set__: setConfigDir()
            __cfg__.setAllTCFG()
            __tk_cfg__ = __cfg__.tcfg_deepsort
            __tk__ = MyDeepSORT(__tk_cfg__)
            __tk_is_set__ = True
            __setGTDTOnly__()
            add_info_log("---PYPPBOX : Set tracker='" + str(tracker) + "'")
        elif tracker.lower() == __unistrings__.none:
            if not __cfg_is_set__: setConfigDir()
            __cfg__.setAllTCFG()
            __tk_cfg__ = __none_cfg__
            __tk__ = NothingTracker()
            __tk_is_set__ = True
            add_info_log("---PYPPBOX : Set tracker='" + str(tracker) + "'")
    else:
        add_warning_log("---PYPPBOX : tracker='" + str(tracker) + "' is not valid")

def trackPeople(img, people, img_is_mat=False):
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
    global __tk__, __tk_cfg__, __tk_is_set__
    res = []
    if __tk_is_set__: 
        if isinstance(people, list):
            if not img_is_mat: img = getCVMat(img)
            res = __tk__.update(people, img=img)
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

__ri_is_set__ = False
__ri_cfg__ = []
__ri__ = []
__deepidlistTMP__ = []
__faceidlistTMP__ = []

def __loadDefaultReIDer__(auto_load=True):
    global __unistrings__, __cfg__, __cfg_is_set__, __ri__, __ri_cfg__
    global __ri_is_set__, __none_cfg__, __dt_cfg__, __tk_cfg__
    if __cfg_is_set__:
        if __cfg__.mcfg.reider.lower() == __unistrings__.facenet:
            from pyppbox.modules.reiders.facenet import MyFaceNet
            __ri_cfg__ = __cfg__.rcfg_facenet
            __ri__ = MyFaceNet(__ri_cfg__, auto_load=auto_load)
            __ri_is_set__ = True
            __setGTDTOnly__()
        elif __cfg__.mcfg.reider.lower() == __unistrings__.torchreid:
            from pyppbox.modules.reiders.torchreid import MyTorchreid
            __ri_cfg__ = __cfg__.rcfg_torchreid
            __ri__ = MyTorchreid(__ri_cfg__, auto_load=auto_load)
            __ri_is_set__ = True
            __setGTDTOnly__()
        elif __cfg__.mcfg.reider.lower() == __unistrings__.none:
            if (__dt_cfg__.dt_name.lower() != __unistrings__.none and 
                __tk_cfg__.tk_name.lower() != __unistrings__.none):
                __ri__ = TKOReider(static=True)
            else:
                __ri__ = NothingReider()
            __ri_cfg__ = __none_cfg__
            __ri_is_set__ = True
            __revokeGTDTOnly__()
        else:
            add_warning_log("---PYPPBOX : The input reider is not recognized.")
            __ri_is_set__ = False

def __setCustomReIDer__(reider_dict, auto_load=True):
    global __unistrings__, __cfg__, __ri__, __ri_cfg__, __ri_is_set__
    global __none_cfg__, __dt_cfg__, __tk_cfg__
    if reider_dict:
        if reider_dict['ri_name'].lower() == __unistrings__.facenet:
            from pyppbox.modules.reiders.facenet import MyFaceNet
            __ri_cfg__ = RCFGFaceNet()
            __ri_cfg__.set(reider_dict)
            __ri__ = MyFaceNet(__ri_cfg__, auto_load=auto_load)
            __ri_is_set__ = True
            add_info_log("---PYPPBOX : Set reider='" + __ri_cfg__.ri_name + "'")
            __setGTDTOnly__()
        elif reider_dict['ri_name'].lower() == __unistrings__.torchreid:
            from pyppbox.modules.reiders.torchreid import MyTorchreid
            __ri_cfg__ = RCFGTorchreid()
            __ri_cfg__.set(reider_dict)
            __ri__ = MyTorchreid(__ri_cfg__, auto_load=auto_load)
            __ri_is_set__ = True
            add_info_log("---PYPPBOX : Set reider='" + __ri_cfg__.ri_name + "'")
            __setGTDTOnly__()
        elif reider_dict['ri_name'].lower() == __unistrings__.none:
            if (__dt_cfg__.dt_name.lower() != __unistrings__.none and 
                __tk_cfg__.tk_name.lower() != __unistrings__.none):
                __ri__ = TKOReider(static=True)
            else:
                __ri__ = NothingReider()
            __ri_cfg__ = __none_cfg__
            __ri_is_set__ = True
            add_info_log("---PYPPBOX : Set reider='" + __ri_cfg__.ri_name + "'")
            __revokeGTDTOnly__()
        else :
            __ri_is_set__ = False
            add_warning_log("---PYPPBOX : reider='" + reider_dict['ri_name'] + 
                  "' is not recognized")

def setMainReIDer(reider="", auto_load=True):
    """Set the main reider by a supported name, a raw YAML/JSON string, a ready YAML/JSON 
    dictionary, or a YAML/JSON file. Calling :func:`setConfigDir()` before :func:`setMainTracker()` 
    is optional. Different from the rest, setting the main reider by its name results in 
    loading the configurations from the config directory set by the last :func:`setConfigDir()`. 
    If :func:`setConfigDir()` has not been called before, setting the main reider by a supported 
    name results in referencing the internal config directory in order to load the 
    corresponding configurations. 

    Parameters
    ----------
    reider : str or dict, default=""
        (1) Set :code:`reider=""` to set the main reider according to the main configurations 
        or main.yaml and load its configurations from the reiders.yaml. 
        (2) Set :code:`reider="FaceNet"`.etc, to set FaceNet as a reider and load its 
        configurations from reiders.yaml.
        (3) Set a raw JSON string or a ready JSON dictionary is also possible; for example, 
        :code:`reider="[{'ri_name': 'Torchreid', 
        'classifier_pkl': 'data/modules/torchreid/classifier/gta5_osnet_ain_ms_d_c.pkl', 
        'train_data': 'data/datasets/GTA_V_DATASET/body_128x256', 'model_name': 'osnet_ain_x1_0', 
        'model_path': 'data/modules/torchreid/models/torchreid/osnet_ain_ms_d_c.pth.tar', 
        'min_confidence': 0.35}]"`.
        (4) Set :code:`reider="a_suported_reider.yaml"` or :code:`reider="a_suported_reider.json"` 
        to set a the main reider and its configurations from a YAML file. 
    auto_load : bool, default=True
        All supported reiders are Pytorch or Tensorflow based module, thus they need to 
        initial and load their models/weights. :obj:`auto_load` is used to decide whether to 
        load the reider automatically once the reider is set. Keep the :code:`auto_load=True` 
        if it is meant for using :func:`reidPeople()`; however, even if the :code:`auto_load=False`, 
        the :func:`reidPeople()` will initial and load the reider by itself, but it requires 
        some time to do so.
    """
    global __unistrings__, __cfg__, __cfg_is_set__, __ri__
    global __ri_cfg__, __ri_is_set__, __none_cfg__
    __ri_is_set__ = False
    __ri__ = []
    if isinstance(reider, dict):
        __setCustomReIDer__(reider, auto_load)
    elif isinstance(reider, str):
        if (isRawYAMLString(reider) or "yaml" in reider.lower() or 
            "json" in reider.lower()):
            __setCustomReIDer__(getCFGDict(reider), auto_load)
        elif reider.lower() == "default":
            if not __cfg_is_set__: setConfigDir()
            __loadDefaultReIDer__(auto_load=auto_load)
            add_info_log("---PYPPBOX : Use reider according to the \"main.yaml\"")
        elif reider.lower() == __unistrings__.facenet:
            from pyppbox.modules.reiders.facenet import MyFaceNet
            if not __cfg_is_set__: setConfigDir()
            __cfg__.setAllRCFG()
            __ri_cfg__ = __cfg__.rcfg_facenet
            __ri__ = MyFaceNet(__ri_cfg__, auto_load=auto_load)
            __ri_is_set__ = True
            add_info_log("---PYPPBOX : Set reider='" + str(reider) + "'")
            __setGTDTOnly__()
        elif reider.lower() == __unistrings__.torchreid:
            from pyppbox.modules.reiders.torchreid import MyTorchreid
            if not __cfg_is_set__: setConfigDir()
            __cfg__.setAllRCFG()
            __ri_cfg__ = __cfg__.rcfg_torchreid
            __ri__ = MyTorchreid(__ri_cfg__, auto_load=auto_load)
            __ri_is_set__ = True
            add_info_log("---PYPPBOX : Set reider='" + str(reider) + "'")
            __setGTDTOnly__()
        elif reider.lower() == __unistrings__.none:
            if not __cfg_is_set__: setConfigDir()
            __cfg__.setAllRCFG()
            if (__dt_cfg__.dt_name.lower() != __unistrings__.none and 
                __tk_cfg__.tk_name.lower() != __unistrings__.none):
                __ri__ = TKOReider(static=True)
            else:
                __ri__ = NothingReider()
            __ri_cfg__ = __none_cfg__
            __ri_is_set__ = True
            add_info_log("---PYPPBOX : Set reider='" + str(reider) + "'")
            __revokeGTDTOnly__()
    else:
        add_warning_log("---PYPPBOX : reider='" + str(reider) + "' is not valid")

def reidPeople(img, people, deduplicate=True, img_is_mat=False):
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
    global __ri__, __ri_cfg__, __ri_is_set__, __unistrings__
    res = []
    reid_count = [0, 0]
    if __ri_is_set__:
        # if not isinstance(__ri__, TKOReider) and not isinstance(__ri__, NothingReider):
        if __ri_cfg__.ri_name.lower() != __unistrings__.none:
            if not __ri__.auto_load:
                __ri__.load_classifier()
                __ri__.auto_load = True
        if isinstance(people, list):
            if len(people) > 0:
                if isinstance(people[0], Person):
                    if not img_is_mat: img = getCVMat(img)
                    res, reid_count[0] = __reidNormal__(img, people)
                    if deduplicate: res, reid_count[1] = __reidDupkiller__(img, res)
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

def __reidNormal__(img, people):
    global __ri_cfg__
    if __ri_cfg__.ri_name.lower() == __unistrings__.facenet:
        return __reidFaceNormal__(img, people)
    elif __ri_cfg__.ri_name.lower() == __unistrings__.torchreid:
        return __reidDeepNormal__(img, people)
    else:
        return __reidEmpty__(img, people)

def __reidDupkiller__(img, people):
    global __ri_cfg__
    if __ri_cfg__.ri_name.lower() == __unistrings__.facenet:
        return __reidDupFacekiller__(img, people)
    elif __ri_cfg__.ri_name.lower() == __unistrings__.torchreid:
        return __reidDupDeepkiller__(img, people)
    else:
        return __reidEmpty__(img, people)

def __reidEmpty__(_, people):
    global __unistrings__, __ri__, __ri_cfg__
    index = 0
    for person in people:
        deepid = str(person.deepid)
        if __unistrings__.err_did in deepid or __unistrings__.unk_did in deepid:
            people[index].deepid = __ri__.recognize(__unistrings__.unk_did)
        index += 1
    return people, 0

def __reidDeepNormal__(img, people):
    global __unistrings__, __ri__, __ri_cfg__, __deepidlistTMP__
    reid_count = 0
    index = 0
    __deepidlistTMP__ = []
    for person in people:
        deepid = person.deepid
        if __unistrings__.err_did in deepid or __unistrings__.unk_did in deepid:
            miniframe = img.copy()
            try:
                [x1, y1, x2, y2] = person.box_xyxy
                miniframe = miniframe[y1:y2, x1:x2]
                people[index].deepid, people[index].deepid_conf = __ri__.recognize(
                    cv2.resize(miniframe, __ri_cfg__.model_wh), 
                    is_bgr=True
                )
                reid_count += 1
            except Exception as e:
                add_warning_log("---PYPPBOX : __reidDeepNormal__() -> " + str(e))
        __deepidlistTMP__.append(deepid)
        index += 1
    return people, reid_count

def __reidDupDeepkiller__(img, people):
    global __ri__, __ri_cfg__, __deepidlistTMP__
    reid_count = 0
    if len(__deepidlistTMP__) != len(set(__deepidlistTMP__)):
        ddeepids = [k for k, v in Counter(__deepidlistTMP__).items() if v > 1]
        for ddeepid in ddeepids:
            index = 0
            for person in people:
                try:
                    if person.deepid == ddeepid:
                        [x1, y1, x2, y2] = person.box_xyxy
                        miniframe = img.copy()
                        miniframe = miniframe[y1:y2, x1:x2]
                        people[index].deepid, people[index].deepid_conf = __ri__.recognize(
                            cv2.resize(miniframe, __ri_cfg__.model_wh), 
                            is_bgr=True
                        )
                        reid_count += 1
                except Exception as e:
                    add_warning_log("---PYPPBOX : __reidDupDeepkiller__() -> " + str(e))
                index += 1
    return people, reid_count

def __reidFaceNormal__(img, people):
    global __unistrings__, __cfg__, __ri__, __faceidlistTMP__
    reid_count = 0
    index = 0
    __faceidlistTMP__ = []
    for person in people:
        faceid = person.faceid
        if __unistrings__.err_fid in faceid or __unistrings__.unk_fid in faceid:
            (x, y) = person.repspoint
            miniframe = img.copy()
            try:
                miniframe = miniframe[
                    y + int(__cfg__.rcfg_facenet.yl_h_calibration[0]):
                    y + int(__cfg__.rcfg_facenet.yl_h_calibration[1]), 
                    x + int(__cfg__.rcfg_facenet.yl_w_calibration[0]):
                    x + int(__cfg__.rcfg_facenet.yl_w_calibration[1])
                ]
                people[index].faceid, people[index].faceid_conf = __ri__.recognize(
                    miniframe, 
                    is_bgr=True
                )
                reid_count += 1
            except Exception as e:
                add_warning_log("---PYPPBOX : __reidFaceNormal__() -> " + str(e))
        __faceidlistTMP__.append(faceid)
        index += 1
    return people, reid_count

def __reidDupFacekiller__(img, people):
    global __ri__, __cfg__, __faceidlistTMP__
    reid_count = 0
    if len(__faceidlistTMP__) != len(set(__faceidlistTMP__)):
        dfaceids = [k for k, v in Counter(__faceidlistTMP__).items() if v > 1]
        for dfaceid in dfaceids:
            index = 0
            for person in people:
                try:
                    if person.faceid == dfaceid:
                        (x, y) = person.repspoint
                        miniframe = img.copy()
                        miniframe = miniframe[
                            y + int(__cfg__.rcfg_facenet.yl_h_calibration[0]):
                            y + int(__cfg__.rcfg_facenet.yl_h_calibration[1]), 
                            x + int(__cfg__.rcfg_facenet.yl_w_calibration[0]):
                            x + int(__cfg__.rcfg_facenet.yl_w_calibration[1])
                        ]
                        people[index].faceid, people[index].faceid_conf = __ri__.recognize(
                            miniframe, 
                            is_bgr=True
                        )
                        reid_count += 1
                except Exception as e:
                    add_warning_log("---PYPPBOX : __reidDupFacekiller__() -> " + str(e))
                index += 1
    return people, reid_count

def trainReIDClassifier(reider="Default", train_data="", classifier_pkl=""):
    """Train classifier of a reider by pointing to a data directory. Calling 
    :func:`setConfigDir()` or :func:`setMainReIDer()` in advance is not required.

    Parameters
    ----------
    reider : str or dict, default="Default" 
        A supported name, a raw YAML/JSON string, a ready YAML/JSON dictionary, or a 
        YAML/JSON file which is passed to :func:`setMainReIDer(reider=reider, auto_load=False)`.
    train_data : str, default=""
        A path of data to train, where consists of 2 or more sub-folders which classify 
        2 or more people. Set :code:`train_data=""` or keep default to use the configured 
        :obj:`train_data` according to the input :code:`reider`. All images in this the sub-folders 
        must be 128x256 for Torchreid and 182x182 for FaceNet.
    classifier_pkl : str, default=""
        A file path for the classifier PKL file. Set :code:`classifier_pkl=""` or keep default 
        to use the configured :obj:`classifier_pkl` in the input :obj:`reider`.
    """
    global __ri__, __ri_cfg__, __ri_is_set__
    setMainReIDer(reider=reider, auto_load=False)
    if __ri_is_set__:
        valid_train_data = True
        valid_pkl = True
        if train_data != "":
            if isExist(train_data): 
                __ri_cfg__.train_data = getAbsPathFDS(train_data)
            else: 
                valid_train_data = False
                add_error_log("---PYPPBOX : train_data='" + str(train_data) + 
                              "' does not exist.")
        if classifier_pkl != "":
            if not isExist(getAncestorDir(classifier_pkl)):
                valid_pkl = False
                add_error_log("---PYPPBOX : classifier_pkl='" + str(classifier_pkl) + 
                              "' is not valid.")
            if valid_pkl: __ri_cfg__.classifier_pkl = getAbsPathFDS(classifier_pkl)
        if valid_train_data and valid_pkl:
            from pyppbox.modules.reiders.facenet import MyFaceNet
            from pyppbox.modules.reiders.torchreid import MyTorchreid
            if isinstance(__ri__, MyFaceNet):
                __ri__ = MyFaceNet(__ri_cfg__, auto_load=False)
                add_info_log("------------- FaceNet --------------")
            elif isinstance(__ri__, MyTorchreid):
                __ri__ = MyTorchreid(__ri_cfg__, auto_load=False)
                add_info_log("------------ Torchreid -------------")
            add_info_log("---PYPPBOX : train_data='" + str(__ri_cfg__.train_data) + "'")
            add_info_log("---PYPPBOX : classifier_pkl='" + str(__ri_cfg__.classifier_pkl) + "'")
            __ri__.train_classifier()
