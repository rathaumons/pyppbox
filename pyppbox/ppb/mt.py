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


# Logging
from pyppbox.utils.logtools import add_info_log, add_warning_log, add_error_log

# Common
import cv2
from collections import Counter
from typing import Any, Dict, List, Union, Optional

# Configurations
from pyppbox.config.configtools import isConfigInput, getCFGDict
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


__none_cfg__ = NoneCFG("Fiat Moneey")

class MT(object):

    """An all-in-one class designed for easy detect, track, and reid people in a single threading 
    or multithreading application.

    Example : >>> import cv2
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
        self.__dt_cfg__ = __none_cfg__
        self.__dt__ = None
        # tracker
        self.__tk_is_set__ = False
        self.__tk_cfg__ = __none_cfg__
        self.__tk__ = None
        # reider
        self.__ri_is_set__ = False
        self.__ri_cfg__ = __none_cfg__
        self.__ri__ = None
        self.__deepidlistTMP__ = []
        self.__faceidlistTMP__ = []


    ###########################################
    # Configurator
    ###########################################

    def __setInternalCFGDir__(self, load_all):
        self.__cfg__.__init__()
        self.__cfg__.setMainModules()
        if load_all:
            add_info_log(f"---PYPPBOX : DT='{self.__cfg__.mcfg.detector}', "
                         f"TK='{self.__cfg__.mcfg.tracker}', "
                         f"RI='{self.__cfg__.mcfg.reider}'")
            self.__loadDefaultDetector__()
            self.__loadDefaultTracker__()
            self.__loadDefaultReIDer__()

    def setConfigDir(self, config_dir: Optional[str] = None, load_all: bool = False):
        """Select a directory containing the four configuration files.

        Parameters
        ----------
        config_dir : str or None
            Defaults to ``None``. Use the internal configuration directory when
            omitted. A custom directory must contain ``main.yaml``, ``detectors.yaml``,
            ``trackers.yaml``, and ``reiders.yaml``. A nonexistent directory triggers
            a warning and falls back to the internal directory.
        load_all : bool
            Defaults to ``False``. If True, construct the selected detector, tracker,
            and reider, including their models/classifiers. If False, load configuration
            values only; any existing active module instances remain in place.

        Raises
        ------
        ValueError
            If the directory argument has an unsupported type or a config cannot be read.

        Notes
        -----
        Paths within internal configs resolve relative to the package directory.
        Paths within custom configs resolve relative to the working directory.
        The directory loader uses the four YAML filenames above; individual module
        setters additionally accept JSON files.
        """
        self.__cfg_is_set__ = False
        if config_dir == None:
            add_info_log("---PYPPBOX : Now use the internal config directory")
            self.__cfg_is_set__ = True
            self.__setInternalCFGDir__(load_all=load_all)
        elif isinstance(config_dir, str):
            if isExist(config_dir):
                add_info_log(f"---PYPPBOX : Now use custom config dir, config_dir='{config_dir}'")
                self.__cfg__.setCustomCFG(cfg_dir=config_dir)
                self.__cfg__.setMainModules()
                self.__cfg_is_set__ = True
                if load_all:
                    add_info_log(f"---PYPPBOX : DT='{self.__cfg__.mcfg.detector}', "
                                 f"TK='{self.__cfg__.mcfg.tracker}', "
                                 f"RI='{self.__cfg__.mcfg.reider}'")
                    self.__loadDefaultDetector__()
                    self.__loadDefaultTracker__()
                    self.__loadDefaultReIDer__()
            else:
                add_warning_log(f"---PYPPBOX : config_dir='{config_dir}' does not exist")
                add_warning_log("---PYPPBOX : Switched to internal config directory")
                self.__cfg_is_set__ = True
                self.__setInternalCFGDir__(load_all=load_all)
        else:
            msg = f"PYPPBOX : setConfigDir() -> config_dir='{config_dir}' is not valid."
            add_error_log(msg)
            raise ValueError(msg)

    def setMainModules(self, main_yaml: Optional[Union[str, Dict[str, Any]]] = None, load_all: bool = True):
        """Select the main detector, tracker, and reider from one configuration mapping.

        Parameters
        ----------
        main_yaml : str or dict or list or None
            Defaults to ``None``. Read ``main.yaml`` from the selected config directory
            when omitted. Otherwise accept a YAML/JSON path, inline mapping, ready
            dictionary, or a legacy list containing one mapping. The keys are
            ``detector``, ``tracker``, and ``reider``.
        load_all : bool
            Defaults to ``True``. Construct all selected modules when True. When False,
            update configuration objects only; existing active instances are unchanged.

        Notes
        -----
        The internal config directory is selected if none has been configured.
        Constructing models can load weights and classifiers and reset tracking state.
        """
        if not self.__cfg_is_set__: self.setConfigDir()
        self.__cfg__.setMainModules(main_yaml=main_yaml)
        if load_all: 
            add_info_log(f"---PYPPBOX : DT='{self.__cfg__.mcfg.detector}', "
                         f"TK='{self.__cfg__.mcfg.tracker}', "
                         f"RI='{self.__cfg__.mcfg.reider}'")
            self.__loadDefaultDetector__()
            self.__loadDefaultTracker__()
            self.__loadDefaultReIDer__()

    def getConfig(self):
        """Return the live configurator used by this pipeline.

        Returns
        -------
        pyppbox.config.myconfig.MyConfigurator
            The existing mutable object, not a copy. Changing it does not automatically
            rebuild active modules or save its configuration files.
        """
        return self.__cfg__

    def getMainConfig(self, current=True):
        """Return the names of the active or configured pipeline stages.

        Parameters
        ----------
        current : bool
            Defaults to ``True``. Use the active module configurations when True.
            When False, use the main configuration stored in the configurator; load
            it with ``setConfigDir()`` or ``setMainModules()`` first.

        Returns
        -------
        dict
            A fresh mapping with ``detector``, ``tracker``, and ``reider`` keys.
            Uninitialized active stages are named ``"None"``. Configured names can
            differ from active names after selection with ``load_all=False`` or
            after setting an individual stage.
        """
        if current:
            return {
                "detector": self.__dt_cfg__.dt_name,
                "tracker": self.__tk_cfg__.tk_name,
                "reider": self.__ri_cfg__.ri_name,
            }
        else:
            return self.__cfg__.getMCFG()


    ###########################################
    # Detector
    ###########################################

    def __syncGTMode__(self):
        if self.__dt_is_set__ and self.__dt_cfg__.dt_name.lower() == self.__unistrings__.gt:
            detect_only = (
                (self.__tk_is_set__ and self.__tk_cfg__.tk_name.lower() != self.__unistrings__.none)
                or (self.__ri_is_set__ and self.__ri_cfg__.ri_name.lower() != self.__unistrings__.none)
            )
            self.__dt__.setDetectOnly(self.__unistrings__.unk_fid, self.__unistrings__.unk_did,
                                     detect_only=detect_only)

    def forceFullGTMode(self):
        """Normally when :code:`DT='GT'`, pyppbox can automatically decide the GT mode based on the 
        name of the tracker and/or the name of reider; however, if the decision is not satisfied 
        calling this :func:`forceFullGTMode()` sets :code:`detect_only=False` until the next
        detector, tracker, or reider selection. Normal frame processing preserves the override.
        """
        success = False
        if self.__dt_is_set__:
            if self.__dt_cfg__.dt_name.lower() == self.__unistrings__.gt:
                self.__dt__.setDetectOnly(self.__unistrings__.unk_fid, self.__unistrings__.unk_did, 
                                          detect_only=False)
                success = True
                msg = ("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                       "---PYPPBOX : Override detect_only=False for GT, FULL GT mode."
                       "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                add_info_log(msg, add_new_line=True)

        if not success:
            msg = ("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                   "---PYPPBOX : DT!='GT' -> Failed to override detect_only=False"
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
        else:
            add_warning_log("---PYPPBOX : The config is not set.")
        self.__syncGTMode__()


    def __setCustomDetector__(self, detector_dict):
        if detector_dict:
            if detector_dict['dt_name'].lower() == self.__unistrings__.yolo_cls:
                from pyppbox.modules.detectors.yolocls import MyYOLOCLS
                self.__dt_cfg__ = DCFGYOLOCLS()
                self.__dt_cfg__.set(detector_dict)
                self.__dt__ = MyYOLOCLS(self.__dt_cfg__)
                self.__dt_is_set__ = True
                add_info_log(f"---PYPPBOX : Set detector='{self.__dt_cfg__.dt_name}'")
            elif detector_dict['dt_name'].lower() == self.__unistrings__.yolo_ult:
                from pyppbox.modules.detectors.yoloult import MyYOLOULT
                self.__dt_cfg__ = DCFGYOLOULT()
                self.__dt_cfg__.set(detector_dict)
                self.__dt__ = MyYOLOULT(self.__dt_cfg__)
                self.__dt_is_set__ = True
                add_info_log(f"---PYPPBOX : Set detector='{self.__dt_cfg__.dt_name}'")
            elif detector_dict['dt_name'].lower() == self.__unistrings__.gt:
                self.__dt_cfg__ = DCFGGT()
                self.__dt_cfg__.set(detector_dict)
                self.__dt__ = GTInterpreter()
                self.__dt__.setGT(self.__dt_cfg__.gt_file)
                self.__dt_is_set__ = True
                add_info_log(f"---PYPPBOX : Set detector='{self.__dt_cfg__.dt_name}'")
            elif detector_dict['dt_name'].lower() == self.__unistrings__.none:
                self.__dt_cfg__ = __none_cfg__
                self.__dt__ = NothingDetecter()
                self.__dt_is_set__ = True
                add_info_log(f"---PYPPBOX : Set detector='{self.__dt_cfg__.dt_name}'")
            else: 
                self.__dt_is_set__ = False
                add_warning_log(f"---PYPPBOX : detector='{detector_dict['dt_name']}' is not recognized.")

    def setMainDetector(self, detector: Union[str, Dict[str, Any]] = ""):
        """Select and initialize the main detector.

        Parameters
        ----------
        detector : str or dict or list
            Defaults to ``""``. An empty string uses the selected main configuration.
            A supported name (YOLO_Classic, YOLO_Ultralytics, GT, or "None") loads settings from ``detectors.yaml`` in the
            selected directory, defaulting to the internal directory. Alternatively,
            provide a complete dictionary, inline YAML/JSON mapping, YAML/JSON file path,
            or legacy list containing one mapping. Custom paths within a mapping resolve
            relative to the working directory, not the configuration file's directory.

        Notes
        -----
        This replaces the existing stage, so retained model/tracking state is discarded.
        The name ``"None"`` is a string. Selection also recomputes GT detection-only
        mode, replacing any earlier ``forceFullGTMode()`` override. Config files are not saved.
        """
        self.__dt_is_set__ = False
        self.__dt__ = None
        if isConfigInput(detector):
            self.__setCustomDetector__(getCFGDict(detector))
        elif isinstance(detector, str):
            if detector == "":
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__loadDefaultDetector__()
                add_info_log('---PYPPBOX : Use detector according to the "main.yaml"')
            elif detector.lower() == self.__unistrings__.yolo_cls:
                from pyppbox.modules.detectors.yolocls import MyYOLOCLS
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllDCFG()
                self.__dt_cfg__ = self.__cfg__.dcfg_yolocs
                self.__dt__ = MyYOLOCLS(self.__dt_cfg__)
                self.__dt_is_set__ = True
                add_info_log(f"---PYPPBOX : Set detector='{detector}'")
            elif detector.lower() == self.__unistrings__.yolo_ult:
                from pyppbox.modules.detectors.yoloult import MyYOLOULT
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllDCFG()
                self.__dt_cfg__ = self.__cfg__.dcfg_yolout
                self.__dt__ = MyYOLOULT(self.__dt_cfg__)
                self.__dt_is_set__ = True
                add_info_log(f"---PYPPBOX : Set detector='{detector}'")
            elif detector.lower() == self.__unistrings__.gt:
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllDCFG()
                self.__dt_cfg__ = self.__cfg__.dcfg_gt
                self.__dt__ = GTInterpreter()
                self.__dt__.setGT(self.__dt_cfg__.gt_file)
                self.__dt_is_set__ = True
                add_info_log(f"---PYPPBOX : Set detector='{detector}'")
            elif detector.lower() == self.__unistrings__.none:
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllDCFG()
                self.__dt_cfg__ = __none_cfg__
                self.__dt__ = NothingDetecter()
                self.__dt_is_set__ = True
                add_info_log(f"---PYPPBOX : Set detector='{detector}'")
        else:
            add_warning_log(f"---PYPPBOX : detector='{detector}' is not recognized.")
        self.__syncGTMode__()


    def detectPeople(self, 
                     img, 
                     img_is_mat=False, 
                     visual=False, 
                     save=False, 
                     save_file="", 
                     min_width_filter=15,
                     alt_repspoint=False, 
                     alt_repspoint_top=True): 
        """Detect people with the active detector, optionally drawing and saving the image.

        Parameters
        ----------
        img : ``str or numpy.ndarray``
            Image filename or nonempty BGR image array.
        img_is_mat : bool
            Defaults to ``False``. Retained for API compatibility. The detector path
            currently validates and accepts both filenames and arrays regardless of this flag.
        visual : bool
            Defaults to ``False``. Draw detections on the returned image. An input
            array is modified in place when drawing is enabled.
        save : bool
            Defaults to ``False``. Save the returned image using OpenCV. Its format
            is selected by the extension of ``save_file``.
        save_file : str
            Defaults to ``""``. Output filename when saving; the parent directory
            must already exist. Saving can overwrite an existing image.
        min_width_filter : int
            Defaults to ``15``. Minimum detection-box width in pixels for YOLO detectors.
        alt_repspoint : bool
            Defaults to ``False``. For YOLO person detections, choose the top/bottom
            box midpoint instead of the configured calibration or pose-based point.
        alt_repspoint_top : bool
            Defaults to ``True``. Choose the top midpoint when the alternative is enabled;
            False chooses the bottom midpoint.

        Returns
        -------
        list[pyppbox.utils.persontools.Person]
            Detected people. GT detection reads the current GT frame and advances its cursor.
        ``numpy.ndarray or str``
            Loaded image, with drawings if requested. If no detector is active or
            the detector is ``"None"``, return an empty people list and the original
            input unchanged; loading and saving are skipped.

        Raises
        ------
        ValueError
            If an active detector receives an invalid image or the requested save fails.

        Notes
        -----
        Initialize a detector first with ``setMainDetector()``, ``setMainModules()``,
        or ``setConfigDir(load_all=True)``. The default ``setConfigDir()`` alone only
        loads configuration. Detection filters and alternative points do not alter GT rows.
        """
        people = []
        if self.__dt_is_set__: 
            if not isinstance(self.__dt__, NothingDetecter):
                img = getCVMat(img)
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
                        try:
                            if not cv2.imwrite(filename=filename, img=img):
                                raise ValueError(f"PYPPBOX : detectPeople() -> Cannot save image: '{filename}'")
                        except cv2.error as e:
                            raise ValueError(f"PYPPBOX : detectPeople() -> Cannot save image: '{filename}': {e}") from e
                    else:
                        msg = (f"PYPPBOX : detectPeople() -> save_file='{save_file}' is not valid.")
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
            elif self.__cfg__.mcfg.tracker.lower() == self.__unistrings__.sort:
                from pyppbox.modules.trackers.sort import MySORT
                self.__tk_cfg__ = self.__cfg__.tcfg_sort
                self.__tk__ = MySORT(self.__tk_cfg__)
                self.__tk_is_set__ = True
            elif self.__cfg__.mcfg.tracker.lower() == self.__unistrings__.deepsort:
                from pyppbox.modules.trackers.deepsort import MyDeepSORT
                self.__tk_cfg__ = self.__cfg__.tcfg_deepsort
                self.__tk__ = MyDeepSORT(self.__tk_cfg__)
                self.__tk_is_set__ = True
            elif self.__cfg__.mcfg.tracker.lower() == self.__unistrings__.none:
                self.__tk_cfg__ = __none_cfg__
                self.__tk__ = NothingTracker()
                self.__tk_is_set__ = True
            else: 
                add_warning_log("---PYPPBOX : The input tracker is not recognized.")
                self.__tk_is_set__ = False
        else:
            add_warning_log("---PYPPBOX : The config is not set.")
        self.__syncGTMode__()


    def __setCustomTracker__(self, tracker_dict):
        if tracker_dict:
            if tracker_dict['tk_name'].lower() == self.__unistrings__.centroid:
                from pyppbox.modules.trackers.centroid import MyCentroid
                self.__tk_cfg__ = TCFGCentroid()
                self.__tk_cfg__.set(tracker_dict)
                self.__tk__ = MyCentroid(self.__tk_cfg__)
                self.__tk_is_set__ = True
                add_info_log(f"---PYPPBOX : Set tracker='{self.__tk_cfg__.tk_name}'")
            elif tracker_dict['tk_name'].lower() == self.__unistrings__.sort:
                from pyppbox.modules.trackers.sort import MySORT
                self.__tk_cfg__ = TCFGSORT()
                self.__tk_cfg__.set(tracker_dict)
                self.__tk__ = MySORT(self.__tk_cfg__)
                self.__tk_is_set__ = True
                add_info_log(f"---PYPPBOX : Set tracker='{self.__tk_cfg__.tk_name}'")
            elif tracker_dict['tk_name'].lower() == self.__unistrings__.deepsort:
                from pyppbox.modules.trackers.deepsort import MyDeepSORT
                self.__tk_cfg__ = TCFGDeepSORT()
                self.__tk_cfg__.set(tracker_dict)
                self.__tk__ = MyDeepSORT(self.__tk_cfg__)
                self.__tk_is_set__ = True
                add_info_log(f"---PYPPBOX : Set tracker='{self.__tk_cfg__.tk_name}'")
            elif tracker_dict['tk_name'].lower() == self.__unistrings__.none:
                self.__tk_cfg__ = __none_cfg__
                self.__tk__ = NothingTracker()
                self.__tk_is_set__ = True
                add_info_log(f"---PYPPBOX : Set tracker='{self.__tk_cfg__.tk_name}'")
            else:
                self.__tk_is_set__ = False
                add_warning_log(f"---PYPPBOX : tracker='{tracker_dict['tk_name']}' is not recognized.")

    def setMainTracker(self, tracker: Union[str, Dict[str, Any]] = ""):
        """Select and initialize the main tracker.

        Parameters
        ----------
        tracker : str or dict or list
            Defaults to ``""``. An empty string uses the selected main configuration.
            A supported name (Centroid, SORT, DeepSORT, or "None") loads settings from ``trackers.yaml`` in the
            selected directory, defaulting to the internal directory. Alternatively,
            provide a complete dictionary, inline YAML/JSON mapping, YAML/JSON file path,
            or legacy list containing one mapping. Custom paths within a mapping resolve
            relative to the working directory, not the configuration file's directory.

        Notes
        -----
        This replaces the existing stage, so retained model/tracking state is discarded.
        The name ``"None"`` is a string. Selection also recomputes GT detection-only
        mode, replacing any earlier ``forceFullGTMode()`` override. Config files are not saved.
        """
        self.__tk_is_set__ = False
        self.__tk__ = None
        if isConfigInput(tracker):
            self.__setCustomTracker__(getCFGDict(tracker))
        elif isinstance(tracker, str):
            if tracker == "":
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__loadDefaultTracker__()
                add_info_log('---PYPPBOX : Use tracker according to the "main.yaml"')
            elif tracker.lower() == self.__unistrings__.centroid:
                from pyppbox.modules.trackers.centroid import MyCentroid
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllTCFG()
                self.__tk_cfg__ = self.__cfg__.tcfg_centroid
                self.__tk__ = MyCentroid(self.__tk_cfg__)
                self.__tk_is_set__ = True
                add_info_log(f"---PYPPBOX : Set tracker='{tracker}'")
            elif tracker.lower() == self.__unistrings__.sort:
                from pyppbox.modules.trackers.sort import MySORT
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllTCFG()
                self.__tk_cfg__ = self.__cfg__.tcfg_sort
                self.__tk__ = MySORT(self.__tk_cfg__)
                self.__tk_is_set__ = True
                add_info_log(f"---PYPPBOX : Set tracker='{tracker}'")
            elif tracker.lower() == self.__unistrings__.deepsort:
                from pyppbox.modules.trackers.deepsort import MyDeepSORT
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllTCFG()
                self.__tk_cfg__ = self.__cfg__.tcfg_deepsort
                self.__tk__ = MyDeepSORT(self.__tk_cfg__)
                self.__tk_is_set__ = True
                add_info_log(f"---PYPPBOX : Set tracker='{tracker}'")
            elif tracker.lower() == self.__unistrings__.none:
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllTCFG()
                self.__tk_cfg__ = __none_cfg__
                self.__tk__ = NothingTracker()
                self.__tk_is_set__ = True
                add_info_log(f"---PYPPBOX : Set tracker='{tracker}'")
        else:
            add_warning_log(f"---PYPPBOX : tracker='{tracker}' is not recognized.")
        self.__syncGTMode__()


    def trackPeople(self, img, people, img_is_mat=False):
        """Update the active tracker once for a frame, including frames without detections.

        Parameters
        ----------
        img : ``str or numpy.ndarray``
            Frame filename or BGR image array. DeepSORT uses pixels for appearance features.
        people : list[pyppbox.utils.persontools.Person]
            Detections in this frame. Pass an empty list for a missed-detection frame
            so SORT/DeepSORT tracks age and can expire.
        img_is_mat : bool
            Defaults to ``False``. If True, pass the array to the tracker without
            loading or validating it with the image helper.

        Returns
        -------
        list[pyppbox.utils.persontools.Person]
            Current detections with updated tracking and identity metadata. Trackers
            mutate person objects and may return fewer people than were supplied.
            An unset tracker returns an empty list; tracker ``"None"`` passes detections through.

        Raises
        ------
        ValueError
            If an active tracker receives a non-list input or image conversion fails.

        Notes
        -----
        Initialize the tracker first, for example with ``setMainTracker()`` or
        ``setConfigDir(load_all=True)``. Keep one pipeline per independent stream.
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
            elif self.__cfg__.mcfg.reider.lower() == self.__unistrings__.torchreid:
                from pyppbox.modules.reiders.torchreid import MyTorchreid
                self.__ri_cfg__ = self.__cfg__.rcfg_torchreid
                self.__ri__ = MyTorchreid(self.__ri_cfg__, auto_load=auto_load)
                self.__ri_is_set__ = True
            elif self.__cfg__.mcfg.reider.lower() == self.__unistrings__.none:
                if (self.__dt_cfg__.dt_name.lower() != self.__unistrings__.none and 
                    self.__tk_cfg__.tk_name.lower() != self.__unistrings__.none):
                    self.__ri__ = TKOReider(static=True)
                else:
                    self.__ri__ = NothingReider()
                self.__ri_cfg__ = __none_cfg__
                self.__ri_is_set__ = True
            else:
                add_warning_log("---PYPPBOX : The input reider is not recognized.")
                self.__ri_is_set__ = False
        else:
            add_warning_log("---PYPPBOX : The config is not set.")
        self.__syncGTMode__()


    def __setCustomReIDer__(self, reider_dict, auto_load=True):
        if reider_dict:
            if reider_dict['ri_name'].lower() == self.__unistrings__.facenet:
                from pyppbox.modules.reiders.facenet import MyFaceNet
                self.__ri_cfg__ = RCFGFaceNet()
                self.__ri_cfg__.set(reider_dict)
                self.__ri__ = MyFaceNet(self.__ri_cfg__, auto_load=auto_load)
                self.__ri_is_set__ = True
                add_info_log(f"---PYPPBOX : Set reider='{self.__ri_cfg__.ri_name}'")
            elif reider_dict['ri_name'].lower() == self.__unistrings__.torchreid:
                from pyppbox.modules.reiders.torchreid import MyTorchreid
                self.__ri_cfg__ = RCFGTorchreid()
                self.__ri_cfg__.set(reider_dict)
                self.__ri__ = MyTorchreid(self.__ri_cfg__, auto_load=auto_load)
                self.__ri_is_set__ = True
                add_info_log(f"---PYPPBOX : Set reider='{self.__ri_cfg__.ri_name}'")
            elif reider_dict['ri_name'].lower() == self.__unistrings__.none:
                if (self.__dt_cfg__.dt_name.lower() != self.__unistrings__.none and 
                    self.__tk_cfg__.tk_name.lower() != self.__unistrings__.none):
                    self.__ri__ = TKOReider(static=True)
                else:
                    self.__ri__ = NothingReider()
                self.__ri_cfg__ = __none_cfg__
                self.__ri_is_set__ = True
                add_info_log(f"---PYPPBOX : Set reider='{self.__ri_cfg__.ri_name}'")
            else:
                self.__ri_is_set__ = False
                add_warning_log(f"---PYPPBOX : reider='{reider_dict['ri_name']}' is not recognized.")

    def setMainReIDer(self, reider: Union[str, Dict[str, Any]] = "", auto_load: bool = True):
        """Select and initialize the main reider.

        Parameters
        ----------
        reider : str or dict or list
            Defaults to ``""``. An empty string uses the selected main configuration.
            A supported name (FaceNet, Torchreid, or "None") loads settings from ``reiders.yaml`` in the
            selected directory, defaulting to the internal directory. Alternatively,
            provide a complete dictionary, inline YAML/JSON mapping, YAML/JSON file path,
            or legacy list containing one mapping. Custom paths within a mapping resolve
            relative to the working directory, not the configuration file's directory.
        auto_load : bool
            Defaults to ``True``. Load the identity classifier immediately when True.
            False defers classifier loading until ``reidPeople()`` first uses the stage.
            Torchreid still constructs its feature extractor and loads its weights during
            initialization. Direct reider constructors default this option to False.

        Notes
        -----
        This replaces the existing stage, so retained model/tracking state is discarded.
        The name ``"None"`` is a string. Selection also recomputes GT detection-only
        mode, replacing any earlier ``forceFullGTMode()`` override. Config files are not saved.
        With detector and tracker enabled, reider ``"None"`` uses static
        fallback identities. In other combinations it passes existing identities through.
        """
        self.__ri_is_set__ = False
        self.__ri__ = None
        if isConfigInput(reider):
            self.__setCustomReIDer__(getCFGDict(reider), auto_load)
        elif isinstance(reider, str):
            if reider == "":
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__loadDefaultReIDer__(auto_load=auto_load)
                add_info_log('---PYPPBOX : Use reider according to the "main.yaml"')
            elif reider.lower() == self.__unistrings__.facenet:
                from pyppbox.modules.reiders.facenet import MyFaceNet
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllRCFG()
                self.__ri_cfg__ = self.__cfg__.rcfg_facenet
                self.__ri__ = MyFaceNet(self.__ri_cfg__, auto_load=auto_load)
                self.__ri_is_set__ = True
                add_info_log(f"---PYPPBOX : Set reider='{reider}'")
            elif reider.lower() == self.__unistrings__.torchreid:
                from pyppbox.modules.reiders.torchreid import MyTorchreid
                if not self.__cfg_is_set__: self.setConfigDir()
                self.__cfg__.setAllRCFG()
                self.__ri_cfg__ = self.__cfg__.rcfg_torchreid
                self.__ri__ = MyTorchreid(self.__ri_cfg__, auto_load=auto_load)
                self.__ri_is_set__ = True
                add_info_log(f"---PYPPBOX : Set reider='{reider}'")
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
                add_info_log(f"---PYPPBOX : Set reider='{reider}'")
        else:
            add_warning_log(f"---PYPPBOX : reider='{reider}' is not recognized.")
        self.__syncGTMode__()


    def reidPeople(self, img, people, deduplicate=True, img_is_mat=False):
        """Update unknown/error identities using the active reider.

        Parameters
        ----------
        img : ``str or numpy.ndarray``
            Frame filename or BGR image array used for person/face crops.
        people : list[pyppbox.utils.persontools.Person]
            People whose identity fields are updated in place. Existing recognized
            identities are retained unless duplicate re-inference is requested.
        deduplicate : bool
            Defaults to ``True``. Re-run recognition for repeated identities. This
            does not remove people or guarantee that the final identities are unique.
        img_is_mat : bool
            Defaults to ``False``. If True, use the array without image-helper conversion.

        Returns
        -------
        list[pyppbox.utils.persontools.Person]
            People with updated identity and confidence fields. Empty when the input
            list is empty or no reider has been initialized.
        tuple[int, int]
            Counts of completed initial and duplicate recognition calls, including
            unknown/error-ID results. Calls raising exceptions are not counted.
            Both counts are zero for a disabled reider.

        Raises
        ------
        ValueError
            If an active reider receives a non-list, an unsupported first element,
            or an image that cannot be converted.

        Notes
        -----
        Initialize the stage first with ``setMainReIDer()`` or an all-module loader.
        Deferred classifier loading happens before processing the people list and can
        therefore occur on an empty frame. Classifier/model loading errors propagate.
        Per-person crop or recognition exceptions are logged and the person is retained.
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
                    add_warning_log(f"---PYPPBOX : __reidDeepNormal__() -> {e}")
            self.__deepidlistTMP__.append(people[index].deepid)
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
                        add_warning_log(f"---PYPPBOX : __reidDupDeepkiller__() -> {e}")
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
                        y + int(self.__ri_cfg__.yl_h_calibration[0]):
                        y + int(self.__ri_cfg__.yl_h_calibration[1]),
                        x + int(self.__ri_cfg__.yl_w_calibration[0]):
                        x + int(self.__ri_cfg__.yl_w_calibration[1])
                    ]
                    people[index].faceid, people[index].faceid_conf = self.__ri__.recognize(
                        miniframe, 
                        is_bgr=True
                    )
                    reid_count += 1
                except Exception as e:
                    add_warning_log(f"---PYPPBOX : __reidFaceNormal__() -> {e}")
            self.__faceidlistTMP__.append(people[index].faceid)
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
                                y + int(self.__ri_cfg__.yl_h_calibration[0]):
                                y + int(self.__ri_cfg__.yl_h_calibration[1]),
                                x + int(self.__ri_cfg__.yl_w_calibration[0]):
                                x + int(self.__ri_cfg__.yl_w_calibration[1])
                            ]
                            people[index].faceid, people[index].faceid_conf = self.__ri__.recognize(
                                miniframe, 
                                is_bgr=True
                            )
                            reid_count += 1
                    except Exception as e:
                        add_warning_log(f"---PYPPBOX : __reidDupFacekiller__() -> {e}")
                    index += 1
        return people, reid_count

    def trainReIDClassifier(
        self, 
        reider: Union[str, Dict[str, Any]] = "", 
        train_data: str = "", 
        classifier_pkl: str = ""
    ):
        """Train and save an identity SVM using a supported reider's pretrained embeddings.

        Parameters
        ----------
        reider : str or dict or list
            Defaults to ``""``. Select the configured reider, a supported name, or a
            complete YAML/JSON configuration as for ``setMainReIDer()``. The classifier
            is not loaded before training, but required feature-extraction weights are.
        train_data : str
            Defaults to ``""``. Use the configured training directory when empty.
            Otherwise specify an existing directory with one image-only subdirectory
            per identity and at least two identities. The bundled examples use 128x256
            body images and 182x182 face images (width x height). Torchreid resizes through
            its extractor; FaceNet training uses 160x160 crops of prepared face images.
        classifier_pkl : str
            Defaults to ``""``. Use the configured output path when empty. An override's
            parent directory must exist. Training overwrites the pickle and companion
            ``.txt`` class-name file.

        Notes
        -----
        This replaces the pipeline's active reider and returns None. Invalid explicit
        input/output directories are logged and training is skipped. Backend training
        errors propagate. Config files are not saved. Load the written classifier before
        subsequent recognition; the pipeline does this on its next ``reidPeople()`` call.
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
                    add_error_log(f"---PYPPBOX : train_data='{train_data}' does not exist.")
            if classifier_pkl != "":
                if not isExist(getAncestorDir(classifier_pkl)):
                    valid_pkl = False
                    add_error_log(f"---PYPPBOX : classifier_pkl='{classifier_pkl}' is not valid.")
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
