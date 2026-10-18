# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


# Logging
from pyppbox.utils.logtools import add_info_log, add_warning_log, add_error_log

# Common
import cv2
from collections import Counter
from copy import copy
from operator import index as integer_index
from typing import Any, Dict, List, Union, Optional

# Configurations
from pyppbox.config.configtools import isConfigInput, getCFGDict
from pyppbox.config.myconfig import (
    MyConfigurator, NoneCFG,
    DCFGYOLOCLS, DCFGYOLOULT, DCFGRFDETR, DCFGGT,
    TCFGCentroid, TCFGSORT, TCFGDeepSORT, 
    RCFGFaceNet, RCFGTorchreid, RCFGArcFace, RCFGAdaFace,
)

# Classes & tools
from pyppbox.utils.persontools import Person
from pyppbox.utils.gttools import GTInterpreter
from pyppbox.utils.evatools import NothingDetecter, NothingTracker, NothingReider, TKOReider
from pyppbox.utils.commontools import getAbsPathFDS, isExist, getCVMat, getAncestorDir
from pyppbox.utils.batching import _batch_size, _batches


__none_cfg__ = NoneCFG("Fiat Moneey")


def _crop_reid_image(img, box_xyxy):
    """Return a clipped ROI view; reject malformed or entirely invisible boxes."""
    try:
        x1, y1, x2, y2 = map(integer_index, box_xyxy)
    except (TypeError, ValueError) as error:
        raise ValueError('ReID crop expects four integer coordinates.') from error
    height, width = img.shape[:2]
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x1 >= x2 or y1 >= y2:
        raise ValueError('ReID crop has no visible pixels.')
    return img[y1:y2, x1:x2]


class MT(object):

    """An all-in-one class designed for easy detect, track, and reid people in a single threading 
    or multithreading application.

    Examples
    --------
    .. code-block:: python

        import cv2
        import threading
        from pyppbox.utils.visualizetools import visualizePeople
        from pyppbox.standalone import MT

        def ppb_task(input, main_configs, name="Task"):
            ppbmt = MT()  # Use MT for multithreading
            ppbmt.setMainModules(main_yaml=main_configs)
            cap = cv2.VideoCapture(input)
            while cap.isOpened():
                hasFrame, frame = cap.read()
                if hasFrame:
                    detected_people, _ = ppbmt.detectPeople(
                        frame, img_is_mat=True, visual=False
                    )
                    tracked_people = ppbmt.trackPeople(
                        frame, detected_people, img_is_mat=True
                    )
                    reidentified_people, reid_count = ppbmt.reidPeople(
                        frame,
                        tracked_people,
                        img_is_mat=True
                    )
                    visualized_mat = visualizePeople(
                        frame,
                        reidentified_people,
                        show_reid=reid_count
                    )
                    cv2.imshow("Multithreading (" + name + ")", visualized_mat)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            cap.release()

        if __name__ == '__main__':
            input_one = "data/gta.mp4"
            input_two = "data/gta.mp4"
            main_configs_one = {'detector': 'YOLO_Classic',
                                'tracker': 'SORT',
                                'reider': 'Torchreid'}
            main_configs_two = {'detector': 'YOLO_Classic',
                                'tracker': 'Centroid',
                                'reider': 'FaceNet'}
            thread_one = threading.Thread(
                target=ppb_task, args=(input_one, main_configs_one, "Task 1")
            )
            thread_two = threading.Thread(
                target=ppb_task, args=(input_two, main_configs_two, "Task 2")
            )
            thread_one.start()
            thread_two.start()
            thread_one.join()
            thread_two.join()

    """

    def __init__(self):
        self._closed = False
        self._retired = []
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

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError('This pipeline is closed; create a new MT instance.')

    def _configured_candidate(self):
        candidate = copy(self.__cfg__)
        if not self.__cfg_is_set__:
            candidate = MyConfigurator()
            candidate.setMainModules()
        return candidate, True

    def _select_stage(self, stage, value, candidate, configured=None):
        specs = {
            'dt': ('dt_name', 'detector', 'setAllDCFG', {
                'YOLO_Classic': (DCFGYOLOCLS, 'dcfg_yolocs'),
                'YOLO_Ultralytics': (DCFGYOLOULT, 'dcfg_yolout'),
                'RF_DETR': (DCFGRFDETR, 'dcfg_rfdetr'), 'GT': (DCFGGT, 'dcfg_gt')}),
            'tk': ('tk_name', 'tracker', 'setAllTCFG', {
                'Centroid': (TCFGCentroid, 'tcfg_centroid'), 'SORT': (TCFGSORT, 'tcfg_sort'),
                'DeepSORT': (TCFGDeepSORT, 'tcfg_deepsort')}),
            'ri': ('ri_name', 'reider', 'setAllRCFG', {
                'FaceNet': (RCFGFaceNet, 'rcfg_facenet'), 'Torchreid': (RCFGTorchreid, 'rcfg_torchreid'),
                'ArcFace': (RCFGArcFace, 'rcfg_arcface'), 'AdaFace': (RCFGAdaFace, 'rcfg_adaface')})}
        key, selection, loader, classes = specs[stage]
        configured = self.__cfg_is_set__ if configured is None else configured
        if isConfigInput(value):
            document = getCFGDict(value)
            name = self.__unistrings__.getUnifiedFormat(document.get(key, ''))
            if name == 'None':
                return __none_cfg__, configured
            if name not in classes:
                raise ValueError(f'{key}: unsupported module selection {name!r}.')
            config = classes[name][0]()
            config.set(document)
        elif isinstance(value, str):
            if not configured:
                prepared, configured = self._configured_candidate()
                candidate.__dict__.update(prepared.__dict__)
            name = self.__unistrings__.getUnifiedFormat(value) if value else getattr(candidate.mcfg, selection)
            if name != 'None' and name not in classes:
                raise ValueError(f'{key}: unsupported module selection {name!r}.')
            if value:
                getattr(candidate, loader)()
            config = __none_cfg__ if name == 'None' else getattr(candidate, classes[name][1])
        else:
            raise ValueError(f'{key}: unsupported module selection {value!r}.')
        return config, configured

    def _build_stage(self, stage, config, auto_load=True, detector=None, tracker=None):
        name = getattr(config, {'dt': 'dt_name', 'tk': 'tk_name', 'ri': 'ri_name'}[stage])
        if name == 'None':
            if stage == 'dt':
                return NothingDetecter()
            if stage == 'tk':
                return NothingTracker()
            detector = self.__dt_cfg__ if detector is None else detector
            tracker = self.__tk_cfg__ if tracker is None else tracker
            return TKOReider(static=True) if detector.dt_name != 'None' and tracker.tk_name != 'None' else NothingReider()
        config.validateRuntime(classifier=auto_load if stage == 'ri' else False)
        if name == 'GT':
            result = GTInterpreter()
            result.setGT(config.gt_file)
            return result
        if name == 'YOLO_Classic':
            from pyppbox.modules.detectors.yolocls import MyYOLOCLS as factory
        elif name == 'YOLO_Ultralytics':
            from pyppbox.modules.detectors.yoloult import MyYOLOULT as factory
        elif name == 'RF_DETR':
            from pyppbox.modules.detectors.rfdetr import MyRFDETR as factory
        elif name == 'Centroid':
            from pyppbox.modules.trackers.centroid import MyCentroid as factory
        elif name == 'SORT':
            from pyppbox.modules.trackers.sort import MySORT as factory
        elif name == 'DeepSORT':
            from pyppbox.modules.trackers.deepsort import MyDeepSORT as factory
        elif name == 'FaceNet':
            from pyppbox.modules.reiders.facenet import MyFaceNet as factory
        elif name == 'ArcFace':
            from pyppbox.modules.reiders.onnxface import MyArcFace as factory
        elif name == 'AdaFace':
            from pyppbox.modules.reiders.onnxface import MyAdaFace as factory
        elif name == 'Torchreid':
            from pyppbox.modules.reiders.torchreid import MyTorchreid as factory
        else:
            raise ValueError(f'Unsupported module: {name}')
        return factory(config, auto_load=auto_load) if stage == 'ri' else factory(config)

    def _retire(self, backend):
        if backend is None:
            return
        try:
            close = getattr(backend, 'close', None)
            if callable(close):
                close()
        except Exception as error:
            if not any(item is backend for item in self._retired):
                self._retired.append(backend)
            add_warning_log(f'---PYPPBOX : backend cleanup failed; close() will retry: {error}')

    def _publish(self, candidate, configured, replacements):
        previous = [getattr(self, f'__{stage}__') for stage in replacements]
        # Preserve the public configurator object's identity for existing callers.
        self.__cfg__.__dict__.clear()
        self.__cfg__.__dict__.update(candidate.__dict__)
        self.__cfg_is_set__ = configured
        for stage, (config, backend) in replacements.items():
            setattr(self, f'__{stage}_cfg__', config)
            setattr(self, f'__{stage}__', backend)
            setattr(self, f'__{stage}_is_set__', True)
        if replacements:
            self.__syncGTMode__()
        for backend in previous:
            self._retire(backend)

    def _replace_stage(self, stage, value, auto_load=True):
        self._ensure_open()
        candidate = copy(self.__cfg__)
        config, configured = self._select_stage(stage, value, candidate)
        backend = self._build_stage(stage, config, auto_load=auto_load)
        self._publish(candidate, configured, {stage: (config, backend)})
        key = {'dt': 'dt_name', 'tk': 'tk_name', 'ri': 'ri_name'}[stage]
        add_info_log(f"---PYPPBOX : Set {key}='{getattr(config, key)}'")

    def _configure(self, candidate, load_all):
        replacements = {}
        if load_all:
            try:
                for stage in ('dt', 'tk', 'ri'):
                    config, _ = self._select_stage(stage, '', candidate, configured=True)
                    backend = self._build_stage(stage, config,
                        detector=replacements.get('dt', (None,))[0],
                        tracker=replacements.get('tk', (None,))[0])
                    replacements[stage] = (config, backend)
            except BaseException:
                for _, backend in reversed(list(replacements.values())):
                    self._retire(backend)
                raise
        self._publish(candidate, True, replacements)
        if load_all:
            add_info_log(f"---PYPPBOX : DT='{candidate.mcfg.detector}', "
                         f"TK='{candidate.mcfg.tracker}', RI='{candidate.mcfg.reider}'")

    def close(self):
        """Release owned stages and mark this pipeline closed.

        Successful cleanup is idempotent. A failing close hook does not prevent
        other stages from closing; failed hooks are retained for the next close()
        retry and reported as RuntimeError. Create a new MT for further processing.
        PyTorch allocator caches are not cleared globally. Caller-owned videos,
        windows, Person objects, and recorders are not closed or modified.
        """
        self._closed = True
        resources = self._retired
        self._retired = []
        for stage in ('dt', 'tk', 'ri'):
            backend = getattr(self, f'__{stage}__')
            if backend is not None and not any(item is backend for item in resources):
                resources.append(backend)
            setattr(self, f'__{stage}__', None)
            setattr(self, f'__{stage}_cfg__', __none_cfg__)
            setattr(self, f'__{stage}_is_set__', False)
        self.__deepidlistTMP__ = []
        self.__faceidlistTMP__ = []
        for backend in resources:
            self._retire(backend)
        if self._retired:
            raise RuntimeError(f'{len(self._retired)} backend(s) could not close; call close() to retry.')

    def reset(self, frame_index=0):
        """Begin another stream while retaining loaded detector and ReID models.

        Parameters
        ----------
        frame_index : int
            Defaults to zero. Set the GT detector's next frame cursor. Clear tracker
            histories, identity metadata, fallback-ID position, and ReID scratch lists.
            SORT IDs need not restart at zero; other live streams keep their counters.

        Notes
        -----
        Existing Person objects, configuration, loaded models/classifiers, and an
        explicit full-GT override are retained. Call between streams, without concurrent
        processing on this MT. A closed MT cannot be reset.
        """
        self._ensure_open()
        if isinstance(frame_index, bool):
            raise ValueError('frame_index must be a nonnegative integer.')
        try:
            frame_index = integer_index(frame_index)
        except TypeError as error:
            raise ValueError('frame_index must be a nonnegative integer.') from error
        if frame_index < 0:
            raise ValueError('frame_index must be a nonnegative integer.')
        if self.__tk_is_set__:
            self.__tk__.reset()
        if self.__dt_is_set__ and self.__dt_cfg__.dt_name == 'GT':
            self.__dt__.current_frame = frame_index
        if isinstance(self.__ri__, TKOReider):
            self.__ri__.reset()
        self.__deepidlistTMP__ = []
        self.__faceidlistTMP__ = []

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
            add_warning_log('---PYPPBOX : cleanup failed while handling an exception; close() can retry.')
        return False


    ###########################################
    # Configurator
    ###########################################


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
            Config/model loading failures retain the current pipeline and configurator.

        Notes
        -----
        Paths within internal configs resolve relative to the package directory.
        Paths within custom configs resolve relative to the working directory.
        The directory loader uses the four YAML filenames above; individual module
        setters additionally accept JSON files.
        """
        self._ensure_open()
        if config_dir is not None and not isinstance(config_dir, str):
            raise ValueError(f"PYPPBOX: config_dir={config_dir!r} is not valid.")
        if config_dir is not None and not isExist(config_dir):
            add_warning_log(f"---PYPPBOX : config_dir='{config_dir}' does not exist; using internal configs.")
            config_dir = None
        candidate = MyConfigurator() if config_dir is None else MyConfigurator(config_dir)
        candidate.setMainModules()
        self._configure(candidate, load_all)
        add_info_log(f"---PYPPBOX : Now use config directory '{candidate.cfg_dir}'")

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
        Construct all candidates before publishing configuration or active stages. A failure
        preserves the previous pipeline and closes earlier candidates. Successful loading
        resets tracking state and closes old stages. Models temporarily coexist; close
        the old pipeline explicitly and create a new MT if releasing memory first is
        required. No config files are saved.
        """
        self._ensure_open()
        candidate, _ = self._configured_candidate()
        candidate.setMainModules(main_yaml=main_yaml)
        self._configure(candidate, load_all)

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
        self._ensure_open()
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


    def setMainDetector(self, detector: Union[str, Dict[str, Any]] = ""):
        """Select and initialize the main detector.

        Parameters
        ----------
        detector : str or dict or list
            Defaults to ``""``. An empty string uses the selected main configuration.
            A supported name (YOLO_Classic, YOLO_Ultralytics, RF_DETR, GT, or "None") loads settings from ``detectors.yaml`` in the
            selected directory, defaulting to the internal directory. Alternatively,
            provide a complete dictionary, inline YAML/JSON mapping, YAML/JSON file path,
            or legacy list containing one mapping. Custom paths within a mapping resolve
            relative to the working directory, not the configuration file's directory.

        Notes
        -----
        Construct the candidate before replacing the active stage. Loading failure
        preserves its config, state, and resources; successful replacement closes it.
        The name ``"None"`` is a string. Selection also recomputes GT detection-only
        mode, replacing any earlier ``forceFullGTMode()`` override. Config files are not saved.
        """
        self._replace_stage('dt', detector)


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
        self._ensure_open()
        people = []
        if self.__dt_is_set__: 
            if not isinstance(self.__dt__, NothingDetecter):
                img = getCVMat(img)
                if (self.__dt_cfg__.dt_name.lower() == self.__unistrings__.yolo_cls or 
                    self.__dt_cfg__.dt_name.lower() == self.__unistrings__.yolo_ult or
                    self.__dt_cfg__.dt_name.lower() == self.__unistrings__.rf_detr):
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
        Construct the candidate before replacing the active stage. Loading failure
        preserves its config, state, and resources; successful replacement closes it.
        The name ``"None"`` is a string. Selection also recomputes GT detection-only
        mode, replacing any earlier ``forceFullGTMode()`` override. Config files are not saved.
        """
        self._replace_stage('tk', tracker)


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
        self._ensure_open()
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


    def setMainReIDer(self, reider: Union[str, Dict[str, Any]] = "", auto_load: bool = True):
        """Select and initialize the main reider.

        Parameters
        ----------
        reider : str or dict or list
            Defaults to ``""``. An empty string uses the selected main configuration.
            A supported name (FaceNet, ArcFace, AdaFace, Torchreid, or "None") loads settings from ``reiders.yaml`` in the
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
        Construct the candidate before replacing the active stage. Loading failure
        preserves its config, state, and resources; successful replacement closes it.
        The name ``"None"`` is a string. Selection also recomputes GT detection-only
        mode, replacing any earlier ``forceFullGTMode()`` override. Config files are not saved.
        With detector and tracker enabled, reider ``"None"`` uses static
        fallback identities. In other combinations it passes existing identities through.
        """
        self._replace_stage('ri', reider, auto_load=auto_load)


    def reidPeople(self, img, people, deduplicate=True, img_is_mat=False, *, batch_size=1):
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
        batch_size : int
            Defaults to one, retaining scalar behavior. Values above one opt into
            bounded Torchreid inference batches. This call-time option does not
            change YAML files or FaceNet's scalar inference/training settings.

        Returns
        -------
        list[pyppbox.utils.persontools.Person]
            People with updated identity and confidence fields. Empty when the input
            list is empty or no reider has been initialized.
        tuple[int, int]
            Counts of completed initial and duplicate person recognitions, including
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
        Integer crop bounds are clipped to the image; malformed or entirely invisible
        crops are skipped without counting a recognition call. Valid crops keep the
        existing resizing/preprocessing. Recognition does not modify the input frame.
        Opt-in Torchreid batches retain group order and per-person counts. Failed
        batches retry scalar recognition per person. Aliased Person objects use the
        scalar path. Floating-point confidence can vary; close decisions are rechecked
        with scalar recognition. Batch calls/rechecks do not add to the counts.
        """
        self._ensure_open()
        size = _batch_size(batch_size)
        res = []
        reid_count = [0, 0]
        if self.__ri_is_set__:
            if self.__ri_cfg__.ri_name.lower() != self.__unistrings__.none:
                if not self.__ri__.auto_load:
                    self.__ri_cfg__.validateRuntime(classifier=True)
                    self.__ri__.load_classifier()
                    self.__ri__.auto_load = True
            if isinstance(people, list):
                if len(people) > 0:
                    if isinstance(people[0], Person):
                        if not img_is_mat: img = getCVMat(img)
                        if (size > 1 and self.__ri_cfg__.ri_name.lower() == self.__unistrings__.torchreid
                                and len({id(person) for person in people}) == len(people)):
                            res, reid_count[0] = self._reid_deep_batch(img, people, size)
                            if deduplicate:
                                res, reid_count[1] = self._reid_deep_batch_duplicates(img, res, size)
                        else:
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
        if self.__ri_cfg__.ri_name in ('FaceNet', 'ArcFace', 'AdaFace'):
            return self.__reidFaceNormal__(img, people)
        elif self.__ri_cfg__.ri_name.lower() == self.__unistrings__.torchreid:
            return self.__reidDeepNormal__(img, people)
        else:
            return self.__reidEmpty__(img, people)

    def __reidDupkiller__(self, img, people):
        if self.__ri_cfg__.ri_name in ('FaceNet', 'ArcFace', 'AdaFace'):
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
                try:
                    miniframe = _crop_reid_image(img, person.box_xyxy)
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

    def _deep_batch_jobs(self, img, people, selected):
        for index, person in enumerate(people):
            if selected(person.deepid):
                try:
                    crop = cv2.resize(_crop_reid_image(img, person.box_xyxy), self.__ri_cfg__.model_wh)
                except Exception as error:
                    add_warning_log(f'---PYPPBOX : Torchreid crop skipped: {error}')
                    continue
                yield index, crop

    def _run_deep_batches(self, people, jobs, size):
        completed = 0
        for chunk in _batches(jobs, size):
            results = None
            try:
                results = self.__ri__.recognize_batch([crop for _, crop in chunk],
                                                     is_bgr=True, batch_size=size)
                if len(results) != len(chunk) or any(len(result) != 2 for result in results):
                    raise ValueError('Torchreid returned an inconsistent recognition batch.')
            except Exception as error:
                # Leave the exception scope before retrying, releasing its inference
                # traceback/temporary tensor references (including failed CUDA batches).
                results = None
                add_warning_log(f'---PYPPBOX : Torchreid batch failed; retrying per person: {error}')
            if results is None:
                for index, crop in chunk:
                    try:
                        people[index].deepid, people[index].deepid_conf = self.__ri__.recognize(crop, is_bgr=True)
                        completed += 1
                    except Exception as error:
                        add_warning_log(f'---PYPPBOX : Torchreid recognition skipped: {error}')
            else:
                for (index, _), (identity, confidence) in zip(chunk, results):
                    people[index].deepid, people[index].deepid_conf = identity, confidence
                    completed += 1
            chunk.clear()
        return completed

    def _reid_deep_batch(self, img, people, size):
        jobs = self._deep_batch_jobs(img, people, lambda value:
            self.__unistrings__.err_did in value or self.__unistrings__.unk_did in value)
        completed = self._run_deep_batches(people, jobs, size)
        self.__deepidlistTMP__ = [person.deepid for person in people]
        return people, completed

    def _reid_deep_batch_duplicates(self, img, people, size):
        completed = 0
        duplicate_ids = [identity for identity, count in Counter(self.__deepidlistTMP__).items() if count > 1]
        # Build each group's jobs only after the preceding group has completed.
        # Earlier re-recognition can move a person into a later duplicate group.
        for identity in duplicate_ids:
            jobs = self._deep_batch_jobs(img, people, lambda value: value == identity)
            completed += self._run_deep_batches(people, jobs, size)
        return people, completed

    def __reidDupDeepkiller__(self, img, people):
        reid_count = 0
        if len(self.__deepidlistTMP__) != len(set(self.__deepidlistTMP__)):
            ddeepids = [k for k, v in Counter(self.__deepidlistTMP__).items() if v > 1]
            for ddeepid in ddeepids:
                index = 0
                for person in people:
                    try:
                        if person.deepid == ddeepid:
                            miniframe = _crop_reid_image(img, person.box_xyxy)
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
                try:
                    (x, y) = person.repspoint
                    miniframe = _crop_reid_image(img, (
                        x + int(self.__ri_cfg__.yl_w_calibration[0]),
                        y + int(self.__ri_cfg__.yl_h_calibration[0]),
                        x + int(self.__ri_cfg__.yl_w_calibration[1]),
                        y + int(self.__ri_cfg__.yl_h_calibration[1]),
                    )).copy()
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
                            miniframe = _crop_reid_image(img, (
                                x + int(self.__ri_cfg__.yl_w_calibration[0]),
                                y + int(self.__ri_cfg__.yl_h_calibration[0]),
                                x + int(self.__ri_cfg__.yl_w_calibration[1]),
                                y + int(self.__ri_cfg__.yl_h_calibration[1]),
                            )).copy()
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
        Construct one training reider, retaining the active one until training succeeds.
        On failure, dispose the candidate and retain the active reider. Returns None.
        Invalid explicit input/output directories are logged and training is skipped.
        Backend errors propagate; output files can still be overwritten by training.
        Config files are not saved. Load the written classifier before
        subsequent recognition; the pipeline does this on its next ``reidPeople()`` call.
        """
        self._ensure_open()
        candidate_cfg = copy(self.__cfg__)
        config, configured = self._select_stage('ri', reider, candidate_cfg)
        config = copy(config)
        if config.ri_name == 'None':
            raise ValueError('Classifier training requires FaceNet, ArcFace, AdaFace, or Torchreid.')
        if train_data != "":
            if not isExist(train_data):
                add_error_log(f"---PYPPBOX : train_data='{train_data}' does not exist.")
                return
            config.train_data = getAbsPathFDS(train_data)
        if classifier_pkl != "":
            if not isExist(getAncestorDir(classifier_pkl)):
                add_error_log(f"---PYPPBOX : classifier_pkl='{classifier_pkl}' is not valid.")
                return
            config.classifier_pkl = getAbsPathFDS(classifier_pkl)
        candidate = self._build_stage('ri', config, auto_load=False)
        try:
            add_info_log(f"---PYPPBOX : train_data='{config.train_data}'")
            add_info_log(f"---PYPPBOX : classifier_pkl='{config.classifier_pkl}'")
            candidate.train_classifier()
        except BaseException:
            self._retire(candidate)
            raise
        self._publish(candidate_cfg, configured, {'ri': (config, candidate)})
