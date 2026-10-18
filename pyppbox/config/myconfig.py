# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


import os
from typing import Any, Dict, List, Union, Optional

from .unifiedstrings import UnifiedStrings
from .validation import validate_config, validate_runtime
from .configtools import (PYPPBOXStructure, getCFGDict, getCFGDictList, 
                          loadDocumentList, dumpDocDict, dumpDocDictList, _dump_config_files,
                          _check_pending_save)
from pyppbox.utils.logtools import add_warning_log, add_error_log
from pyppbox.utils.commontools import (
    getFileName, 
    joinFPathFull, 
    getGlobalRootDir, 
    getAncestorDir, 
    getAdaptiveAbsPathFDS, 
    normalizePathFDS
)


internal_root_dir = getGlobalRootDir()
internal_config_dir = joinFPathFull(internal_root_dir, 'config')
internal_cfg_dir = joinFPathFull(internal_config_dir, 'cfg')
unified_strings = UnifiedStrings()


class BaseCGF(object):

    """An base CFG class used to store the necessary configurations of a module.

    Attributes
    ----------
    unified_strings : pyppbox.config.unifiedstrings.UnifiedStrings
        Set automatically.
        A :class:`~pyppbox.config.unifiedstrings.UnifiedStrings` object used to store unified strings.
    config : Dict[str, Any]
        Defaults to ``{}``.
        A configuration dictionary of a single document of the configuration.
    configs : List[Dict[str, Any]]
        Defaults to ``[]``.
        A list of multiple documents of the configurations.

    Notes
    -----
    Base mappings/lists are held by reference. Supported subclasses validate known
    fields into a new mapping, preserving extension fields without deep copying them.
    Loading a single document and a document list updates ``config`` and ``configs``
    separately. Dump methods select JSON for a .json output suffix and YAML otherwise.
    """

    def __init__(self):
        self.unified_strings = unified_strings
        self.config: Dict[str, Any] = {}
        self.configs: List[Dict[str, Any]] = []
    
    def loadDoc(self, input: Union[str, Dict[str, Any]]):
        """Load and set dictionary of a single document from a YAML/JSON file or string.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary. Supported subclasses
            validate required fields, numeric ranges, path types, and device syntax
            before storing the mapping. Model files are not opened here.
        """
        self.config = validate_config(type(self).__name__, getCFGDict(input), self.unified_strings)
    
    def validateRuntime(self, classifier=False):
        """Check inference assets before constructing models, without importing a backend.

        Parameters
        ----------
        classifier : bool
            Defaults to False. Include identity classifier files and deferred FaceNet
            models when True. Training directories and GT mapping files are not checked.

        Raises
        ------
        ValueError
            A required inference asset is missing, has the wrong type, or a Torchreid
            checkpoint has no dimensions in the model catalog. Device availability and
            checkpoint contents are validated by the backend when it loads them.
        """
        validate_runtime(self, classifier=classifier)

    def loadDocs(self, input: Union[str, Dict[str, Any]]):
        """Load and set a list of multiple documents from a YAML/JSON file or string.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        self.configs = getCFGDictList(input)
    
    def dumpDoc(self, output: str, header: str = ""):
        """Save the current ``config`` as YAML or JSON.

        Parameters
        ----------
        output : str
            Output filename. A .json suffix selects JSON; other suffixes select YAML.
            The parent directory must exist.
        header : str
            Defaults to ``""``. Prefix inserted verbatim for YAML; supply YAML comments
            if a description is needed. Ignored for JSON.

        Notes
        -----
        Returns None. Replaces one file atomically without reloading live modules.
        """
        dumpDocDict(output_file=output, doc=self.config, header=header)
    
    def dumpDocs(self, output: str, header: str = ""):
        """Save the current ``configs`` as YAML or JSON.

        Parameters
        ----------
        output : str
            Output filename. A .json suffix selects JSON; other suffixes select YAML.
            The parent directory must exist.
        header : str
            Defaults to ``""``. Prefix inserted verbatim for YAML; supply YAML comments
            if a description is needed. Ignored for JSON.

        Notes
        -----
        Returns None. Replaces one file atomically without reloading live modules.
        """
        dumpDocDictList(output_file=output, doc_list=self.configs, header=header)


class NoneCFG(BaseCGF):

    """
    A class used to act as a "None" configurator. 

    Attributes
    ----------
    dt_name : str
        "None" name of a detector.
    tk_name : str
        "None" name of a tracker.
    ri_name : str
        "None" name of a reider.
    detector : str
        "None" name of a detector.
    tracker : str
        "None" name of a tracker.
    reider : str
        "None" name of a reider.
    """

    def __init__(self, name: Any) -> None:
        """Initialize each stage name to the canonical string ``"None"``.

        Parameters
        ----------
        name : Any
            Ignored; retained for compatibility with config construction calls.
        """
        super().__init__()
        name = self.unified_strings.getUnifiedFormat("None")
        self.dt_name = name
        self.tk_name = name
        self.ri_name = name
        self.detector = name
        self.tracker = name
        self.reider = name

    def set(self, name: Any) -> "NoneCFG":
        """
        Return this already initialized "None" configuration; the name is ignored.

        Parameters
        ----------
        name : Any
            A parameter to be overridden with a unified string of :code:`"None"`.
        """
        return self


class MainCFG(BaseCGF):

    """
    A class used to store the main configurations in main.yaml.

    Attributes
    ----------
    detector : str
        Name of a detector.
    tracker : str
        Name of a tracker.
    reider : str
        Name of a reider.
    """

    def __init__(self) -> None:
        super().__init__()

    def set(self, input: Union[str, Dict[str, Any]]):
        """
        Set main configurations according to ``input``.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.config:
            try:
                self.detector = self.unified_strings.getUnifiedFormat(self.config['detector'])
                self.tracker = self.unified_strings.getUnifiedFormat(self.config['tracker'])
                self.reider = self.unified_strings.getUnifiedFormat(self.config['reider'])
                self.config = self.getDocument()
            except Exception as e:
                msg = f"MainCFG : set() -> {e}"
                add_error_log(msg)
                raise ValueError(msg)
        else:
            add_warning_log("MainCFG : set() -> The configuration is empty.")

    def getDocument(self):
        """Return a configuration dictionary of a single document of the parameters of the 
        main configurations.

        Returns
        -------
        dict
            A configuration dictionary of a single document of the configurations.
        """
        main_doc = {
            "detector": self.detector,
            "tracker": self.tracker,
            "reider": self.reider,
        }
        return dict(self.config, **main_doc)


class DCFGYOLOCLS(BaseCGF):

    """A class used to store the necessary configurations of detector YOLO Classic which
    use :code:`.weights` model.

    Attributes
    ----------
    dt_name : str
        Configured name of detector YOLO Classic.
    nms : float
        Parameter nms of YOLO Classic.
    conf : float
        Parameter conf of YOLO Classic.
    class_file : str
        Path of coco.names file of YOLO Classic.
    model_cfg_file : str
        Path of .cfg file of YOLO Classic.
    model_weights : str
        Path of .weights file of YOLO Classic.
    model_image_size : int
        Input image size of YOLO Classic.
    model_resolution : tuple(int, int)
        Input image resolution of YOLO Classic.
    repspoint_calibration : float
        Internal parameter, weight for calibatrating the repspoint of a
        :class:`pyppbox.utils.persontools.Person` object. Check :func:`~pyppbox.utils.persontools.findRepspoint` in
        :py:mod:`pyppbox.utils.persontools` for more details.
    from_dir : str
        Path of the root directory, relative to path of :attr:`model_weights`.

    Notes
    -----
    Call ``set()`` with a complete configuration before reading derived attributes
    or passing this object to a runtime module. Construction alone does not supply
    the values from the bundled YAML files. ``set()`` returns None; it is not a
    partial-update or fluent-builder API.
    """

    def __init__(self, relative_to_pyppbox_root=False):
        """Initialize the class according to the :obj:`relative_to_pyppbox_root` which defines 
        whether all the paths inside your configuration file are relative to :code:`{pyppbox root}` 
        or not. If all the paths inside your configuration file have full absolute paths, 
        setting :obj:`relative_to_pyppbox_root` is optional.

        Parameters
        ----------
        relative_to_pyppbox_root : bool
            Defaults to ``False``.
            (1) Set :code:`relative_to_pyppbox_root=False` to use your current working directory 
            as where all the paths in your configuration file are relative to; for example, 
            the path of :attr:`model_weights` inside your configuration file is set relatively 
            to your current working directory. 
            (2) Set :code:`relative_to_pyppbox_root=True` when all the paths in your configuration 
            file are relative to :code:`{pyppbox root}`.
        """
        super().__init__()
        self.from_dir = ""
        if relative_to_pyppbox_root:
            self.from_dir = internal_root_dir

    def set(self, input: Union[str, Dict[str, Any]]):
        """
        Set configurations according to ``input``.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.config:
            try:
                self.dt_name = self.unified_strings.getUnifiedFormat(self.config['dt_name'])
                self.nms = self.config['nms']
                self.conf = self.config['conf']
                self.class_file = getAdaptiveAbsPathFDS(self.from_dir, self.config['class_file'])
                self.model_cfg_file = getAdaptiveAbsPathFDS(self.from_dir, self.config['model_cfg_file'])
                self.model_weights = getAdaptiveAbsPathFDS(self.from_dir, self.config['model_weights'])
                self.model_image_size = self.config['model_image_size']
                self.model_resolution = (self.model_image_size, self.model_image_size)
                self.repspoint_calibration = self.config['repspoint_calibration']
                self.config = self.getDocument()
            except Exception as e:
                msg = f"DCFGYOLOCLS : set() -> {e}"
                add_error_log(msg)
                raise ValueError(msg)
        else:
            add_warning_log("DCFGYOLOCLS : set() -> The configuration is empty.")

    def getDocument(self):
        """Return a configuration dictionary of a single document of the attributes which 
        are the parameters of detector YOLO Classic.

        Returns
        -------
        dict
            A configuration dictionary of a single document of the configurations.
        """
        yolocs_doc = {
            "dt_name": self.dt_name,
            "nms": self.nms,
            "conf": self.conf,
            "class_file": normalizePathFDS(internal_root_dir, self.class_file),
            "model_cfg_file": normalizePathFDS(internal_root_dir, self.model_cfg_file),
            "model_weights": normalizePathFDS(internal_root_dir, self.model_weights),
            "model_image_size": self.model_image_size,
            "repspoint_calibration": self.repspoint_calibration
        }
        return dict(self.config, **yolocs_doc)


class DCFGYOLOULT(BaseCGF):

    """A class used to store the necessary configurations of detector YOLO_Ultralytics.

    Attributes
    ----------
    dt_name : str
        Configured name of detector YOLO_Ultralytics.
    conf : float
        Parameter conf of YOLO_Ultralytics.
    iou : float
        Parameter iou of YOLO_Ultralytics.
    imgsz : int
        Parameter imgsz of YOLO_Ultralytics.
    show_boxes : bool
        Parameter show_boxes of YOLO_Ultralytics.
    device : int or str or list[int] or None
        Device name, GPU index/indices, or the backend's automatic selection.
    max_det : int
        Parameter max_det of YOLO_Ultralytics.
    model_file : str
        Path of :attr:`model_file` for YOLO_Ultralytics.
    repspoint_calibration : float
        Weight for calibatrating the repspoint of a :class:`pyppbox.utils.persontools.Person`
        object. Check :func:`~pyppbox.utils.persontools.findRepspoint` in :py:mod:`pyppbox.utils.persontools` for more details.
    from_dir : str
        Path of the root directory, relative to path of :attr:`model_file`.

    Notes
    -----
    Call ``set()`` with a complete configuration before reading derived attributes
    or passing this object to a runtime module. Construction alone does not supply
    the values from the bundled YAML files. ``set()`` returns None; it is not a
    partial-update or fluent-builder API.
    """

    def __init__(self, relative_to_pyppbox_root=False):
        """Initialize the class according to the :obj:`relative_to_pyppbox_root` which 
        defines whether all the paths inside your configuration file are relative to 
        :code:`{pyppbox root}` or not. If all the paths inside your configuration file have 
        full absolute paths, setting :obj:`relative_to_pyppbox_root` is optional.

        Parameters
        ----------
        relative_to_pyppbox_root : bool
            Defaults to ``False``.
            (1) Set :code:`relative_to_pyppbox_root=False` to use your current working 
            directory as where all the paths in your configuration file are relative to; 
            for example, the path of :attr:`model_file` inside your configuration file is 
            set relatively to your current working directory. 
            (2) Set :code:`relative_to_pyppbox_root=True` when all the paths in your 
            configuration file are relative to :code:{pyppbox root}.
        """
        super().__init__()
        self.from_dir = ""
        if relative_to_pyppbox_root:
            self.from_dir = internal_root_dir

    def set(self, input: Union[str, Dict[str, Any]]):
        """
        Set configurations according to ``input``.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.config:
            try:
                self.dt_name = self.unified_strings.getUnifiedFormat(self.config['dt_name'])
                self.conf = self.config['conf']
                self.iou = self.config['iou']
                self.imgsz = self.config['imgsz']
                show_boxes = self.config['show_boxes']
                # Older GUI saves can contain quoted boolean strings. Normalize
                # this known boolean field without changing general YAML values.
                if isinstance(show_boxes, str) and show_boxes.strip().lower() in ('true', 'false'):
                    show_boxes = show_boxes.strip().lower() == 'true'
                if not isinstance(show_boxes, bool):
                    raise ValueError("show_boxes must be a boolean or the string 'True'/'False'.")
                self.show_boxes = show_boxes
                self.device = self.config['device']
                self.max_det = self.config['max_det']
                self.model_file = getAdaptiveAbsPathFDS(self.from_dir, self.config['model_file'])
                self.repspoint_calibration = self.config['repspoint_calibration']
                self.config = self.getDocument()
            except Exception as e:
                msg = f"DCFGYOLOULT : set() -> {e}"
                add_error_log(msg)
                raise ValueError(msg)
        else:
            add_warning_log("DCFGYOLOULT : set() -> The configuration is empty.")

    def getDocument(self):
        """Return yolout_doc, a configuration dictionary of a single document of the attributes which 
        are the parameters of detector YOLO_Ultralytics.

        Returns
        -------
        dict
            A configuration dictionary of a single document of the configurations.
        """
        yolout_doc = {
            "dt_name": self.dt_name,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "show_boxes": self.show_boxes,
            "device": self.device,
            "max_det": self.max_det,
            "model_file": normalizePathFDS(internal_root_dir, self.model_file),
            "repspoint_calibration": self.repspoint_calibration
        }
        return dict(self.config, **yolout_doc)


class DCFGRFDETR(BaseCGF):
    """Configuration for the optional RF-DETR detection backend.

    Attributes
    ----------
    dt_name : str
        Canonical detector name, ``RF_DETR``.
    model_size : str
        ``nano`` (default), ``small``, ``medium``, or ``large``. Each uses its
        published input resolution. Segmentation and Plus models are not included.
    model_file : str
        Local checkpoint, defaulting to the selected model's filename under
        ``data/modules/rf_detr``. Empty uses RF-DETR's download/cache instead.
        Custom checkpoints must match ``model_size``.
    device : int or str
        GPU index 0 (default), ``cpu``, ``cuda``, ``cuda:N``, ``mps``, or
        ``auto``. Integer indices are retained in configuration documents.
    conf : float
        Minimum detection confidence on a 0-1 scale; defaults to 0.4.
    max_det : int
        Maximum returned detections after class/width filtering; defaults to 100.
    person_class_id : int
        Native RF-DETR class ID for people; defaults to 1 for pretrained COCO
        models. Set the correct ID for custom checkpoints (often 0).
    optimize : bool
        Defaults to True. Prepare an inference-only model in place without JIT
        compilation. Set False to retain the original upstream model instance.
    repspoint_calibration : float
        Box representative-point calibration in [0, 1]; defaults to 0.25.
    from_dir : str
        Root used to resolve relative checkpoint paths.

    Notes
    -----
    Construction supplies defaults without importing RF-DETR or downloading
    weights. ``set()`` overlays a mapping on those defaults, not on previous
    values. Existing detector configurations retain their original behavior.
    """

    DEFAULTS = dict(dt_name='RF_DETR', conf=0.4, model_size='nano', device=0,
                    max_det=100, model_file='data/modules/rf_detr/rf-detr-nano.pth', person_class_id=1,
                    optimize=True, repspoint_calibration=0.25)

    MODEL_FILES = dict(nano='rf-detr-nano.pth', small='rf-detr-small.pth',
                       medium='rf-detr-medium.pth', large='rf-detr-large-2026.pth')

    def __init__(self, relative_to_pyppbox_root=False):
        """Initialize defaults and checkpoint path resolution.

        Parameters
        ----------
        relative_to_pyppbox_root : bool
            Defaults to False: resolve paths from the working directory.
            True resolves them from the pyppbox package root.
        """
        super().__init__()
        self.from_dir = internal_root_dir if relative_to_pyppbox_root else ''
        self.set(self.DEFAULTS)

    def set(self, input):
        """Load a YAML/JSON path or raw/ready mapping over RF-DETR defaults.

        Parameters
        ----------
        input : str or dict
            Configuration accepted by the existing configuration parser. An
            omitted model_file selects the local filename for model_size;
            an explicitly empty model_file enables automatic downloading.

        Raises
        ------
        ValueError
            A key is unknown or a value is invalid. Model files are checked when
            the detector is initialized, so configs can be edited before download.
        """
        supplied = getCFGDict(input)
        unknown = set(supplied) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f'Unknown RF_DETR configuration keys: {sorted(unknown)}')
        values = dict(self.DEFAULTS, **supplied)
        values['dt_name'] = self.unified_strings.getUnifiedFormat(values['dt_name'])
        if values['dt_name'] != 'RF_DETR':
            raise ValueError('dt_name must be RF_DETR.')
        values['model_size'] = str(values['model_size']).lower()
        if values['model_size'] not in ('nano', 'small', 'medium', 'large'):
            raise ValueError('model_size must be nano, small, medium, or large.')
        if 'model_file' not in supplied:
            values['model_file'] = 'data/modules/rf_detr/' + self.MODEL_FILES[values['model_size']]
        for name in ('conf', 'repspoint_calibration'):
            value = values[name]
            if isinstance(value, bool):
                raise ValueError(f'{name} must be a number in [0, 1].')
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f'{name} must be a number in [0, 1].') from error
            if not 0 <= value <= 1:
                raise ValueError(f'{name} must be a number in [0, 1].')
            values[name] = value
        for name, minimum in (('max_det', 1), ('person_class_id', 0)):
            value = values[name]
            try:
                integer = int(value)
                valid = not isinstance(value, bool) and integer == float(value) and integer >= minimum
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f'{name} must be an integer >= {minimum}.') from error
            if not valid:
                raise ValueError(f'{name} must be an integer >= {minimum}.')
            values[name] = integer
        optimize = values['optimize']
        if isinstance(optimize, str) and optimize.lower() in ('true', 'false'):
            optimize = optimize.lower() == 'true'
        if not isinstance(optimize, bool):
            raise ValueError('optimize must be a boolean.')
        values['optimize'] = optimize
        device = str(values['device']).strip().lower()
        if device.isdecimal():
            device = int(device)
        elif device not in ('auto', 'cpu', 'cuda', 'mps') and not (
                device.startswith('cuda:') and device[5:].isdecimal()):
            raise ValueError('device must be cpu, cuda, cuda:N, mps, auto, or an integer index.')
        values['device'] = device
        path = values['model_file']
        if not isinstance(path, str):
            raise ValueError('model_file must be a local path string, or empty for pretrained weights.')
        values['model_file'] = getAdaptiveAbsPathFDS(self.from_dir, path) if path else ''
        for name, value in values.items():
            setattr(self, name, value)
        self.config = self.getDocument()

    def getDocument(self):
        """Return a serializable mapping using the existing package-relative paths.

        Returns
        -------
        dict
            RF-DETR configuration; an empty checkpoint remains empty.
        """
        document = {name: getattr(self, name) for name in self.DEFAULTS}
        if self.model_file:
            document['model_file'] = normalizePathFDS(internal_root_dir, self.model_file)
        return document


class DCFGGT(BaseCGF):

    """A class used to store the necessary configurations of detector GT (Ground-truth).

    Attributes
    ----------
    dt_name : str
        Configured name of detector GT (Ground-truth).
    gt_file : str
        Path of GT text file.
    gt_map_file : str
        Path of GT map text file.
    from_dir : str
        Path of the root directory, relative to path of ``model_file``.

    Notes
    -----
    Call ``set()`` with a complete configuration before reading derived attributes
    or passing this object to a runtime module. Construction alone does not supply
    the values from the bundled YAML files. ``set()`` returns None; it is not a
    partial-update or fluent-builder API.
    """

    def __init__(self, relative_to_pyppbox_root=False):
        """Initialize the class according to the :obj:`relative_to_pyppbox_root` which defines 
        whether all the paths inside your configuration file are relative to :obj:`{pyppbox root}`
        or not. If all the paths inside your configuration file have full absolute paths, 
        setting :obj:`relative_to_pyppbox_root` is optional.

        Parameters
        ----------
        relative_to_pyppbox_root : bool
            Defaults to ``False``.
            (1) Set :code:`relative_to_pyppbox_root=False` to use your current working directory 
            as where all the paths in your configuration file are relative to; for example, 
            the path of :attr:`gt_file` inside your configuration file is set relatively to your 
            current working directory. 
            (2) Set :code:`relative_to_pyppbox_root=True` when all the paths in your configuration 
            file are relative to :code:`{pyppbox root}` like all the default paths inside in pyppbox's 
            internal configuration files are originally set relatively to :code:{pyppbox root}
        """
        super().__init__()
        self.from_dir = ""
        if relative_to_pyppbox_root:
            self.from_dir = internal_root_dir

    def set(self, input: Union[str, Dict[str, Any]]):
        """
        Set configurations according to ``input``.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.config:
            try:
                self.dt_name = self.unified_strings.getUnifiedFormat(self.config['dt_name'])
                self.gt_file = getAdaptiveAbsPathFDS(self.from_dir, self.config['gt_file'])
                self.gt_map_file = getAdaptiveAbsPathFDS(self.from_dir, self.config['gt_map_file'])
                self.config = self.getDocument()
            except Exception as e:
                msg = f"DCFGGT : set() -> {e}"
                add_error_log(msg)
                raise ValueError(msg)
        else:
            add_warning_log("DCFGGT : set() -> The configuration is empty.")

    def getDocument(self):
        """Return a configuration dictionary of a single document of the attributes which 
        are the parameters of detector GT (Ground-truth).

        Returns
        -------
        dict
            A configuration dictionary of a single document of the configurations.
        """
        gt_doc = {
            "dt_name": self.dt_name,
            "gt_file": normalizePathFDS(internal_root_dir, self.gt_file),
            "gt_map_file": normalizePathFDS(internal_root_dir, self.gt_map_file)
        }
        return dict(self.config, **gt_doc)


class TCFGCentroid(BaseCGF):

    """A class used to store the necessary configurations of tracker Centroid.

    Attributes
    ----------
    tk_name : str
        Configured name of tracker Centroid.
    max_spread : int
        Maximum distance of the being tracked :class:`pyppbox.utils.persontools.Person`
        object of previous and current state.

    Notes
    -----
    Call ``set()`` with a complete configuration before reading derived attributes
    or passing this object to a runtime module. Construction alone does not supply
    the values from the bundled YAML files. ``set()`` returns None; it is not a
    partial-update or fluent-builder API.
    """

    def __init__(self) -> None:
        super().__init__()

    def set(self, input):
        """
        Set configurations according to ``input``.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.config:
            try:
                self.tk_name = self.unified_strings.getUnifiedFormat(self.config['tk_name'])
                self.max_spread = self.config['max_spread']
                self.config = self.getDocument()
            except Exception as e:
                msg = f"TCFGCentroid : set() -> {e}"
                add_error_log(msg)
                raise ValueError(msg)
        else:
            add_warning_log("TCFGCentroid : set() -> The configuration is empty.")

    def getDocument(self):
        """Return a configuration dictionary of a single document of the attributes which 
        are the parameters of tracker Centroid.

        Returns
        -------
        dict
            A configuration dictionary of a single document of the configurations.
        """
        centroid_doc = {
            "tk_name": self.tk_name,
            "max_spread": self.max_spread
        }
        return dict(self.config, **centroid_doc)


class TCFGSORT(BaseCGF):

    """A class used to store the necessary configurations of tracker SORT.

    Attributes
    ----------
    tk_name : str
        Configured name of tracker SORT.
    max_age : int
        Parameter :obj:`max_age` of tracker SORT.
    min_hits : int
        Parameter :obj:`min_hits` of tracker SORT.
    iou_threshold : float
        Parameter :obj:`iou_threshold` of tracker SORT.

    Notes
    -----
    Call ``set()`` with a complete configuration before reading derived attributes
    or passing this object to a runtime module. Construction alone does not supply
    the values from the bundled YAML files. ``set()`` returns None; it is not a
    partial-update or fluent-builder API.
    """

    def __init__(self) -> None:
        super().__init__()

    def set(self, input: Union[str, Dict[str, Any]]):
        """
        Set configurations according to ``input``.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.config:
            try:
                self.tk_name = self.unified_strings.getUnifiedFormat(self.config['tk_name'])
                self.max_age = self.config['max_age']
                self.min_hits = self.config['min_hits']
                self.iou_threshold = self.config['iou_threshold']
                self.config = self.getDocument()
            except Exception as e:
                msg = f"TCFGSORT : set() -> {e}"
                add_error_log(msg)
                raise ValueError(msg)
        else:
            add_warning_log("TCFGSORT : set() -> The configuration is empty.")

    def getDocument(self):
        """
        Return a configuration dictionary of a single document of the attributes which 
        are the parameters of tracker SORT.

        Returns
        -------
        dict
            A configuration dictionary of a single document of the configurations.
        """
        sort_doc = {
            "tk_name": self.tk_name,
            "max_age": self.max_age,
            "min_hits": self.min_hits,
            "iou_threshold": self.iou_threshold
        }
        return dict(self.config, **sort_doc)
        

class TCFGDeepSORT(BaseCGF):

    """A class used to store the necessary configurations of tracker DeepSORT.

    Attributes
    ----------
    tk_name : str
        Configured name of tracker DeepSORT.
    nn_budget : int or None
        Nonnegative feature budget; None retains unlimited feature history.
    nms_max_overlap : float
        Parameter :obj:`nms_max_overlap` of tracker DeepSORT.
    max_cosine_distance : float
        Parameter :obj:`max_cosine_distance` of tracker DeepSORT.
    model_file : str
        Path of model file for tracker DeepSORT.
    from_dir : str
        Path of the root directory, relative to path of :attr:`model_file`.

    Notes
    -----
    Call ``set()`` with a complete configuration before reading derived attributes
    or passing this object to a runtime module. Construction alone does not supply
    the values from the bundled YAML files. ``set()`` returns None; it is not a
    partial-update or fluent-builder API.
    """

    def __init__(self, relative_to_pyppbox_root=False):
        """Initialize the class according to the :obj:`relative_to_pyppbox_root` which defines 
        whether all the paths inside your configuration file are relative to :code:`{pyppbox root}` 
        or not. If all the paths inside your configuration file have full absolute paths, setting 
        :obj:`relative_to_pyppbox_root` is optional.

        Parameters
        ----------
        relative_to_pyppbox_root : bool
            Defaults to ``False``.
            (1) Set :code:`relative_to_pyppbox_root=False` to use your current working directory 
            as where all the paths in your configuration file are relative to; for example, 
            the path of :attr:`model_file` inside your configuration file is set relatively to 
            your current working directory. 
            (2) Set :code:`relative_to_pyppbox_root=True` when all the paths in your configuration 
            file are relative to :code:`{pyppbox root}`.
        """
        super().__init__()
        self.from_dir = ""
        if relative_to_pyppbox_root:
            self.from_dir = internal_root_dir

    def set(self, input: Union[str, Dict[str, Any]]):
        """
        Set configurations according to ``input``.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.config:
            try:
                self.tk_name = self.unified_strings.getUnifiedFormat(self.config['tk_name'])
                self.nn_budget = self.config['nn_budget']
                self.nms_max_overlap = self.config['nms_max_overlap']
                self.max_cosine_distance = self.config['max_cosine_distance']
                self.model_file = getAdaptiveAbsPathFDS(self.from_dir, self.config['model_file'])
                self.config = self.getDocument()
            except Exception as e:
                msg = f"TCFGDeepSORT : set() -> {e}"
                add_error_log(msg)
                raise ValueError(msg)
        else:
            add_warning_log("TCFGDeepSORT : set() -> The configuration is empty.")

    def getDocument(self):
        """Return a configuration dictionary of a single document of the attributes which 
        are the parameters of tracker DeepSORT.

        Returns
        -------
        dict
            A configuration dictionary of a single document of the configurations.
        """
        deepsort_doc = {
            "tk_name": self.tk_name,
            "nn_budget": self.nn_budget,
            "nms_max_overlap": self.nms_max_overlap,
            "max_cosine_distance": self.max_cosine_distance,
            "model_file": normalizePathFDS(internal_root_dir, self.model_file)
        }
        return dict(self.config, **deepsort_doc)


class RCFGFaceNet(BaseCGF):

    """A class used to store the necessary configurations of reider FaceNet.

    Attributes
    ----------
    unified_strings : pyppbox.config.unifiedstrings.UnifiedStrings
        Set automatically.
        A :class:`~pyppbox.config.unifiedstrings.UnifiedStrings` object used to store unified strings.
    ri_name : str
        Configured name of reider FaceNet.
    gpu_mem : float
        Fraction of GPU memory supplied to the TensorFlow session.
    model_det : str
        Path of the det directory where stores .npy files.
    model_file : str
        Path of a pretrained model file for reider FaceNet.
    classifier_pkl : str
        Path of classifier PKL file.
    train_data : str
        Path of a data directory where there must be 2 or more sub-folders which
        classify different people.
    batch_size : int
        Parameter :obj:`batch_size` of reider FaceNet.
    min_confidence : float
        Configured confidence fraction on a 0-1 scale. Reider instances convert
        this to an integer percentage threshold.
    yl_h_calibration : list[int, int]
        Defaults to ``[-125, 75]``.
        When YOLO is used as the detector, this list of :code:`[val_1, val_2]` and a
        :class:`pyppbox.utils.persontools.Person`'s respoint :code:`(X, Y)` are used to find
        the from-to :code:`Y` for cropping the face: :code:`[Y + val_1 : Y + val_2, ...]`.
    yl_w_calibration : list[int, int]
        Defaults to ``[-55, 55]``.
        When YOLO is used as the detector, this list of :code:`[val_1, val_2]` and a
        :class:`pyppbox.utils.persontools.Person`'s respoint :code:`(X, Y)` are used to find
        the from-to :code:`X` for cropping the face: :code:`[..., X + val_1 : X + val_2]`.
    from_dir : str
        Path of the root directory, relative to path of :attr:`model_file`.

    Notes
    -----
    Call ``set()`` with a complete configuration before reading derived attributes
    or passing this object to a runtime module. Construction alone does not supply
    the values from the bundled YAML files. ``set()`` returns None; it is not a
    partial-update or fluent-builder API.
    """

    def __init__(self, relative_to_pyppbox_root=False):
        """Initialize the class according to the :obj:`relative_to_pyppbox_root` which defines 
        whether all the paths inside your configuration file are relative to :code:`{pyppbox root}` 
        or not. If all the paths inside your configuration file have full absolute paths, 
        setting :obj:`relative_to_pyppbox_root` is optional.

        Parameters
        ----------
        relative_to_pyppbox_root : bool
            Defaults to ``False``.
            (1) Set :code:`relative_to_pyppbox_root=False` to use your current working directory 
            as where all the paths in your configuration file are relative to; for example, 
            the path of :attr:`model_file` inside your configuration file is set relatively to 
            your current working directory. 
            (2) Set :code:`relative_to_pyppbox_root=True` when all the paths in your configuration 
            file are relative to :code:`{pyppbox root}`.
        """
        super().__init__()
        self.unified_strings = unified_strings
        self.from_dir = ""
        if relative_to_pyppbox_root:
            self.from_dir = internal_root_dir

    def set(self, input: Union[str, Dict[str, Any]]):
        """
        Set configurations according to ``input``.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.config:
            try:
                self.ri_name = self.unified_strings.getUnifiedFormat(self.config['ri_name'])
                self.gpu_mem = self.config['gpu_mem']
                self.model_det = getAdaptiveAbsPathFDS(self.from_dir, self.config['model_det'])
                self.model_file = getAdaptiveAbsPathFDS(self.from_dir, self.config['model_file'])
                self.classifier_pkl = getAdaptiveAbsPathFDS(self.from_dir, self.config['classifier_pkl'])
                self.train_data = getAdaptiveAbsPathFDS(self.from_dir, self.config['train_data'])
                self.batch_size = self.config['batch_size']
                self.min_confidence = self.config['min_confidence']
                self.yl_h_calibration = self.config['yl_h_calibration']
                self.yl_w_calibration = self.config['yl_w_calibration']
                self.config = self.getDocument()
            except Exception as e:
                msg = f"RCFGFaceNet : set() -> {e}"
                add_error_log(msg)
                raise ValueError(msg)
        else:
            add_warning_log("RCFGFaceNet : set() -> The configuration is empty.")

    def getDocument(self):
        """Return a configuration dictionary of a single document of the attributes which 
        are the parameters of reider FaceNet.

        Returns
        -------
        dict
            A configuration dictionary of a single document of the configurations.
        """
        facenet_doc = {
            "ri_name": self.ri_name,
            "gpu_mem": self.gpu_mem,
            "model_det": normalizePathFDS(internal_root_dir, self.model_det), 
            "model_file": normalizePathFDS(internal_root_dir, self.model_file),
            "classifier_pkl": normalizePathFDS(internal_root_dir, self.classifier_pkl),
            "train_data": normalizePathFDS(internal_root_dir, self.train_data),
            "batch_size": self.batch_size,
            "min_confidence": self.min_confidence,
            "yl_h_calibration": self.yl_h_calibration,
            "yl_w_calibration": self.yl_w_calibration
        }
        return dict(self.config, **facenet_doc)


class RCFGArcFace(BaseCGF):
    """Optional ONNX face recognizer with explicit local detector and weights.

    ``set`` overlays defaults, validates without loading inference libraries, and
    resolves paths as other pyppbox configs do. Both networks are loaded lazily.
    ``min_confidence`` applies to the identity SVM, not cosine similarity.
    Preprocessing settings must match the exported embedding checkpoint.
    """
    NAME = 'ArcFace'
    DEFAULTS = dict(ri_name='ArcFace',
                    model_det='data/modules/arcface/models/det_10g.onnx',
                    model_file='data/modules/arcface/models/w600k_r50.onnx',
                    classifier_pkl='data/modules/arcface/classifier/gta5.pkl',
                    train_data='data/datasets/GTA_V_DATASET/face_182x182', min_confidence=0.75,
                    device='cpu', det_size=320, det_thresh=0.5,
                    input_color='RGB', input_mean=127.5, input_std=127.5,
                    num_threads=4, yl_h_calibration=[-125, 75], yl_w_calibration=[-55, 55])

    def __init__(self, relative_to_pyppbox_root=False):
        super().__init__()
        self.from_dir = internal_root_dir if relative_to_pyppbox_root else ''
        self.set({})

    def set(self, input):
        supplied = getCFGDict(input)
        unknown = set(supplied) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f'Unknown {self.NAME} configuration keys: {sorted(unknown)}')
        values = validate_config(type(self).__name__, dict(self.DEFAULTS, **supplied), self.unified_strings)
        values['ri_name'] = self.unified_strings.getUnifiedFormat(values['ri_name'])
        if values['ri_name'] != self.NAME:
            raise ValueError(f'ri_name must be {self.NAME}.')
        if values['device'] != 'cpu' and not values['device'].startswith('cuda'):
            raise ValueError('device must be cpu, cuda, or cuda:N for ONNX face recognition.')
        if values['det_size'] % 32:
            raise ValueError('det_size must be a positive multiple of 32.')
        for field in ('model_det', 'model_file', 'classifier_pkl', 'train_data'):
            if not values[field]:
                raise ValueError(f'{field} must be a nonempty path.')
            values[field] = getAdaptiveAbsPathFDS(self.from_dir, values[field])
        # Publish only after every field has passed validation.
        self.__dict__.update(values)
        self.config = self.getDocument()

    def getDocument(self):
        result = {field: getattr(self, field) for field in self.DEFAULTS}
        for field in ('model_det', 'model_file', 'classifier_pkl', 'train_data'):
            result[field] = normalizePathFDS(internal_root_dir, result[field])
        return result


class RCFGAdaFace(RCFGArcFace):
    """AdaFace ONNX configuration; defaults match the CVLFace RGB IR18 export."""
    NAME = 'AdaFace'
    DEFAULTS = dict(RCFGArcFace.DEFAULTS, ri_name='AdaFace',
                    model_det='data/modules/adaface/models/det_10g.onnx',
                    model_file='data/modules/adaface/models/adaface_ir18_vgg2.onnx',
                    classifier_pkl='data/modules/adaface/classifier/gta5.pkl')


class RCFGTorchreid(BaseCGF):

    """A class used to store the necessary configurations of reider Torchreid.

    Attributes
    ----------
    unified_strings : pyppbox.config.unifiedstrings.UnifiedStrings
        Set automatically.
        A :class:`~pyppbox.config.unifiedstrings.UnifiedStrings` object used to store unified strings.
    ri_name : str
        Configured name of reider Torchreid.
    classifier_pkl : str
        Path of classifier PKL file.
    train_data : str
        Path of a data directory where there must be 2 or more sub-folders which classify
        different people.
    model_name : str
        Name of a model corresponding to the pretrained model.
    model_path : str
        Path of a pretrained model file for reider Torchreid.
    min_confidence : float
        Configured confidence fraction on a 0-1 scale. Reider instances convert
        this to an integer percentage threshold.
    device : str
        Device name such as cpu, cuda, or cuda:N. Legacy integer GPU indices
        are normalized to cuda:N before feature-extractor initialization.
    base_model_path : str
        Path of a base model corresponding to the pretrained model or :attr:`model_name`.
    model_dict : ``TorchreidModelDict``
        Set automatically.
        A ``TorchreidModelDict`` object used to store the dictionary of a Torchreid model.
    model_wh : tuple(int, int)
        Set automatically.
        A tuple used to store the input image size :code:`(width, height)` for a corresponding
        Torchreid model.
    from_dir : str
        Path of the root directory, relative to path of :attr:`model_path`.

    Notes
    -----
    Call ``set()`` with a complete configuration before reading derived attributes
    or passing this object to a runtime module. Construction alone does not supply
    the values from the bundled YAML files. ``set()`` returns None; it is not a
    partial-update or fluent-builder API.
    """

    def __init__(self, relative_to_pyppbox_root=False):
        """Initialize the class according to the :obj:`relative_to_pyppbox_root` which defines 
        whether all the paths inside your configuration file are relative to :code:`{pyppbox root}` 
        or not. If all the paths inside your configuration file have full absolute paths, 
        setting :obj:`relative_to_pyppbox_root` is optional.

        Parameters
        ----------
        relative_to_pyppbox_root : bool
            Defaults to ``False``.
            (1) Set :code:`relative_to_pyppbox_root=False` to use your current working directory 
            as where all the paths in your configuration file are relative to; for example, 
            the path of :attr:`model_path` inside your configuration file is set relatively to your 
            current working directory. 
            (2) Set :code:`relative_to_pyppbox_root=True` when all the paths in your configuration 
            file are relative to :code:`{pyppbox root}`.
        """
        super().__init__()
        self.unified_strings = unified_strings
        self.from_dir = ""
        if relative_to_pyppbox_root:
            self.from_dir = internal_root_dir

    def set(self, input: Union[str, Dict[str, Any]]):
        """
        Set configurations according to ``input``.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.config:
            try:
                from .torchreidmodels import TorchreidModelDict
                self.ri_name = self.unified_strings.getUnifiedFormat(self.config['ri_name'])
                self.classifier_pkl = getAdaptiveAbsPathFDS(self.from_dir, self.config['classifier_pkl'])
                self.train_data = getAdaptiveAbsPathFDS(self.from_dir, self.config['train_data'])
                self.model_name = self.config['model_name']
                self.model_path = getAdaptiveAbsPathFDS(self.from_dir, self.config['model_path'])
                self.min_confidence = self.config['min_confidence']
                self.device = self.config['device']
                self.config = self.getDocument()
                self.base_model_path = getAdaptiveAbsPathFDS(
                    self.from_dir, 
                    joinFPathFull(getAncestorDir(self.model_path, 1), 'base')
                )
                self.model_dict = TorchreidModelDict()
                self.model_wh = self.model_dict.getWH(getFileName(self.model_path))
            except Exception as e:
                msg = f"RCFGTorchreid : set() -> {e}"
                add_error_log(msg)
                raise ValueError(msg)
        else:
            add_warning_log("RCFGTorchreid : set() -> The configuration is empty.")

    def getDocument(self):
        """Return a configuration dictionary of a single document of the attributes which 
        are the parameters of reider Torchreid.

        Returns
        -------
        dict
            A configuration dictionary of a single document of the configurations.
        """
        torchreid_doc = {
            "ri_name": self.ri_name,
            "classifier_pkl": normalizePathFDS(internal_root_dir, self.classifier_pkl),
            "train_data": normalizePathFDS(internal_root_dir, self.train_data),
            "model_name": self.model_name,
            "model_path": normalizePathFDS(internal_root_dir, self.model_path),
            "min_confidence": self.min_confidence,
            "device": self.device
        }
        return dict(self.config, **torchreid_doc)


class MyCFGHeaders(object):

    """
    A class used to store the headers of configuration files and others.
    """

    def mainHeader(self):
        """Return header string for main.yaml file.

        Returns
        -------
        str
            Header string for main.yaml file.
        """
        header=("###########################################################\n"
                "# Main config:\n"
                "###########################################################\n"
                "# detector: None | YOLO_Classic | YOLO_Ultralytics | RF_DETR | GT\n"
                "# tracker: None | Centroid | SORT | DeepSORT\n"
                "# reider: None | FaceNet | ArcFace | AdaFace | Torchreid\n"
                "###########################################################\n")
        return header

    def detectorHeader(self):
        """Return header string for detectors.yaml file.

        Returns
        -------
        str
            Header string for detectors.yaml file.
        """
        header=("###########################################################\n"
                "# Detector config:\n"
                "###########################################################\n"
                "# --- # YOLO_Classic\n"
                "# dt_name: YOLO_Classic\n"
                "# nms: 0.7\n"
                "# conf: 0.4\n"
                "# class_file: data/modules/yolo_classic/coco.names\n"
                "# model_cfg_file: data/modules/yolo_classic/yolov4.cfg\n"
                "# model_weights: data/modules/yolo_classic/yolov4.weights\n"
                "# model_image_size: 1024\n"
                "# repspoint_calibration: 0.25\n"
                "###########################################################\n"
                "# --- # YOLO_Ultralytics\n"
                "# dt_name: YOLO_Ultralytics\n"
                "# conf: 0.4\n"
                "# iou: 0.7\n"
                "# imgsz: 1024\n"
                "# show_boxes: True\n"
                "# device: 0\n"
                "# max_det: 100\n"
                "# model_file: data/modules/yolo_ultralytics/yolov8s-pose.pt\n"
                "# repspoint_calibration: 0.25\n"
                "###########################################################\n"
                "# --- # RF_DETR (optional: pyppbox[rfdetr], Python 3.10+)\n"
                "# dt_name: RF_DETR\n"
                "# conf: 0.4\n"
                "# model_size: nano\n"
                "# device: 0\n"
                "# max_det: 100\n"
                "# model_file: data/modules/rf_detr/rf-detr-nano.pth\n"
                "# person_class_id: 1\n"
                "# optimize: True\n"
                "# repspoint_calibration: 0.25\n"
                "###########################################################\n"
                "# --- # GT aka Ground Truth\n"
                "# dt_name: GT\n"
                "# gt_file: data/datasets/GTA_V_DATASET/ground_truth/realID_hard_sur.txt\n"
                "# gt_map_file: data/datasets/GTA_V_DATASET/ground_truth/gt_map.txt\n"
                "###########################################################\n")
        return header

    def trackerHeader(self):
        """Return header string for trackers.yaml file.

        Returns
        -------
        str
            Header string for trackers.yaml file.
        """
        header=("###########################################################\n"
                "# Tracker config:\n"
                "###########################################################\n"
                "# --- # Centroid\n"
                "# tk_name: Centroid\n"
                "# max_spread: 64\n"
                "###########################################################\n"
                "# --- # SORT\n"
                "# tk_name: SORT\n"
                "# max_age: 1\n"
                "# min_hits: 3\n"
                "# iou_threshold: 0.3\n"
                "###########################################################\n"
                "# --- # DeepSORT\n"
                "# tk_name: DeepSORT\n"
                "# nn_budget: 100\n"
                "# nms_max_overlap: 0.5\n"
                "# max_cosine_distance: 0.1\n"
                "# model_file: data/modules/deepsort/mars-small128.pb\n"
                "###########################################################\n")
        return header

    def reiderHeader(self):
        """Return header string for reiders.yaml file.

        Returns
        -------
        str
            Header string for reiders.yaml file.
        """
        header=("###########################################################\n"
                "# ReIDer config:\n"
                "###########################################################\n"
                "# --- # FaceNet\n"
                "# ri_name: FaceNet\n"
                "# gpu_mem: 0.585\n"
                "# model_det: data/modules/facenet/models/det\n"
                "# model_file: data/modules/facenet/models/20180402-114759/20180402-114759.pb\n"
                "# classifier_pkl: data/modules/facenet/classifier/gta5.pkl\n"
                "# train_data: data/datasets/GTA_V_DATASET/face_182x182\n"
                "# batch_size: 1000\n"
                "# min_confidence: 0.75\n"
                "# yl_h_calibration: [-125, 75]\n"
                "# yl_w_calibration: [-55, 55]\n"
                "###########################################################\n"
                "# --- # Torchreid\n"
                "# ri_name: Torchreid\n"
                "# classifier_pkl: data/modules/torchreid/classifier/gta5_osnet_ain_ms_d_c.pkl\n"
                "# train_data: data/datasets/GTA_V_DATASET/body_128x256\n"
                "# model_name: osnet_ain_x1_0\n"
                "# model_path: data/modules/torchreid/models/torchreid/osnet_ain_ms_d_c.pth.tar\n"
                "# min_confidence: 0.35\n"
                "# device: cuda\n"
                "###########################################################\n")
        return header

    def copyrightCMDHeader(self):
        """
        Return copyright header string.

        Returns
        -------
        str
            Copyright header string.
        """
        header = ":: Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License\n\n"
        return header


class MyConfigurator(PYPPBOXStructure):

    """A class used to store and manage all the configurations of pyppbox.

    Attributes
    ----------
    mcfg : MainCFG
        Set automatically.
        A :class:`MainCFG` object used to identify the main detector/tracker/reider.
    dcfg_yolocs : pyppbox.config.myconfig.DCFGYOLOCLS
        Set automatically.
        A :class:`~pyppbox.config.myconfig.DCFGYOLOCLS` object used to store the configurations of detector
        YOLO Classic.
    dcfg_yolout : pyppbox.config.myconfig.DCFGYOLOULT
        Set automatically.
        A :class:`~pyppbox.config.myconfig.DCFGYOLOULT` object used to store the configurations of detector
        YOLO_Ultralytics.
    dcfg_rfdetr : DCFGRFDETR
        RF-DETR settings; defaults are available even in V3 config directories.
    dcfg_gt : DCFGGT
        Set automatically.
        A :class:`DCFGGT` object used to store the configurations of detector GT (Ground-truth).
    tcfg_centroid : TCFGCentroid
        Set automatically.
        A :class:`TCFGCentroid` object used to store the configurations of tracker Centroid.
    tcfg_sort : TCFGSORT
        Set automatically.
        A :class:`TCFGSORT` object used to store the configurations of tracker SORT.
    tcfg_deepsort : pyppbox.config.myconfig.TCFGDeepSORT
        Set automatically.
        A :class:`~pyppbox.config.myconfig.TCFGDeepSORT` object used to store the configurations of tracker DeepSORT.
    rcfg_facenet : pyppbox.config.myconfig.RCFGFaceNet
        Set automatically.
        A :class:`~pyppbox.config.myconfig.RCFGFaceNet` object used to store the configurations of reider FaceNet.
    rcfg_torchreid : pyppbox.config.myconfig.RCFGTorchreid
        Set automatically.
        A :class:`~pyppbox.config.myconfig.RCFGTorchreid` object used to store the configurations of reider Torchreid.
    dt_map : list[str, ...]
        Set automatically.
        A list used to store all detectors' names loaded from the configuration file.
    tk_map : list[str, ...]
        Set automatically.
        A list used to store all trackers' names loaded from the configuration file.
    ri_map : list[str, ...]
        Set automatically.
        A list used to store all reiders' names loaded from the configuration file.
    cfg_headers : MyCFGHeaders
        A :class:`MyCFGHeaders` object used to store the headers of configuration.
    relative_to_pyppbox_root : bool
        Set automatically.
        An indication of whether the paths inside all configuration files are relative
        to :code:`{pyppbox root}` or not.
    abstractCFGs : list
        A list used to store abstract or unsupported :class:`BaseCGF` objects.

    Notes
    -----
    This class manages configuration data only; it does not construct detector,
    tracker, or reider models. With the default ``set_all_modules=False``, call the
    appropriate loading methods before accessing their config attributes. Dump
    methods save supplied mappings but do not reload live pipeline instances.
    """

    def __init__(self, cfg_dir: str = internal_cfg_dir, set_all_modules: bool = False):
        """Initialize according to the given directory of the YAML configurations.

        Parameters
        ----------
        cfg_dir : str
            Defaults to ``'{pyppbox root}/config/cfg'``.
            A path of the config directory where stores main.yaml, detectors.yaml, 
            trackers.yaml, and reiders.yaml.
        set_all_modules : bool
            Defaults to ``False``.
            An indication of whether to load and set all configurations of all supported 
            modules. :code:`set_all_modules=True` will trigger :meth:`setMCFG()`, 
            :meth:`setAllDCFG()`, :meth:`setAllTCFG()`, and :meth:`setAllRCFG()`.
        """
        super().__init__(cfg_dir=cfg_dir)
        self.abstractCFGs = []
        self.relative_to_pyppbox_root = False
        if cfg_dir == internal_cfg_dir:
            self.relative_to_pyppbox_root = True
        self.cfg_headers = MyCFGHeaders()
        if set_all_modules:
            self.setMCFG()
            self.setAllDCFG()
            self.setAllTCFG()
            self.setAllRCFG()

    def setCustomCFG(self, cfg_dir: str):
        """Set a path of a custom config directory where stores main.yaml, detectors.yaml, 
        trackers.yaml, and reiders.yaml.

        Parameters
        ----------
        cfg_dir : str
            A path of the config directory.
        """
        super().setCustomCFG(cfg_dir=cfg_dir)
        self.relative_to_pyppbox_root = False
        if cfg_dir == internal_cfg_dir:
            self.relative_to_pyppbox_root = True

    def getMCFG(self):
        """Get the main configurations which used to identify the main detector/tracker/reider.

        Returns
        -------
        dict
            A configuration dictionary of the main configurations.
        """
        return self.mcfg.getDocument()

    def setMCFG(self, main_yaml: Optional[Union[str, Dict[str, Any]]] = None):
        """Load and set the main configurations which used to identify the main detector/tracker/reider.

        Parameters
        ----------
        main_yaml : str or dict
            Defaults to ``None``.
            A YAML/JSON file path, or a raw/ready dictionary of the main configurations 
            or main.yaml. This ``main_yaml`` helps override the original one configured
            during the ``__init__()``.
            Leave it as default :code:`main_yaml=None`, to load and set according to the 
            configurations inside config directory :attr:`~pyppbox.config.configtools.PYPPBOXStructure.cfg_dir` set in the ``__init__()``.
        """
        self.mcfg = MainCFG()
        if main_yaml is None:
            self.mcfg.set(self.main_yaml)
        else:
            self.mcfg.set(main_yaml)
    
    def setMainModules(self, main_yaml: Optional[Union[str, Dict[str, Any]]] = None):
        """Call :meth:`setMCFG()` and load the selected stages' configuration objects.

        This does not construct detector, tracker, or reider runtime instances.

        Parameters
        ----------
        main_yaml : str or dict
            Defaults to ``None``.
            A YAML/JSON file path, or a raw/ready dictionary of the main configurations 
            or main.yaml. This ``main_yaml`` helps override the original one configured
            during the ``__init__()``.
            Leave it as default :code:`main_yaml=None`, to load and set according to the 
            configurations inside config directory :attr:`~pyppbox.config.configtools.PYPPBOXStructure.cfg_dir` set in the ``__init__()``.
        """
        self.setMCFG(main_yaml=main_yaml)
        if not self.mcfg.config:
            raise ValueError('main.yaml: the main configuration is empty.')
        for stage, key, filename in (('detector', 'dt_name', self.detectors_yaml),
                                      ('tracker', 'tk_name', self.trackers_yaml),
                                      ('reider', 'ri_name', self.reiders_yaml)):
            selected = self.mcfg.config[stage]
            supported = {'detector': ('YOLO_Classic', 'YOLO_Ultralytics', 'RF_DETR', 'GT', 'None'),
                         'tracker': ('Centroid', 'SORT', 'DeepSORT', 'None'),
                         'reider': ('FaceNet', 'ArcFace', 'AdaFace', 'Torchreid', 'None')}
            if selected not in supported[stage]:
                raise ValueError(f'main.yaml: unsupported {stage} selection: {selected}.')
            documents = loadDocumentList(filename)
            if selected == 'None':
                continue
            for document in documents:
                if self.unified_strings.getUnifiedFormat(document.get(key, '')).lower() == selected.lower():
                    self.setASupportedModuleCFG(document, self.relative_to_pyppbox_root)
                    break
            else:
                if selected == 'RF_DETR':
                    # Older config directories do not contain this optional detector.
                    self.dcfg_rfdetr = DCFGRFDETR(self.relative_to_pyppbox_root)
                else:
                    raise ValueError(f'{filename}: missing {key} configuration for {selected}.')
    
    def setGTCFG(
        self, 
        detectors_yaml: Optional[Union[str, Dict[str, Any]]] = None, 
        relative_to_pyppbox_root: Optional[bool] = None
    ):
        """Manually load and set the configurations for :attr:`dcfg_gt`.

        Parameters
        ----------
        detectors_yaml : str or dict
            Defaults to ``None``.
            A YAML/JSON file path, or a raw/ready dictionary of the detectors' configurations 
            or detectors.yaml. This ``detectors_yaml`` helps override the original one
            configured during the ``__init__()``.
            Leave it as default :code:`detectors_yaml=None`, to load and set according to the 
            configurations inside config directory :attr:`~pyppbox.config.configtools.PYPPBOXStructure.cfg_dir` set in the ``__init__()``.
        relative_to_pyppbox_root : bool
            Defaults to ``None``.
            An indication of whether the paths inside your configuration file are relative 
            to :code:`{pyppbox root}` or not. This indication or behavior was automatically 
            decided during the ``__init__()``, and you don't need to set or change it
            unless you set ``detectors_yaml`` to override the original configurations. If
            all the paths inside your configuration file have full absolute paths, setting 
            :obj:`relative_to_pyppbox_root` is optional.
        """
        if isinstance(relative_to_pyppbox_root, bool):
            self.dcfg_gt = DCFGGT(relative_to_pyppbox_root)
        elif relative_to_pyppbox_root is None:
            self.dcfg_gt = DCFGGT(self.relative_to_pyppbox_root)
        else:
            msg = "MyConfigurator : setGTCFG() -> 'relative_to_pyppbox_root' is not valid."
            add_error_log(msg)
            raise ValueError(msg)
        docs = []
        if detectors_yaml is None:
            docs = loadDocumentList(self.detectors_yaml)
        elif detectors_yaml is not None:
            docs = getCFGDictList(detectors_yaml)
        for d in docs:
            try:
                if d['dt_name'].lower() == self.unified_strings.gt:
                    self.dcfg_gt.set(d)
            except Exception as e:
                msg = f"MyConfigurator : setGTCFG() -> {e}"
                add_error_log(msg)
                raise ValueError(msg)

    def setAllDCFG(
        self, 
        detectors_yaml: Optional[Union[str, Dict[str, Any]]] = None, 
        relative_to_pyppbox_root: Optional[bool] = None
    ):
        """Load and set the configurations for all supported detectors.

        Parameters
        ----------
        detectors_yaml : str or dict
            Defaults to ``None``.
            A YAML/JSON file path, or a raw/ready dictionary of the detectors' configurations 
            or detectors.yaml. This ``detectors_yaml`` helps override the original one
            configured during the ``__init__()``.
            Leave it as default :code:`detectors_yaml=None`, to load and set according to the 
            configurations inside config directory :attr:`~pyppbox.config.configtools.PYPPBOXStructure.cfg_dir` set in the ``__init__()``.
        relative_to_pyppbox_root : bool
            Defaults to ``None``.
            An indication of whether the paths inside your configuration file are relative 
            to :code:`{pyppbox root}` or not. This indication or behavior was automatically 
            decided during the ``__init__()``, and you don't need to set or change it
            unless you set ``detectors_yaml`` to override the original configurations. If
            all the paths inside your configuration file have full absolute paths, setting 
            :obj:`relative_to_pyppbox_root` is optional.
        """
        if isinstance(relative_to_pyppbox_root, bool):
            self.dcfg_yolocs = DCFGYOLOCLS(relative_to_pyppbox_root)
            self.dcfg_yolout = DCFGYOLOULT(relative_to_pyppbox_root)
            self.dcfg_rfdetr = DCFGRFDETR(relative_to_pyppbox_root)
            self.dcfg_gt = DCFGGT(relative_to_pyppbox_root)
        elif relative_to_pyppbox_root is None:
            self.dcfg_yolocs = DCFGYOLOCLS(self.relative_to_pyppbox_root)
            self.dcfg_yolout = DCFGYOLOULT(self.relative_to_pyppbox_root)
            self.dcfg_rfdetr = DCFGRFDETR(self.relative_to_pyppbox_root)
            self.dcfg_gt = DCFGGT(self.relative_to_pyppbox_root)
        else:
            msg = "MyConfigurator : setAllDCFG() -> 'relative_to_pyppbox_root' is not valid."
            add_error_log(msg)
            raise ValueError(msg)
        docs = []
        if detectors_yaml is None:
            docs = loadDocumentList(self.detectors_yaml)
        elif detectors_yaml is not None:
            docs = getCFGDictList(detectors_yaml)
        self.dt_map = []
        for d in docs:
            try:
                if d['dt_name'].lower() == self.unified_strings.yolo_cls:
                    self.dcfg_yolocs.set(d)
                    self.dt_map.append(self.dcfg_yolocs.dt_name)
                elif d['dt_name'].lower() == self.unified_strings.yolo_ult:
                    self.dcfg_yolout.set(d)
                    self.dt_map.append(self.dcfg_yolout.dt_name)
                elif self.unified_strings.getUnifiedFormat(d['dt_name']).lower() == self.unified_strings.rf_detr:
                    self.dcfg_rfdetr.set(d)
                    self.dt_map.append(self.dcfg_rfdetr.dt_name)
                elif d['dt_name'].lower() == self.unified_strings.gt:
                    self.dcfg_gt.set(d)
                    self.dt_map.append(self.dcfg_gt.dt_name)
                else:
                    msg = f"MyConfigurator : setAllDCFG() -> Name '{d['dt_name']}' is not supported."
                    add_warning_log(msg)
            except Exception as e:
                msg = f"MyConfigurator : setAllDCFG() -> {e}"
                add_error_log(msg)
                raise ValueError(msg)

        # Keep RF-DETR after Ultralytics, including in older config directories.
        self.dt_map = [name for name in self.dt_map if name != self.dcfg_rfdetr.dt_name]
        ultralytics = self.unified_strings.getUnifiedFormat('YOLO_Ultralytics')
        position = self.dt_map.index(ultralytics) + 1 if ultralytics in self.dt_map else len(self.dt_map)
        self.dt_map.insert(position, self.dcfg_rfdetr.dt_name)

    def setAllTCFG(
        self, 
        trackers_yaml: Optional[Union[str, Dict[str, Any]]] = None, 
        relative_to_pyppbox_root: Optional[bool] = None
    ):
        """Load and set the configurations for all supported trackers.

        Parameters
        ----------
        trackers_yaml : str or dict
            Defaults to ``None``.
            A YAML/JSON file path, or a raw/ready dictionary of the trackers' configurations 
            or trackers.yaml. This ``trackers_yaml`` helps override the original one
            configured during the ``__init__()``.
            Leave it as default :code:`trackers_yaml=None`, to load and set according to the 
            configurations inside config directory :attr:`~pyppbox.config.configtools.PYPPBOXStructure.cfg_dir` set in the ``__init__()``.
        relative_to_pyppbox_root : bool
            Defaults to ``None``.
            An indication of whether the paths inside your configuration file are relative 
            to :code:`{pyppbox root}` or not. This indication or behavior was automatically 
            decided during the ``__init__()``, and you don't need to set or change it
            unless you set ``trackers_yaml`` to override the original configurations. If
            all the paths inside your configuration file have full absolute paths, setting 
            :obj:`relative_to_pyppbox_root` is optional.
        """
        self.tcfg_centroid = TCFGCentroid()
        self.tcfg_sort = TCFGSORT()

        if isinstance(relative_to_pyppbox_root, bool):
            self.tcfg_deepsort = TCFGDeepSORT(relative_to_pyppbox_root)
        elif relative_to_pyppbox_root is None:
            self.tcfg_deepsort = TCFGDeepSORT(self.relative_to_pyppbox_root)
        else:
            msg = "MyConfigurator : setAllTCFG() -> 'relative_to_pyppbox_root' is not valid."
            add_error_log(msg)
            raise ValueError(msg)
        docs = []
        if trackers_yaml is None:
            docs = loadDocumentList(self.trackers_yaml)
        elif trackers_yaml is not None:
            docs = getCFGDictList(trackers_yaml)
        self.tk_map = []
        for d in docs:
            try:
                if d['tk_name'].lower() == self.unified_strings.centroid:
                    self.tcfg_centroid.set(d)
                    self.tk_map.append(self.tcfg_centroid.tk_name)
                elif d['tk_name'].lower() == self.unified_strings.sort:
                    self.tcfg_sort.set(d)
                    self.tk_map.append(self.tcfg_sort.tk_name)
                elif d['tk_name'].lower() == self.unified_strings.deepsort:
                    self.tcfg_deepsort.set(d)
                    self.tk_map.append(self.tcfg_deepsort.tk_name)
                else:
                    msg = f"MyConfigurator : setAllTCFG() -> Name '{d['tk_name']}' is not supported."
                    add_warning_log(msg)
            except Exception as e:
                msg = f"MyConfigurator : setAllTCFG() -> {e}"
                add_error_log(msg)
                raise ValueError(msg)

    def setAllRCFG(
        self, 
        reiders_yaml: Optional[Union[str, Dict[str, Any]]] = None, 
        relative_to_pyppbox_root: Optional[bool] = None
    ):
        """Load and set the configurations for all reiders.

        Parameters
        ----------
        reiders_yaml : str or dict
            Defaults to ``None``.
            A YAML/JSON file path, or a raw/ready dictionary of the reiders' configurations or 
            reiders.yaml. This ``reiders_yaml`` helps override the original one configured
            during the ``__init__()``.
            Leave it as default :code:`reiders_yaml=None`, to load and set according to the 
            configurations inside config directory :attr:`~pyppbox.config.configtools.PYPPBOXStructure.cfg_dir` set in the ``__init__()``.
        relative_to_pyppbox_root : bool
            Defaults to ``None``.
            An indication of whether the paths inside your configuration file are relative 
            to :code:`{pyppbox root}` or not. This indication or behavior was automatically 
            decided during the ``__init__()``, and you don't need to set or change it
            unless you set ``reiders_yaml`` to override the original configurations. If
            all the paths inside your configuration file have full absolute paths, setting 
            :obj:`relative_to_pyppbox_root` is optional.
        """
        if isinstance(relative_to_pyppbox_root, bool):
            self.rcfg_facenet = RCFGFaceNet(relative_to_pyppbox_root)
            self.rcfg_arcface = RCFGArcFace(relative_to_pyppbox_root)
            self.rcfg_adaface = RCFGAdaFace(relative_to_pyppbox_root)
            self.rcfg_torchreid = RCFGTorchreid(relative_to_pyppbox_root)
        elif relative_to_pyppbox_root is None:
            self.rcfg_facenet = RCFGFaceNet(self.relative_to_pyppbox_root)
            self.rcfg_arcface = RCFGArcFace(self.relative_to_pyppbox_root)
            self.rcfg_adaface = RCFGAdaFace(self.relative_to_pyppbox_root)
            self.rcfg_torchreid = RCFGTorchreid(self.relative_to_pyppbox_root)
        else:
            msg = "MyConfigurator : setAllRCFG() -> 'relative_to_pyppbox_root' is not valid."
            add_error_log(msg)
            raise ValueError(msg)
        docs = []
        if reiders_yaml is None:
            docs = loadDocumentList(self.reiders_yaml)
        elif reiders_yaml is not None:
            docs = getCFGDictList(reiders_yaml)
        self.ri_map = []
        for d in docs:
            try:
                if d['ri_name'].lower() == self.unified_strings.facenet:
                    self.rcfg_facenet.set(d)
                    self.ri_map.append(self.rcfg_facenet.ri_name)
                elif d['ri_name'].lower() in (self.unified_strings.arcface, self.unified_strings.adaface):
                    config = getattr(self, 'rcfg_' + d['ri_name'].lower())
                    config.set(d)
                    self.ri_map.append(config.ri_name)
                elif d['ri_name'].lower() == self.unified_strings.torchreid:
                    self.rcfg_torchreid.set(d)
                    self.ri_map.append(self.rcfg_torchreid.ri_name)
                else:
                    msg = f"MyConfigurator : setAllRCFG() -> Name '{d['ri_name']}' is not supported."
                    add_warning_log(msg)
            except Exception as e:
                raise ValueError(f"MyConfigurator : setAllRCFG() -> {e}")
    
    def setASupportedModuleCFG(
        self, 
        input_cfg: Union[str, Dict[str, Any]], 
        relative_to_pyppbox_root: bool = False
    ):
        """Load and set configurations for a supported module. Be aware that calling 
        this method will override the original or previous configurations.

        Parameters
        ----------
        input_cfg : str or dict
            A YAML/JSON file path, or a raw/ready dictionary of the configurations 
            of a supported module.
        relative_to_pyppbox_root : bool
            Defaults to ``False``.
            An indication of whether the paths inside the given ``input_cfg`` are
            relative to :code:`{pyppbox root}` or not. 
        """
        cfg = getCFGDict(input_cfg)
        k_list = list(cfg.keys())
        if "dt_name" in k_list:
            try:
                if cfg['dt_name'].lower() == self.unified_strings.yolo_cls:
                    self.dcfg_yolocs = DCFGYOLOCLS(relative_to_pyppbox_root)
                    self.dcfg_yolocs.set(cfg)
                elif cfg['dt_name'].lower() == self.unified_strings.yolo_ult:
                    self.dcfg_yolout = DCFGYOLOULT(relative_to_pyppbox_root)
                    self.dcfg_yolout.set(cfg)
                elif self.unified_strings.getUnifiedFormat(cfg['dt_name']).lower() == self.unified_strings.rf_detr:
                    self.dcfg_rfdetr = DCFGRFDETR(relative_to_pyppbox_root)
                    self.dcfg_rfdetr.set(cfg)
                elif cfg['dt_name'].lower() == self.unified_strings.gt:
                    self.dcfg_gt = DCFGGT(relative_to_pyppbox_root)
                    self.dcfg_gt.set(cfg)
                else:
                    msg = f"MyConfigurator : setASupportedModuleCFG() -> Name '{cfg['dt_name']}' is not supported."
                    add_warning_log(msg)
            except Exception as e:
                msg = f"MyConfigurator : setASupportedModuleCFG() -> {e}"
                add_error_log(msg)
                raise ValueError(msg)
        elif "tk_name" in k_list:
            try:
                if cfg['tk_name'].lower() == self.unified_strings.centroid:
                    self.tcfg_centroid = TCFGCentroid()
                    self.tcfg_centroid.set(cfg)
                elif cfg['tk_name'].lower() == self.unified_strings.sort:
                    self.tcfg_sort = TCFGSORT()
                    self.tcfg_sort.set(cfg)
                elif cfg['tk_name'].lower() == self.unified_strings.deepsort:
                    self.tcfg_deepsort = TCFGDeepSORT(relative_to_pyppbox_root)
                    self.tcfg_deepsort.set(cfg)
                else:
                    msg = f"MyConfigurator : setASupportedModuleCFG() -> Name '{cfg['tk_name']}' is not supported."
                    add_warning_log(msg)
            except Exception as e:
                msg = f"MyConfigurator : setASupportedModuleCFG() -> {e}"
                add_error_log(msg)
                raise ValueError(msg)
        elif "ri_name" in k_list:
            try:
                if cfg['ri_name'].lower() == self.unified_strings.facenet:
                    self.rcfg_facenet = RCFGFaceNet(relative_to_pyppbox_root)
                    self.rcfg_facenet.set(cfg)
                elif cfg['ri_name'].lower() in (self.unified_strings.arcface, self.unified_strings.adaface):
                    cls = RCFGArcFace if cfg['ri_name'].lower() == self.unified_strings.arcface else RCFGAdaFace
                    config = cls(relative_to_pyppbox_root)
                    config.set(cfg)
                    setattr(self, 'rcfg_' + cfg['ri_name'].lower(), config)
                elif cfg['ri_name'].lower() == self.unified_strings.torchreid:
                    self.rcfg_torchreid = RCFGTorchreid(relative_to_pyppbox_root)
                    self.rcfg_torchreid.set(cfg)
                else:
                    msg = f"MyConfigurator : setASupportedModuleCFG() -> Name '{cfg['ri_name']}' is not supported."
                    add_warning_log(msg)
            except Exception as e:
                msg = f"MyConfigurator : setASupportedModuleCFG() -> {e}"
                add_error_log(msg)
                raise ValueError(msg)

    def addAnAbstractCFG(self, input_cfg: Union[str, Dict[str, Any]]):
        """Add the configurations for an anbstract or unsupported module as a :class:`BaseCGF` 
        object and add it to the independent config list :attr:`abstractCFGs`.

        Parameters
        ----------
        input_cfg : str or dict
            A YAML/JSON file path, or a raw/ready dictionary of the configurations of a module.
        """
        tmpCFG = BaseCGF()
        tmpCFG.loadDoc(input_cfg)
        self.abstractCFGs.append(tmpCFG)

    def dumpMainCFG(self, document: Dict[str, Any]):
        """Dump the main configurations to main.yaml.

        Parameters
        ----------
        document : Dict[str, Any]
            Main selection fields to merge into main.yaml. Existing extension
            fields are retained. A new file requires all three stage selections.
        """
        _check_pending_save(self.cfg_dir)
        document = {**getCFGDict(self.main_yaml), **getCFGDict(document)} if os.path.exists(self.main_yaml) else getCFGDict(document)
        if not document:
            raise ValueError('The main configuration must contain detector, tracker, and reider.')
        MainCFG().set(document)
        dumpDocDict(self.main_yaml, document, self.cfg_headers.mainHeader())

    def dumpDetectorCFG(self, document: Dict[str, Any]):
        """Save one detector while preserving other documents and their order.

        Parameters
        ----------
        document : dict
            Detector mapping containing ``dt_name``. Merge supplied fields into
            matching documents, or append a complete document if absent. Retain
            extension fields and save this file atomically after validation.

        Notes
        -----
        Existing YAML values for other detectors are retained. Comments are
        regenerated using the existing writer; live models are not reloaded.
        """
        self._dump_module(document, 'dt_name', self.detectors_yaml)

    def dumpTrackerCFG(self, document: Dict[str, Any]):
        """Merge one tracker mapping into trackers.yaml, retaining extension fields
        and all other documents in their existing order. Validate before saving.

        Parameters
        ----------
        document : dict
            Tracker name in ``tk_name`` and fields to update. A new tracker requires
            a complete mapping. Unknown modules retain their opaque settings.
        """
        self._dump_module(document, 'tk_name', self.trackers_yaml)

    def dumpReiderCFG(self, document: Dict[str, Any]):
        """Merge one ReID mapping into reiders.yaml, retaining extension fields
        and all other documents in their existing order. Validate before saving.

        Parameters
        ----------
        document : dict
            ReID name in ``ri_name`` and fields to update. A new reider requires
            a complete mapping. Unknown modules retain their opaque settings.
        """
        self._dump_module(document, 'ri_name', self.reiders_yaml)

    def _dump_module(self, document, key, filename):
        _check_pending_save(self.cfg_dir)
        document = getCFGDict(document)
        if key not in document or not isinstance(document[key], str) or not document[key].strip():
            raise ValueError(f'{key} must be a nonempty module name.')
        name = self.unified_strings.getUnifiedFormat(document[key]).lower()
        documents = loadDocumentList(filename) if os.path.exists(filename) else []
        matched = False
        for index, existing in enumerate(documents):
            if self.unified_strings.getUnifiedFormat(existing.get(key, '')).lower() == name:
                merged = {**existing, **document}
                self._validate_documents([merged], key)
                documents[index] = merged
                matched = True
        if not matched:
            self._validate_documents([document], key)
            documents.append(document)
        # Other modules are opaque here: editing one must not normalize or reject them.
        header = {'dt_name': self.cfg_headers.detectorHeader,
                  'tk_name': self.cfg_headers.trackerHeader,
                  'ri_name': self.cfg_headers.reiderHeader}[key]()
        dumpDocDictList(filename, documents, header)

    def _validate_documents(self, documents, key):
        classes = {'YOLO_Classic': DCFGYOLOCLS, 'YOLO_Ultralytics': DCFGYOLOULT,
                   'RF_DETR': DCFGRFDETR, 'GT': DCFGGT, 'Centroid': TCFGCentroid,
                   'SORT': TCFGSORT, 'DeepSORT': TCFGDeepSORT, 'FaceNet': RCFGFaceNet,
                   'Torchreid': RCFGTorchreid, 'ArcFace': RCFGArcFace, 'AdaFace': RCFGAdaFace}
        for document in documents:
            if not isinstance(document.get(key), str) or not document[key].strip():
                raise ValueError(f'{key} must be a nonempty module name.')
            name = self.unified_strings.getUnifiedFormat(document.get(key, ''))
            cls = classes.get(name)
            if cls is not None:
                cls().set(document)

    def dumpConfigs(self, main=None, detectors=None, trackers=None, reiders=None):
        """Save complete selected config documents as a recoverable group.

        Parameters
        ----------
        main : dict or None
            Complete main mapping. None leaves main.yaml untouched.
        detectors : list[dict] or None
            Complete detector documents. None leaves detectors.yaml untouched.
        trackers : list[dict] or None
            Complete tracker documents. None leaves trackers.yaml untouched.
        reiders : list[dict] or None
            Complete ReID documents. None leaves reiders.yaml untouched.

        Notes
        -----
        Validate and serialize everything before replacing files. On a handled
        write failure, restore the originals, including their exact bytes. If
        interrupted, stop other users of the directory and call
        :func:`~pyppbox.config.configtools.recoverConfigSave` before retrying.
        Retain recovery folders if restoration fails. This is not a transaction
        for concurrent readers/writers; use one writer and pause readers. Resolved
        symlink targets must be on the same filesystem as the config directory.
        Existing per-file dump methods keep their replacement semantics.
        """
        files = []
        if main is not None:
            main = getCFGDict(main)
            if not main:
                raise ValueError('The main configuration must contain detector, tracker, and reider.')
            MainCFG().set(main)
            files.append(('main.yaml', main, self.cfg_headers.mainHeader(), True))
        for name, documents, key, header in (
                ('detectors', detectors, 'dt_name', self.cfg_headers.detectorHeader),
                ('trackers', trackers, 'tk_name', self.cfg_headers.trackerHeader),
                ('reiders', reiders, 'ri_name', self.cfg_headers.reiderHeader)):
            if documents is not None:
                documents = getCFGDictList(documents)
                self._validate_documents(documents, key)
                files.append((name + '.yaml', documents, header(), False))
        if files:
            _dump_config_files(self.cfg_dir, files)

    def dumpAllDCFG(self, document_list: List[Dict[str, Any]]):
        """Dump the detectors' configurations to detectors.yaml.

        Parameters
        ----------
        document_list : List[Dict[str, Any]]
            A list of multiple documents of the configurations.
        """
        _check_pending_save(self.cfg_dir)
        document_list = getCFGDictList(document_list)
        self._validate_documents(document_list, 'dt_name')
        dumpDocDictList(self.detectors_yaml, document_list, self.cfg_headers.detectorHeader())

    def dumpAllTCFG(self, document_list: List[Dict[str, Any]]):
        """Dump the trackers' configurations to trackers.yaml.

        Parameters
        ----------
        document_list : List[Dict[str, Any]]
            A list of multiple documents of the configurations.
        """
        _check_pending_save(self.cfg_dir)
        document_list = getCFGDictList(document_list)
        self._validate_documents(document_list, 'tk_name')
        dumpDocDictList(self.trackers_yaml, document_list, self.cfg_headers.trackerHeader())

    def dumpAllRCFG(self, document_list: List[Dict[str, Any]]):
        """Dump the reiders' configurations to reiders.yaml.

        Parameters
        ----------
        document_list : List[Dict[str, Any]]
            A list of multiple documents of the configurations.
        """
        _check_pending_save(self.cfg_dir)
        document_list = getCFGDictList(document_list)
        self._validate_documents(document_list, 'ri_name')
        dumpDocDictList(self.reiders_yaml, document_list, self.cfg_headers.reiderHeader())

