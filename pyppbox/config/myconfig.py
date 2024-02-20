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


from .unifiedstrings import UnifiedStrings
from .configtools import (PYPPBOXStructure, getCFGDict, getListCFGDoc, 
                          loadListDocument, dumpDocDict, dumpListDocDict)
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

    """
    An base CFG class used to store the necessary configurations of a module.

    Attributes
    ----------
    unified_strings : MyStrings, auto
        A :class:`MyStrings` object used to store unified strings.
    configs : dict or list[dict], default={}
        A configuration dictionary of a single document or a list of multiple documents 
        of the configurations.
    """

    def __init__(self):
        self.unified_strings = unified_strings
        self.configs = {}
    
    def loadDoc(self, input):
        """Load and set dictionary of a single document from a YAML/JSON file or string.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        self.configs = getCFGDict(input)
    
    def loadDocs(self, input):
        """Load and set a list of multiple documents from a YAML/JSON file or string.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        self.configs = getListCFGDoc(input)
    
    def dumpDoc(self, output, header=""):
        """Dump the :attr:`configs` into a YAML file with simple format.

        Parameters
        ----------
        output : str
            A path file to dump.
        header : str
            A file header descriptoin.
        """
        dumpDocDict(output_file=output, doc=self.configs, header=header)
    
    def dumpDocs(self, output, header=""):
        """Dump the :attr:`configs` into a YAML file with simple format.

        Parameters
        ----------
        output : str
            A path file to dump.
        header : str
            A file header descriptoin.
        """
        dumpListDocDict(output_file=output, doc=self.configs, header=header)


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

    def set(self, input):
        """
        Set every attribute with a unified string of "None" regardless the :obj:`input`.

        Parameters
        ----------
        input : any
            A parameter to be overwritten to a unified string of :code:`"None"`.
        """
        input = "None"
        self.dt_name = self.unified_strings.getUnifiedFormat(input)
        self.tk_name = self.unified_strings.getUnifiedFormat(input)
        self.ri_name = self.unified_strings.getUnifiedFormat(input)
        self.detector = self.unified_strings.getUnifiedFormat(input)
        self.tracker = self.unified_strings.getUnifiedFormat(input)
        self.reider = self.unified_strings.getUnifiedFormat(input)


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

    def set(self, input):
        """
        Set main configurations according to :obj:`input`.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.configs:
            try:
                self.detector = self.unified_strings.getUnifiedFormat(self.configs['detector'])
                self.tracker = self.unified_strings.getUnifiedFormat(self.configs['tracker'])
                self.reider = self.unified_strings.getUnifiedFormat(self.configs['reider'])
                self.configs = self.getDocument()
            except Exception as e:
                msg = "MainCFG : set() -> " + str(e)
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
        return main_doc


class DCFGYOLOCLS(BaseCGF):

    """
    A class used to store the necessary configurations of detector YOLO Classic which 
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
        Internal parameter, weight for calibatrating the repspoint of a :class:`Person` object. 
        Check :func:`findRepspoint()` in :py:mod:`pyppbox.persontools` for more details.
    from_dir : str
        Path of the root directory, relative to path of :attr:`model_weights`.
    """

    def __init__(self, relative_to_pyppbox_root=False):
        """Initialize the class according to the :obj:`relative_to_pyppbox_root` which defines 
        whether all the paths inside your configuration file are relative to :code:`{pyppbox root}` 
        or not. If all the paths inside your configuration file have full absolute paths, 
        setting :obj:`relative_to_pyppbox_root` is optional.

        Parameters
        ----------
        relative_to_pyppbox_root : bool, default=False
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

    def set(self, input):
        """
        Set configurations according to :obj:`input`.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.configs:
            try:
                self.dt_name = self.unified_strings.getUnifiedFormat(self.configs['dt_name'])
                self.nms = self.configs['nms']
                self.conf = self.configs['conf']
                self.class_file = getAdaptiveAbsPathFDS(self.from_dir, 
                                                        self.configs['class_file'])
                self.model_cfg_file = getAdaptiveAbsPathFDS(self.from_dir, 
                                                            self.configs['model_cfg_file'])
                self.model_weights = getAdaptiveAbsPathFDS(self.from_dir, 
                                                           self.configs['model_weights'])
                self.model_image_size = self.configs['model_image_size']
                self.model_resolution = (self.model_image_size, self.model_image_size)
                self.repspoint_calibration = self.configs['repspoint_calibration']
                self.configs = self.getDocument()
            except Exception as e:
                msg = "DCFGYOLOCLS : set() -> " + str(e)
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
        return yolocs_doc


class DCFGYOLOULT(BaseCGF):

    """
    A class used to store the necessary configurations of detector YOLO_Ultralytics.

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
    device : int
        Parameter device of YOLO_Ultralytics.
    max_det : int
        Parameter max_det of YOLO_Ultralytics.
    line_width : int
        Parameter line_width of YOLO_Ultralytics.
    model_file : str
        Path of :attr:`model_file` for YOLO_Ultralytics.
    repspoint_calibration : float
        Weight for calibatrating the repspoint of a :class:`Person` object. 
        Check :func:`findRepspoint()` in :py:mod:`pyppbox.persontools` for more details.
    from_dir : str
        Path of the root directory, relative to path of :attr:`model_file`.
    """

    def __init__(self, relative_to_pyppbox_root=False):
        """Initialize the class according to the :obj:`relative_to_pyppbox_root` which 
        defines whether all the paths inside your configuration file are relative to 
        :code:`{pyppbox root}` or not. If all the paths inside your configuration file have 
        full absolute paths, setting :obj:`relative_to_pyppbox_root` is optional.

        Parameters
        ----------
        relative_to_pyppbox_root : bool, default=False
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

    def set(self, input):
        """
        Set configurations according to :obj:`input`.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.configs:
            try:
                self.dt_name = self.unified_strings.getUnifiedFormat(self.configs['dt_name'])
                self.conf = self.configs['conf']
                self.iou = self.configs['iou']
                self.imgsz = self.configs['imgsz']
                self.show_boxes = self.configs['show_boxes']
                self.device = self.configs['device']
                self.max_det = self.configs['max_det']
                self.line_width = self.configs['line_width']
                self.model_file = getAdaptiveAbsPathFDS(self.from_dir, 
                                                        self.configs['model_file'])
                self.repspoint_calibration = self.configs['repspoint_calibration']
                self.configs = self.getDocument()
            except Exception as e:
                msg = "DCFGYOLOULT : set() -> " + str(e)
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
            "line_width": self.line_width,
            "model_file": normalizePathFDS(internal_root_dir, self.model_file),
            "repspoint_calibration": self.repspoint_calibration
        }
        return yolout_doc


class DCFGGT(BaseCGF):

    """
    A class used to store the necessary configurations of detector GT (Ground-truth).

    Attributes
    ----------
    dt_name : str
        Configured name of detector GT (Ground-truth).
    gt_file : str
        Path of GT text file.
    gt_map_file : str
        Path of GT map text file.
    from_dir : str
        Path of the root directory, relative to path of :obj:`model_file`.
    """

    def __init__(self, relative_to_pyppbox_root=False):
        """Initialize the class according to the :obj:`relative_to_pyppbox_root` which defines 
        whether all the paths inside your configuration file are relative to :obj:`{pyppbox root}`
        or not. If all the paths inside your configuration file have full absolute paths, 
        setting :obj:`relative_to_pyppbox_root` is optional.

        Parameters
        ----------
        relative_to_pyppbox_root : bool, default=False
            (1) Set :code:`relative_to_pyppbox_root=False` to use your current working directory 
            as where all the paths in your configuration file are relative to; for example, 
            the path of :attr:`gt_file` inside your configuration file is set relatively to your 
            current working directory. 
            (2) Set :code:`relative_to_pyppbox_root=True` when all the paths in your configuration 
            file are relative to :code:`{pyppbox root}` like all the default paths inside in pyppbox's 
            internal configuration files are orginally set relatively to :code:{pyppbox root}
        """
        super().__init__()
        self.from_dir = ""
        if relative_to_pyppbox_root:
            self.from_dir = internal_root_dir

    def set(self, input):
        """
        Set configurations according to :obj:`input`.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.configs:
            try:
                self.dt_name = self.unified_strings.getUnifiedFormat(self.configs['dt_name'])
                self.gt_file = getAdaptiveAbsPathFDS(self.from_dir, self.configs['gt_file'])
                self.gt_map_file = getAdaptiveAbsPathFDS(self.from_dir, self.configs['gt_map_file'])
                self.configs = self.getDocument()
            except Exception as e:
                msg = "DCFGGT : set() -> " + str(e)
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
        return gt_doc


class TCFGCentroid(BaseCGF):

    """
    A class used to store the necessary configurations of tracker Centroid.

    Attributes
    ----------
    tk_name : str
        Configured name of tracker Centroid.
    max_spread : int
        Maximum distance of the being tracked :class:`Person` object of previous and current state.
    """

    def set(self, input):
        """
        Set configurations according to :obj:`input`.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.configs:
            try:
                self.tk_name = self.unified_strings.getUnifiedFormat(self.configs['tk_name'])
                self.max_spread = self.configs['max_spread']
                self.configs = self.getDocument()
            except Exception as e:
                msg = "TCFGCentroid : set() -> " + str(e)
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
        return centroid_doc


class TCFGSORT(BaseCGF):

    """
    A class used to store the necessary configurations of tracker SORT.

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
    """

    def set(self, input):
        """
        Set configurations according to :obj:`input`.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.configs:
            try:
                self.tk_name = self.unified_strings.getUnifiedFormat(self.configs['tk_name'])
                self.max_age = self.configs['max_age']
                self.min_hits = self.configs['min_hits']
                self.iou_threshold = self.configs['iou_threshold']
                self.configs = self.getDocument()
            except Exception as e:
                msg = "TCFGSORT : set() -> " + str(e)
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
        return sort_doc
        

class TCFGDeepSORT(BaseCGF):

    """
    A class used to store the necessary configurations of tracker DeepSORT.

    Attributes
    ----------
    tk_name : str
        Configured name of tracker DeepSORT.
    nn_budget : int
        Parameter :obj:`nn_budget` of tracker DeepSORT.
    nms_max_overlap : float
        Parameter :obj:`nms_max_overlap` of tracker DeepSORT.
    max_cosine_distance : float
        Parameter :obj:`max_cosine_distance` of tracker DeepSORT.
    model_file : str
        Path of model file for tracker DeepSORT.
    from_dir : str
        Path of the root directory, relative to path of :attr:`model_file`.
    """

    def __init__(self, relative_to_pyppbox_root=False):
        """Initialize the class according to the :obj:`relative_to_pyppbox_root` which defines 
        whether all the paths inside your configuration file are relative to :code:`{pyppbox root}` 
        or not. If all the paths inside your configuration file have full absolute paths, setting 
        :obj:`relative_to_pyppbox_root` is optional.

        Parameters
        ----------
        relative_to_pyppbox_root : bool, default=False
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

    def set(self, input):
        """
        Set configurations according to :obj:`input`.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.configs:
            try:
                self.tk_name = self.unified_strings.getUnifiedFormat(self.configs['tk_name'])
                self.nn_budget = self.configs['nn_budget']
                self.nms_max_overlap = self.configs['nms_max_overlap']
                self.max_cosine_distance = self.configs['max_cosine_distance']
                self.model_file = getAdaptiveAbsPathFDS(self.from_dir, self.configs['model_file'])
                self.configs = self.getDocument()
            except Exception as e:
                msg = "TCFGDeepSORT : set() -> " + str(e)
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
        return deepsort_doc


class RCFGFaceNet(BaseCGF):

    """
    A class used to store the necessary configurations of reider FaceNet.

    Attributes
    ----------
    unified_strings : MyStrings, auto
        A :class:`MyStrings` object used to store unified strings.
    ri_name : str
        Configured name of reider FaceNet.
    gpu_mem : float
        Limit GPU memory usage.
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
        Mininum confidence of the prediction.
    yl_h_calibration : list[int, int], default=[-125, 75]
        When YOLO is used as the detector, this list of :code:`[val_1, val_2]` and a :class:`Person`'s 
        respoint :code:`(X, Y)` are used to find the from-to :code:`Y` for cropping the face: 
        :code:`[Y + val_1 : Y + val_2, ...]`.
    yl_w_calibration : list[int, int], default=[-55, 55]
        When YOLO is used as the detector, this list of :code:`[val_1, val_2]` and a :class:`Person`'s 
        respoint :code:`(X, Y)` are used to find the from-to :code:`X` for cropping the face: 
        :code:`[..., X + val_1 : X + val_2]`.
    from_dir : str
        Path of the root directory, relative to path of :attr:`model_file`.
    """

    def __init__(self, relative_to_pyppbox_root=False):
        """Initialize the class according to the :obj:`relative_to_pyppbox_root` which defines 
        whether all the paths inside your configuration file are relative to :code:`{pyppbox root}` 
        or not. If all the paths inside your configuration file have full absolute paths, 
        setting :obj:`relative_to_pyppbox_root` is optional.

        Parameters
        ----------
        relative_to_pyppbox_root : bool, default=False
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

    def set(self, input):
        """
        Set configurations according to :obj:`input`.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.configs:
            try:
                self.ri_name = self.unified_strings.getUnifiedFormat(self.configs['ri_name'])
                self.gpu_mem = self.configs['gpu_mem']
                self.model_det = getAdaptiveAbsPathFDS(self.from_dir, self.configs['model_det'])
                self.model_file = getAdaptiveAbsPathFDS(self.from_dir, self.configs['model_file'])
                self.classifier_pkl = getAdaptiveAbsPathFDS(self.from_dir, self.configs['classifier_pkl'])
                self.train_data = getAdaptiveAbsPathFDS(self.from_dir, self.configs['train_data'])
                self.batch_size = self.configs['batch_size']
                self.min_confidence = self.configs['min_confidence']
                self.yl_h_calibration = self.configs['yl_h_calibration']
                self.yl_w_calibration = self.configs['yl_w_calibration']
                self.configs = self.getDocument()
            except Exception as e:
                msg = "RCFGFaceNet : set() -> " + str(e)
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
        return facenet_doc


class RCFGTorchreid(BaseCGF):

    """
    A class used to store the necessary configurations of reider Torchreid.

    Attributes
    ----------
    unified_strings : MyStrings, auto
        A :class:`MyStrings` object used to store unified strings.
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
        Mininum confidence of the prediction.
    device : str
        Parameter device for specifying a computing device.
    base_model_path : str
        Path of a base model corresponding to the pretrained model or :attr:`model_name`. 
    model_dict : TorchreidModelDict, auto
        A :class:`TorchreidModelDict` object used to store the dictionary of a Torchreid model.
    model_wh : tuple(int, int), auto
        A tuple used to store the input image size :code:`(width, height)` for a corresponding 
        Torchreid model.
    from_dir : str
        Path of the root directory, relative to path of :attr:`model_path`.
    """

    def __init__(self, relative_to_pyppbox_root=False):
        """Initialize the class according to the :obj:`relative_to_pyppbox_root` which defines 
        whether all the paths inside your configuration file are relative to :code:`{pyppbox root}` 
        or not. If all the paths inside your configuration file have full absolute paths, 
        setting :obj:`relative_to_pyppbox_root` is optional.

        Parameters
        ----------
        relative_to_pyppbox_root : bool, default=False
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

    def set(self, input):
        """
        Set configurations according to :obj:`input`.

        Parameters
        ----------
        input : str or dict
            A YAML/JSON file path, or a raw/ready dictionary.
        """
        super().loadDoc(input)
        if self.configs:
            try:
                from pyppbox.modules.reiders.torchreid.model_dict import TorchreidModelDict
                self.ri_name = self.unified_strings.getUnifiedFormat(self.configs['ri_name'])
                self.classifier_pkl = getAdaptiveAbsPathFDS(self.from_dir, 
                                                            self.configs['classifier_pkl'])
                self.train_data = getAdaptiveAbsPathFDS(self.from_dir, 
                                                        self.configs['train_data'])
                self.model_name = self.configs['model_name']
                self.model_path = getAdaptiveAbsPathFDS(self.from_dir, 
                                                        self.configs['model_path'])
                self.min_confidence = self.configs['min_confidence']
                self.device = self.configs['device']
                self.configs = self.getDocument()
                self.base_model_path = getAdaptiveAbsPathFDS(
                    self.from_dir, 
                    joinFPathFull(getAncestorDir(self.model_path, 1), 'base')
                )
                self.model_dict = TorchreidModelDict()
                self.model_wh = self.model_dict.getWH(getFileName(self.model_path))
            except Exception as e:
                msg = "RCFGTorchreid : set() -> " + str(e)
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
        return torchreid_doc


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
                "# detector: None | YOLO_Classic | YOLO_Ultralytics | GT\n"
                "# tracker: None | Centroid | SORT | DeepSORT\n"
                "# reider: None | FaceNet | Torchreid\n"
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
                "# nms: 0.45\n"
                "# conf: 0.5\n"
                "# class_file: data/modules/yolo_classic/coco.names\n"
                "# model_cfg_file: data/modules/yolo_classic/yolov4.cfg\n"
                "# model_weights: data/modules/yolo_classic/yolov4.weights\n"
                "# model_image_size: 416\n"
                "# repspoint_calibration: 0.25\n"
                "###########################################################\n"
                "# --- # YOLO_Ultralytics\n"
                "# dt_name: YOLO_Ultralytics\n"
                "# conf: 0.5\n"
                "# iou: 0.7\n"
                "# imgsz: 416\n"
                "# show_boxes: True\n"
                "# device: 0\n"
                "# max_det: 50\n"
                "# line_width: 500\n"
                "# model_file: data/modules/yolo_ultralytics/yolov8l-pose.pt\n"
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
        header=("::    pyppbox: Toolbox for people detecting, tracking, and re-identifying.\n"
                "::    Copyright (C) 2022 UMONS-Numediart\n"
                "::\n"
                "::    This program is free software: you can redistribute it and/or modify\n"
                "::    it under the terms of the GNU General Public License as published by\n"
                "::    the Free Software Foundation, either version 3 of the License, or\n"
                "::    (at your option) any later version.\n"
                "::\n"
                "::    This program is distributed in the hope that it will be useful,\n"
                "::    but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
                "::    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
                "::    GNU General Public License for more details.\n"
                "::\n"
                "::    You should have received a copy of the GNU General Public License\n"
                "::    along with this program.  If not, see <https://www.gnu.org/licenses/>.\n\n\n")
        return header


class MyConfigurator(PYPPBOXStructure):

    """
    A class used to store and manage all the configurations of pyppbox.

    Attributes
    ----------
    mcfg : MainCFG, auto
        A :class:`MainCFG` object used to identify the main detector/tracker/reider.
    dcfg_yolocs : DCFGYOLOCLS, auto
        A :class:`DCFGYOLOCLS` object used to store the configurations of detector 
        YOLO Classic.
    dcfg_yolout : DCFGYOLOULT, auto
        A :class:`DCFGYOLOULT` object used to store the configurations of detector 
        YOLO_Ultralytics.
    dcfg_gt : DCFGGT, auto
        A :class:`DCFGGT` object used to store the configurations of detector GT (Ground-truth).
    tcfg_centroid : TCFGCentroid, auto
        A :class:`TCFGCentroid` object used to store the configurations of tracker Centroid.
    tcfg_sort : TCFGSORT, auto
        A :class:`TCFGSORT` object used to store the configurations of tracker SORT.
    tcfg_deepsort : TCFGDeepSORT, auto
        A :class:`TCFGDeepSORT` object used to store the configurations of tracker DeepSORT.
    rcfg_facenet : RCFGFaceNet, auto
        A :class:`RCFGFaceNet` object used to store the configurations of reider FaceNet.
    rcfg_torchreid : RCFGTorchreid, auto
        A :class:`RCFGTorchreid` object used to store the configurations of reider Torchreid.
    dt_map : list[str, ...], auto
        A list used to store all detectors' names loaded from the configuration file.
    tk_map : list[str, ...], auto
        A list used to store all trackers' names loaded from the configuration file.
    ri_map : list[str, ...], auto
        A list used to store all reiders' names loaded from the configuration file.
    cfg_headers : MyCFGHeaders
        A :class:`MyCFGHeaders` object used to store the headers of configuration.
    relative_to_pyppbox_root: bool, auto
        An indication of whether the paths inside all configuration files are relative 
        to :code:`{pyppbox root}` or not. 
    abstractCFGs: list
        A list used to store abstract or unsupported :class:`BaseCGF` objects.
    """

    def __init__(self, cfg_dir=internal_cfg_dir, set_all_modules=False):
        """Initailize according to the given directory of the YAML configurations.

        Parameters
        ----------
        cfg_dir : str, default='{pyppbox root}/config/cfg'
            A path of the config directory where stores main.yaml, detectors.yaml, 
            trackers.yaml, and reiders.yaml.
        set_all_modules : bool, default=False
            An idication of whether to load and set all configurations of all supported 
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

    def setCustomCFG(self, cfg_dir):
        super().setCustomCFG(cfg_dir=cfg_dir)
        self.relative_to_pyppbox_root = False
        if cfg_dir == internal_cfg_dir:
            self.relative_to_pyppbox_root = True

    def setMCFG(self, main_yaml=None):
        """Load and set the main configurations which used to identify the main 
        detector/tracker/reider.

        Parameters
        ----------
        main_yaml : str or dict, default=None
            A YAML/JSON file path, or a raw/ready dictionary of the main configurations 
            or main.yaml. This :obj:`main_yaml` helps overwrite the original one configured 
            during the :meth:`__init__()`.
            Leave it as default :code:`main_yaml=None`, to load and set according to the 
            configurations inside config directory :attr:`cfg_dir` set in the :meth:`__init__()`.
        """
        self.mcfg = MainCFG()
        if main_yaml is None:
            self.mcfg.set(self.main_yaml)
        else:
            self.mcfg.set(main_yaml)
    
    def setMainModules(self, main_yaml=None):
        """Call :meth:`setMCFG()` and then load and set the main detector/tracker/reider accordingly.

        Parameters
        ----------
        main_yaml : str or dict, default=None
            A YAML/JSON file path, or a raw/ready dictionary of the main configurations 
            or main.yaml. This :obj:`main_yaml` helps overwrite the original one configured 
            during the :meth:`__init__()`.
            Leave it as default :code:`main_yaml=None`, to load and set according to the 
            configurations inside config directory :attr:`cfg_dir` set in the :meth:`__init__()`.
        """
        self.setMCFG(main_yaml=main_yaml)
        selected_dt = self.mcfg.configs['detector']
        dt_list = loadListDocument(self.detectors_yaml)
        for dt in dt_list:
            if dt['dt_name'].lower() == selected_dt.lower():
                self.setASupportedModuleCFG(dt, self.relative_to_pyppbox_root)
                break
        selected_tk = self.mcfg.configs['tracker']
        tk_list = loadListDocument(self.trackers_yaml)
        for tk in tk_list:
            if tk['tk_name'].lower() == selected_tk.lower():
                self.setASupportedModuleCFG(tk, self.relative_to_pyppbox_root)
                break
        selected_ri = self.mcfg.configs['reider']
        ri_list = loadListDocument(self.reiders_yaml)
        for ri in ri_list:
            if ri['ri_name'].lower() == selected_ri.lower():
                self.setASupportedModuleCFG(ri, self.relative_to_pyppbox_root)
                break
    
    def setGTCFG(self, detectors_yaml=None, relative_to_pyppbox_root=None):
        """Manually load and set the configurations for :attr:`dcfg_gt`.

        Parameters
        ----------
        detectors_yaml : str or dict, default=None
            A YAML/JSON file path, or a raw/ready dictionary of the detectors' configurations 
            or detectors.yaml. This :obj:`detectors_yaml` helps overwrite the original one 
            configured during the :meth:`__init__()`. 
            Leave it as default :code:`detectors_yaml=None`, to load and set according to the 
            configurations inside config directory :attr:`cfg_dir` set in the :meth:`__init__()`.
        relative_to_pyppbox_root : bool, default=None
            An indication of whether the paths inside your configuration file are relative 
            to :code:`{pyppbox root}` or not. This indication or behavior was automatically 
            decided during the :meth:`__init__()`, and you don't need to set or change it 
            unless you set :obj:`detectors_yaml` to overwrite the original configurations. If 
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
            docs = loadListDocument(self.detectors_yaml)
        elif detectors_yaml is not None:
            docs = getListCFGDoc(detectors_yaml)
        for d in docs:
            try:
                if d['dt_name'].lower() == self.unified_strings.gt:
                    self.dcfg_gt.set(d)
            except Exception as e:
                msg = "MyConfigurator : setGTCFG() -> " + str(e)
                add_error_log(msg)
                raise ValueError(msg)

    def setAllDCFG(self, detectors_yaml=None, relative_to_pyppbox_root=None):
        """Load and set the configurations for all supported detectors.

        Parameters
        ----------
        detectors_yaml : str or dict, default=None
            A YAML/JSON file path, or a raw/ready dictionary of the detectors' configurations 
            or detectors.yaml. This :obj:`detectors_yaml` helps overwrite the original one 
            configured during the :meth:`__init__()`. 
            Leave it as default :code:`detectors_yaml=None`, to load and set according to the 
            configurations inside config directory :attr:`cfg_dir` set in the :meth:`__init__()`.
        relative_to_pyppbox_root : bool, default=None
            An indication of whether the paths inside your configuration file are relative 
            to :code:`{pyppbox root}` or not. This indication or behavior was automatically 
            decided during the :meth:`__init__()`, and you don't need to set or change it 
            unless you set :obj:`detectors_yaml` to overwrite the original configurations. If 
            all the paths inside your configuration file have full absolute paths, setting 
            :obj:`relative_to_pyppbox_root` is optional.
        """
        if isinstance(relative_to_pyppbox_root, bool):
            self.dcfg_yolocs = DCFGYOLOCLS(relative_to_pyppbox_root)
            self.dcfg_yolout = DCFGYOLOULT(relative_to_pyppbox_root)
            self.dcfg_gt = DCFGGT(relative_to_pyppbox_root)
        elif relative_to_pyppbox_root is None:
            self.dcfg_yolocs = DCFGYOLOCLS(self.relative_to_pyppbox_root)
            self.dcfg_yolout = DCFGYOLOULT(self.relative_to_pyppbox_root)
            self.dcfg_gt = DCFGGT(self.relative_to_pyppbox_root)
        else:
            msg = "MyConfigurator : setAllDCFG() -> 'relative_to_pyppbox_root' is not valid."
            add_error_log(msg)
            raise ValueError(msg)
        docs = []
        if detectors_yaml is None:
            docs = loadListDocument(self.detectors_yaml)
        elif detectors_yaml is not None:
            docs = getListCFGDoc(detectors_yaml)
        self.dt_map = []
        for d in docs:
            try:
                if d['dt_name'].lower() == self.unified_strings.yolo_cls:
                    self.dcfg_yolocs.set(d)
                    self.dt_map.append(self.dcfg_yolocs.dt_name)
                elif d['dt_name'].lower() == self.unified_strings.yolo_ult:
                    self.dcfg_yolout.set(d)
                    self.dt_map.append(self.dcfg_yolout.dt_name)
                elif d['dt_name'].lower() == self.unified_strings.gt:
                    self.dcfg_gt.set(d)
                    self.dt_map.append(self.dcfg_gt.dt_name)
                else:
                    msg = ("MyConfigurator : setAllDCFG() -> Name '" + 
                           str(d['dt_name']) + "' is not supported.")
                    add_warning_log(msg)
            except Exception as e:
                msg = "MyConfigurator : setAllDCFG() -> " + str(e)
                add_error_log(msg)
                raise ValueError(msg)

    def setAllTCFG(self, trackers_yaml=None, relative_to_pyppbox_root=None):
        """Load and set the configurations for all supported trackers.

        Parameters
        ----------
        trackers_yaml : str or dict, default=None
            A YAML/JSON file path, or a raw/ready dictionary of the trackers' configurations 
            or trackers.yaml. This :obj:`trackers_yaml` helps overwrite the original one 
            configured during the :meth:`__init__()`. 
            Leave it as default :code:`trackers_yaml=None`, to load and set according to the 
            configurations inside config directory :attr:`cfg_dir` set in the :meth:`__init__()`.
        relative_to_pyppbox_root : bool, default=None
            An indication of whether the paths inside your configuration file are relative 
            to :code:`{pyppbox root}` or not. This indication or behavior was automatically 
            decided during the :meth:`__init__()`, and you don't need to set or change it 
            unless you set :obj:`trackers_yaml` to overwrite the original configurations. If 
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
            docs = loadListDocument(self.trackers_yaml)
        elif trackers_yaml is not None:
            docs = getListCFGDoc(trackers_yaml)
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
                    msg = ("MyConfigurator : setAllTCFG() -> Name '" + 
                           str(d['dt_name']) + "' is not supported.")
                    add_warning_log(msg)
            except Exception as e:
                msg = "MyConfigurator : setAllTCFG() -> "  + str(e)
                add_error_log(msg)
                raise ValueError(msg)

    def setAllRCFG(self, reiders_yaml=None, relative_to_pyppbox_root=None):
        """Load and set the configurations for all reiders.

        Parameters
        ----------
        reiders_yaml : str or dict, default=None
            A YAML/JSON file path, or a raw/ready dictionary of the reiders' configurations or 
            reiders.yaml. This :obj:`reiders_yaml` helps overwrite the original one configured 
            during the :meth:`__init__()`. 
            Leave it as default :code:`reiders_yaml=None`, to load and set according to the 
            configurations inside config directory :attr:`cfg_dir` set in the :meth:`__init__()`.
        relative_to_pyppbox_root : bool, default=None
            An indication of whether the paths inside your configuration file are relative 
            to :code:`{pyppbox root}` or not. This indication or behavior was automatically 
            decided during the :meth:`__init__()`, and you don't need to set or change it 
            unless you set :obj:`reiders_yaml` to overwrite the original configurations. If 
            all the paths inside your configuration file have full absolute paths, setting 
            :obj:`relative_to_pyppbox_root` is optional.
        """
        if isinstance(relative_to_pyppbox_root, bool):
            self.rcfg_facenet = RCFGFaceNet(relative_to_pyppbox_root)
            self.rcfg_torchreid = RCFGTorchreid(relative_to_pyppbox_root)
        elif relative_to_pyppbox_root is None:
            self.rcfg_facenet = RCFGFaceNet(self.relative_to_pyppbox_root)
            self.rcfg_torchreid = RCFGTorchreid(self.relative_to_pyppbox_root)
        else:
            msg = "MyConfigurator : setAllRCFG() -> 'relative_to_pyppbox_root' is not valid."
            add_error_log(msg)
            raise ValueError(msg)
        docs = []
        if reiders_yaml is None:
            docs = loadListDocument(self.reiders_yaml)
        elif reiders_yaml is not None:
            docs = getListCFGDoc(reiders_yaml)
        self.ri_map = []
        for d in docs:
            try:
                if d['ri_name'].lower() == self.unified_strings.facenet:
                    self.rcfg_facenet.set(d)
                    self.ri_map.append(self.rcfg_facenet.ri_name)
                elif d['ri_name'].lower() == self.unified_strings.torchreid:
                    self.rcfg_torchreid.set(d)
                    self.ri_map.append(self.rcfg_torchreid.ri_name)
                else:
                    msg = ("MyConfigurator : setAllRCFG() -> Name '" + 
                           str(d['dt_name']) + "' is not supported.")
                    add_warning_log(msg)
            except Exception as e:
                raise ValueError("MyConfigurator : setAllRCFG() -> "  + str(e))
    
    def setASupportedModuleCFG(self, input_cfg, relative_to_pyppbox_root=False):
        """Load and set configurations for a supported module. Be aware that calling 
        this method will overwrite the original or previous configurations.

        Parameters
        ----------
        input_cfg: str or dict
            A YAML/JSON file path, or a raw/ready dictionary of the configurations 
            of a supported module.
        relative_to_pyppbox_root : bool, defualt=False
            An indication of whether the paths inside the given :obj:`input_cfg` are 
            relative to :code:`{pyppbox root}` or not. 
        """
        cfg = getCFGDict(input_cfg)
        k_list = list(input_cfg.keys())
        if "dt_name" in k_list:
            try:
                if cfg['dt_name'].lower() == self.unified_strings.yolo_cls:
                    self.dcfg_yolocs = DCFGYOLOCLS(relative_to_pyppbox_root)
                    self.dcfg_yolocs.set(cfg)
                elif cfg['dt_name'].lower() == self.unified_strings.yolo_ult:
                    self.dcfg_yolout = DCFGYOLOULT(relative_to_pyppbox_root)
                    self.dcfg_yolout.set(cfg)
                elif cfg['dt_name'].lower() == self.unified_strings.gt:
                    self.dcfg_gt = DCFGGT(relative_to_pyppbox_root)
                    self.dcfg_gt.set(cfg)
                else:
                    msg = ("MyConfigurator : setASupportedModuleCFG() -> Name '" + 
                           str(cfg['dt_name']) + "' is not supported.")
                    add_warning_log(msg)
            except Exception as e:
                msg = "MyConfigurator : setASupportedModuleCFG() -> " + str(e)
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
                    msg = ("MyConfigurator : setASupportedModuleCFG() -> Name '" + 
                           str(cfg['tk_name']) + "' is not supported.")
                    add_warning_log(msg)
            except Exception as e:
                msg = "MyConfigurator : setASupportedModuleCFG() -> " + str(e)
                add_error_log(msg)
                raise ValueError(msg)
        elif "ri_name" in k_list:
            try:
                if cfg['ri_name'].lower() == self.unified_strings.facenet:
                    self.rcfg_facenet = RCFGFaceNet(relative_to_pyppbox_root)
                    self.rcfg_facenet.set(cfg)
                elif cfg['ri_name'].lower() == self.unified_strings.torchreid:
                    self.rcfg_torchreid = RCFGTorchreid(relative_to_pyppbox_root)
                    self.rcfg_torchreid.set(cfg)
                else:
                    msg = ("MyConfigurator : setASupportedModuleCFG() -> Name '" + 
                           str(cfg['ri_name']) + "' is not supported.")
                    add_warning_log(msg)
            except Exception as e:
                msg = "MyConfigurator : setASupportedModuleCFG() -> " + str(e)
                add_error_log(msg)
                raise ValueError(msg)

    def addAnAbstractCFG(self, input_cfg):
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

    def dumpMainCFG(self, document):
        """Dump the main configurations to main.yaml.

        Parameters
        ----------
        document : dict
            A configuration dictionary of a single document of the configurations.
        """
        dumpDocDict(self.main_yaml, document, self.cfg_headers.mainHeader())

    def dumpAllDCFG(self, document_list):
        """Dump the detectors' configurations to detectors.yaml.

        Parameters
        ----------
        document_list : list[dict]
            A list of multiple documents of the configurations.
        """
        dumpListDocDict(self.detectors_yaml, document_list, self.cfg_headers.detectorHeader())

    def dumpAllTCFG(self, document_list):
        """Dump the trackers' configurations to trackers.yaml.

        Parameters
        ----------
        document_list : list[dict]
            A list of multiple documents of the configurations.
        """
        dumpListDocDict(self.trackers_yaml, document_list, self.cfg_headers.trackerHeader())

    def dumpAllRCFG(self, document_list):
        """Dump the reiders' configurations to reiders.yaml.

        Parameters
        ----------
        document_list : list[dict]
            A list of multiple documents of the configurations.
        """
        dumpListDocDict(self.reiders_yaml, document_list, self.cfg_headers.reiderHeader())

