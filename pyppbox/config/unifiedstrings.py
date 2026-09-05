# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


import yaml
from yaml.loader import SafeLoader
from pyppbox.utils.commontools import joinFPathFull, getGlobalRootDir


default_strings_yaml = joinFPathFull(getGlobalRootDir(), "config/strings/strings.yaml")

class UnifiedStrings(object):

    """
    A class used to set up unified strings of pyppbox based on the internal strings.yaml.

    Attributes
    ----------
    data : Dict[str, Any]
        Set automatically.
        Data or documents read from strings.yaml.
    none : str
        Set automatically.
        Unified string of word 'None'.
    detector : str
        Set automatically.
        Unified string of word 'Detector'.
    tracker : str
        Set automatically.
        Unified string of word 'Tracker'.
    reider : str
        Set automatically.
        Unified string of word 'ReIDer'.
    gt : str
        Set automatically.
        Unified string of words 'Ground-truth'.
    yolo_cls : str
        Set automatically.
        Unified string of words 'Yolo Classic'.
    yolo_ult : str
        Set automatically.
        Unified string of words 'Yolo Ultralytics'.
    sort : str
        Set automatically.
        Unified string of word 'SORT'.
    deepsort : str
        Set automatically.
        Unified string of word 'DeepSORT'.
    centroid : str
        Set automatically.
        Unified string of word 'Centroid'.
    facenet : str
        Set automatically.
        Unified string of word 'FaceNet'.
    torchreid : str
        Set automatically.
        Unified string of word 'Torchreid'.
    dtname_yl : str
        Set automatically.
        Unified string of words 'Detector YOLO'.
    dtname_gt : str
        Set automatically.
        Unified string of words 'Detector GT'.
    tkname_ct : str
        Set automatically.
        Unified string of words 'Tracker Centroid'.
    tkname_st : str
        Set automatically.
        Unified string of words 'Tracker SORT'.
    tkname_ds : str
        Set automatically.
        Unified string of words 'Tracker DeepSORT'.
    riname_fn : str
        Set automatically.
        Unified string of words 'ReIDer FaceNet'.
    riname_tr : str
        Set automatically.
        Unified string of words 'ReIDer Torchreid'.
    unk_did : str
        Set automatically.
        Unified string of words 'Unknown deep ID'.
    unk_fid : str
        Set automatically.
        Unified string of words 'Unknown face ID'.
    err_did : str
        Set automatically.
        Unified string of words 'Error deep ID'.
    err_fid : str
        Set automatically.
        Unified string of words 'Error face ID'.
    """

    def __init__(self, strings_yaml=default_strings_yaml):
        """Initialize by calling :meth:`load(strings_yaml=strings_yaml)`.

        Parameters
        ----------
        strings_yaml : str
            Defaults to ``'{pyppbox root}/config/strings/strings.yaml'``.
            A path of a YAML file which stores the unified strings.
        """
        self.load(strings_yaml=strings_yaml)

    def load(self, strings_yaml): 
        """Load a configuration dictionary of a single document as a dictionary from 
        a ``strings_yaml`` file and automatically pass to :meth:`set()`.

        Parameters
        ----------
        strings_yaml : str
            A path of a YAML file which stores the unified strings.
        """
        with open(strings_yaml, 'r') as str_cfg:
            self.data = yaml.load(str_cfg, Loader=SafeLoader)
        self.set(self.data)

    def set(self, data):
        """Set a configuration dictionary of a single document to all attributes.

        Parameters
        ----------
        data : Dict[str, Any]
            A configuration dictionary of a single document of the unified strings.
        """
        # module
        self.none = data['none']
        self.detector = data['detector']
        self.tracker = data['tracker']
        self.reider = data['reider']
        # detector
        self.gt = data['gt']
        self.yolo_cls = data['yolo_cls']
        self.yolo_ult = data['yolo_ult']
        # tracker
        self.sort = data['sort']
        self.deepsort = data['deepsort']
        self.centroid = data['centroid']
        # reider
        self.facenet = data['facenet']
        self.torchreid = data['torchreid']
        # internal
        self.dtname_yl = data['dtname_yl']
        self.dtname_gt = data['dtname_gt']
        self.tkname_ct = data['tkname_ct']
        self.tkname_st = data['tkname_st']
        self.tkname_ds = data['tkname_ds']
        self.riname_fn = data['riname_fn']
        self.riname_tr = data['riname_tr']
        self.unk_did = data['unk_did']
        self.unk_fid = data['unk_fid']
        self.err_did = data['err_did']
        self.err_fid = data['err_fid']

    def getUnifiedFormat(self, input_str):
        """Return a standard unified format string.

        Parameters
        ----------
        input_str : str
            An input string.
        
        Returns
        -------
        str
            A unified format string.
        """
        res = ""
        input_str = str(input_str)

        if 'yolo' in input_str.lower():
            res = input_str.title().replace("Yolo", "YOLO")
        elif self.gt.lower() == input_str.lower():
            res =  input_str.upper()
        elif self.centroid.lower() == input_str.lower():
            res = input_str.title()
        elif self.sort.lower() == input_str.lower():
            res = input_str.upper()
        elif self.deepsort.lower() == input_str.lower():
            res = input_str.title().replace("Deepsort", "DeepSORT")
        elif self.facenet.lower() == self.reider.lower():
            res= input_str.title().replace("Facenet", "FaceNet")
        elif self.torchreid.lower() == self.reider.lower():
            res = input_str.title()
        elif self.none.lower() == input_str.lower():
            res = input_str.title()
        else:
            res = input_str
        
        return res
