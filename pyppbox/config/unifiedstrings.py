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


import yaml
from yaml.loader import SafeLoader
from pyppbox.utils.commontools import joinFPathFull, getGlobalRootDir


default_strings_yaml = joinFPathFull(getGlobalRootDir(), "config/strings/strings.yaml")

class UnifiedStrings(object):

    """
    A class used to set up unified strings of pyppbox based on the internal strings.yaml.

    Attributes
    ----------
    data : dict, auto
        Data or documents read from strings.yaml.
    none : str, auto
        Unified string of word 'None'.
    detector : str, auto
        Unified string of word 'Detector'.
    tracker : str, auto
        Unified string of word 'Tracker'.
    reider : str, auto
        Unified string of word 'ReIDer'.
    gt : str, auto
        Unified string of words 'Ground-truth'.
    yolo_cls : str, auto
        Unified string of words 'Yolo Classic'.
    yolo_ult : str, auto
        Unified string of words 'Yolo Ultralytics'.
    sort : str, auto
        Unified string of word 'SORT'.
    deepsort : str, auto
        Unified string of word 'DeepSORT'.
    centroid : str, auto
        Unified string of word 'Centroid'.
    facenet : str, auto
        Unified string of word 'FaceNet'.
    torchreid : str, auto
        Unified string of word 'Torchreid'.
    dtname_yl : str, auto
        Unified string of words 'Detector YOLO'.
    dtname_gt : str, auto
        Unified string of words 'Detector GT'.
    tkname_ct : str, auto
        Unified string of words 'Tracker Centroid'.
    tkname_st : str, auto
        Unified string of words 'Tracker SORT'.
    tkname_ds : str, auto
        Unified string of words 'Tracker DeepSORT'.
    riname_fn : str, auto
        Unified string of words 'ReIDer FaceNet'.
    riname_tr : str, auto
        Unified string of words 'ReIDer Torchreid'.
    unk_did : str, auto
        Unified string of words 'Unknown deep ID'.
    unk_fid : str, auto
        Unified string of words 'Unknown face ID'.
    err_did : str, auto
        Unified string of words 'Error deep ID'.
    err_fid : str, auto
        Unified string of words 'Error face ID'.
    """

    def __init__(self, strings_yaml=default_strings_yaml):
        """Initailize by calling :meth:`load(strings_yaml=strings_yaml)`.

        Parameters
        ----------
        strings_yaml : str, default='{pyppbox root}/config/strings/strings.yaml'
            A path of a YAML file which stores the unified strings.
        """
        self.load(strings_yaml=strings_yaml)

    def load(self, strings_yaml): 
        """Load a configuration dictionary of a single document as a dictionary from 
        a :obj:`strings_yaml` file and automatically pass to :meth:`set()`.

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
        data : dict
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
