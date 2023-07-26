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


import json
import yaml
from yaml.loader import SafeLoader

from .unifiedstrings import UnifiedStrings
from pyppbox.utils.commontools import (joinFPathFull, getAbsPathFDS, 
                                       getGlobalRootDir, isExist)
from pyppbox.utils.logtools import add_warning_log, add_error_log


internal_root_dir = getGlobalRootDir()
internal_config_dir = joinFPathFull(internal_root_dir, 'config')
internal_cfg_dir = joinFPathFull(internal_config_dir, 'cfg')

class PYPPBOXStructure(object):

    """
    A class used to organize the structure of pyppbox.

    Attributes
    ----------
    cfg_dir : str, default='{pyppbox root}/config/cfg'
        Path of config directory where stores main.yaml, detectors.yaml, 
        trackers.yaml, reiders.yaml.
    internal_root_dir : str, auto
        Path of pyppbox's root directory.
    gui_root : str, auto
        Path of GUI's root directory.
    gui_tmp_dir : str, auto
        Path of GUI's tmp directory.
    data_dir : str, auto
        Internal path of pyppbox's data directory.
    dataset_dir : str, auto
        Path of a supported dataset like 
        :code:`{pyppbox root}/data/datasets/GTA_V_DATASET`, etc.
    gt_dir : str, auto
        Path of a dataset's GT (Ground-truth) directory where stores all 
        the ground-truth text files and the mapping text file.
    main_yaml : str, auto
        Path of main.yaml.
    detectors_yaml : str, auto
        Path of detectors.yaml.
    trackers_yaml : str, auto
        Path of trackers.yaml.
    reiders_yaml : str, auto
        Path of reiders.yaml.
    unified_strings : MyStrings, auto
        A :class:`MyStrings` object used to store unified strings.
    """

    def __init__(self, cfg_dir=internal_cfg_dir):
        """Initailize according to the input :obj:`cfg_dir` and automatically call 
        :meth:`setDIR()`, :meth:`setYAMLPath()`, :meth:`setSTR()`, and :meth:`setVP()`.

        Parameters
        ----------
        cfg_dir : str, default='{pyppbox root}/config/cfg'
            A path of the config directory where stores main.yaml, detectors.yaml, 
            trackers.yaml, and reiders.yaml.
        """
        if cfg_dir != internal_cfg_dir:
            if not isExist(cfg_dir):
                add_warning_log("PYPPBOXStructure : __init__() -> cfg_dir='" + 
                                str(cfg_dir) + "' does not exist.")
                add_warning_log("PYPPBOXStructure : __init__() -> " + 
                                "Switched to internal cfg directory !")
                cfg_dir = internal_cfg_dir
        self.cfg_dir = cfg_dir
        self.internal_root_dir = internal_root_dir
        self.unified_strings = UnifiedStrings()
        self.setDIR()
        self.setYAMLPath()

    def setDIR(self):
        """Automatically config or set all necessary directories of the base structure 
        of pyppbox such as :attr:`gui_root`, :attr:`gui_tmp_dir`, :attr:`data_dir`, 
        :attr:`dataset_dir`, and :attr:`gt_dir`.
        """
        # GUI 
        self.gui_root = joinFPathFull(self.internal_root_dir, 'gui')
        self.gui_tmp_dir = joinFPathFull(self.gui_root, 'tmp')
        # Dataset
        self.data_dir = joinFPathFull(self.internal_root_dir, 'data')
        self.dataset_dir = joinFPathFull(self.data_dir, 'datasets/GTA_V_DATASET')
        self.gt_dir = joinFPathFull(self.dataset_dir, 'ground_truth')

    def setYAMLPath(self):
        """
        Set the paths of the necessary YAML files such as :attr:`main_yaml`, 
        :attr:`detectors_yaml`, :attr:`trackers_yaml`, and :attr:`reiders_yaml`.
        """
        self.main_yaml = joinFPathFull(self.cfg_dir, "main.yaml")
        self.detectors_yaml = joinFPathFull(self.cfg_dir, "detectors.yaml")
        self.trackers_yaml = joinFPathFull(self.cfg_dir, "trackers.yaml")
        self.reiders_yaml = joinFPathFull(self.cfg_dir, "reiders.yaml")
    
    def setCustomCFG(self, cfg_dir):
        """Set a custom path of a config directory where stores main.yaml, 
        detectors.yaml, trackers.yaml, and reiders.yaml.

        Parameters
        ----------
        cfg_dir : str
            A path of the config directory.
        """
        if not isExist(cfg_dir):
            add_warning_log("PYPPBOXStructure : setCustomCFG() -> cfg_dir='" 
                            + str(cfg_dir) + "' does not exist.")
            add_warning_log("PYPPBOXStructure : setCustomCFG() -> " + 
                            "Switched to internal cfg directory !")
            cfg_dir = internal_cfg_dir
        self.cfg_dir = cfg_dir
        self.setYAMLPath()


#########################################################################################


def isDictString(input_string):
    """Check whether the :obj:`input_string` is a valid raw dictionary.

    Parameters
    ----------
    input_string : str
        An input of raw string.

    Returns
    -------
    bool
        Validation status.
    """
    res = True
    if (
        isinstance(input_string, str) and 
        len(input_string) > 4 and
        input_string[0] == '[' and
        input_string[1] == '{' and  
        input_string[-1] == ']' and 
        input_string[-2] == '}'
    ):
        try:
            yaml.load(input_string, Loader=SafeLoader)
        except ValueError as e:
            res = False
    else:
        res = False
    return res

def getCFGDict(input):
    """Get a configuration dictionary of a single document from the given :obj:`input`.

    Parameters
    ----------
    input : str or dict
        A YAML/JSON file path, or a raw/ready dictionary.

    Returns
    -------
    dict
        A configuration dictionary of a single document.
    """
    doc = {}
    if isinstance(input, str):
        if ".yaml" in input.lower() or ".json" in input.lower():
            doc = loadDocument(getAbsPathFDS(input))
        else:
            doc = loadRawYAMLString(input)
    elif isinstance(input, dict):
        doc = input
    return doc

def getListCFGDoc(input):
    """Get a list of configuration dictionary of document from the given :obj:`input`.

    Parameters
    ----------
    input : str or dict
        A YAML/JSON file path, or a raw/ready dictionary.

    Returns
    -------
    list[dict, ...]
        A list of configuration dictionary.
    """
    doc_list = []
    if isinstance(input, str):
        if ".yaml" in input.lower():
            doc_list = loadListDocument(getAbsPathFDS(input))
        else:
            doc_list = loadRawYAMLStringMT(input)
    elif isinstance(input, dict):
        for doc in input:
            doc_list.append(doc)
    return doc_list

def loadDocument(yaml_json):
    """Return a configuration dictionary of a single document from the given file 
    :obj:`yaml_json`.

    Parameters
    ----------
    yaml_json : str
        A path of a YAML/JSON file.

    Returns
    -------
    dict
        A configuration dictionary of a single document.
    """
    document = {}
    if isExist(yaml_json):
        with open(yaml_json, 'rb') as cfg:
            try:
                if '.json' in yaml_json.lower():
                    document = json.load(cfg)
                elif '.yaml' in yaml_json.lower():
                    document = yaml.load(cfg, Loader=SafeLoader)
            except ValueError as e:
                msg = 'loadDocument() -> ' + str(e)
                add_error_log(msg)
                raise ValueError(msg)
    else:
        msg = "loadDocument() -> " + str(yaml_json) + "' does not exist!"
        add_error_log(msg)
        raise ValueError(msg)
    return document

def loadListDocument(yaml_json):
    """Return a list of configuration dictionary from the given file :obj:`yaml_json`.

    Parameters
    ----------
    yaml_json : str
        A path of a YAML/JSON file.
    
    Returns
    -------
    list[dict, ...]
        A list of configuration dictionary.
    """
    document_list = []
    if isExist(yaml_json):
        with open(yaml_json, 'rb') as cfg:
            try:
                if '.json' in yaml_json.lower():
                    docs = json.load(cfg)
                elif '.yaml' in yaml_json.lower():
                    docs = yaml.load_all(cfg, Loader=SafeLoader)
            except ValueError as e:
                msg = 'loadListDocument() -> ' + str(e)
                add_error_log(msg)
                raise ValueError(msg)
            for doc in docs:
                document_list.append(doc)
    else:
        msg = "loadListDocument() -> " + str(yaml_json) + "' does not exist!"
        add_error_log(msg)
        raise ValueError(msg)
    return document_list

def loadRawYAMLString(raw_string):
    """Return a configuration dictionary of a single document from the given string 
    :obj:`raw_string`.

    Parameters
    ----------
    raw_string : str
        A raw string of JSON or YAML; for example, 
        :code:`raw_string="[{'tk_name': 'SORT'}]"`.
    
    Returns
    -------
    dict
        A configuration dictionary of a single document.
    """
    document = {}
    try:
        d = yaml.load(raw_string, Loader=SafeLoader)
        document = next(iter(d))
    except ValueError as e:
        msg = 'loadRawYAMLString() -> ' + str(e)
        add_error_log(msg)
        raise ValueError(msg)
    return document

def loadRawYAMLStringMT(raw_string):
    """Return a list of multiple YAML dictionary from the given string :obj:`raw_string`.

    Parameters
    ----------
    raw_string : str
        A raw string of JSON or YAML; for example, 
        :code:`raw_string="[{'tk_name': 'SORT'}, {'tk_name': 'DeepSORT'}]"`.
    
    Returns
    -------
    list[dict, ...]
        A list of configuration dictionary.
    """
    document_list = []
    try:
        ds = yaml.load_all(raw_string, Loader=SafeLoader)
        for d in ds:
            document_list.append(d)
    except ValueError as e:
        msg = 'loadRawYAMLStringMT() -> ' + str(e)
        add_error_log(msg)
        raise ValueError(msg)
    return document_list

def dumpDocDict(output_file, doc, header):
    """Dump a configuration dictionary of a single document into a YAML file 
    with simple format.

    Parameters
    ----------
    output_file : str
        A path file to dump.
    doc : dict
        A configuration dictionary of a single document.
    header : str
        A file header descriptoin.
    """
    try:
        with open(output_file, 'w') as dumping:
            dumping.write(header)
            for key, value in doc.items():
                dumping.write('%s: %s\n' % (key, value))
    except ValueError as e:
        msg = 'dumpDocDict() -> ' + str(e)
        add_error_log(msg)
        raise ValueError(msg)

def dumpListDocDict(output_file, doc_list, header):
    """Dump a list of YAML dictionary into a YAML file with simple format.

    Parameters
    ----------
    output_file : str
        A path file to dump.
    doc_list : list[dict, ...]
        A list of configuration dictionary.
    header : str
        A file header descriptoin.
    """
    try: 
        with open(output_file, 'w') as dumping:
            dumping.write(header)
            sep_index = 1
            for d in doc_list:
                for key, value in d.items():
                    dumping.write('%s: %s\n' % (key, value))
                if sep_index < len(doc_list):
                    dumping.write("---\n")
                    sep_index += 1
    except ValueError as e:
        msg = 'dumpListDocDict() -> ' + str(e)
        add_error_log(msg)
        raise ValueError(msg)

