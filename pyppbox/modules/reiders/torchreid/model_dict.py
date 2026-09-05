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


import yaml

from yaml.loader import SafeLoader
from pyppbox.utils.commontools import joinFPathFull, getGlobalRootDir


default_model_dict_yaml = joinFPathFull(getGlobalRootDir(), "modules/reiders/torchreid/model_dict.yaml")

class TorchreidModel(object):

    """Store one model-catalog entry; call ``set()`` to populate its attributes.

    Attributes
    ----------
    name : str
        Catalog model name.
    arch : str
        Feature-extractor architecture name.
    height : int
        Expected input height in pixels.
    width : int
        Expected input width in pixels.
    model_files : list[str]
        Weight filenames recognized for this entry.
    """

    def __init__(self):
        pass

    def set(self, mcfg):
        """Set attributes according to the input :obj:`mcfg`.

        Parameters
        ----------
        mcfg : Dict[str, Any]
            A configuration dictionary of a single document of the configurations.
        """
        self.name = mcfg['name']
        self.arch = mcfg['arch']
        self.height = mcfg['height']
        self.width = mcfg['width']
        self.model_files = mcfg['model_files']


class TorchreidModelDict(object):

    """Load the model catalog and look up architectures and image dimensions by filename.

    Attributes
    ----------
    raw_model : object
        YAML document iterator, consumed during construction.
    model_list : list[TorchreidModel]
        Loaded catalog entries, in file order.
    """

    def __init__(self, model_dict_yaml=default_model_dict_yaml):
        """Initialize and set attributes according to model_dict_yaml.

        Parameters
        ----------
        model_dict_yaml : str
            Defaults to ``'{pyppbox root}/modules/reiders/torchreid/model_dict.yaml'``.
            A path of a YAML file which stores the dictionary of Torchreid models.
        """
        with open(model_dict_yaml) as input_file:
            self.raw_model = yaml.load_all(input_file, Loader=SafeLoader)
            self.model_list = []
            for raw_m in list(self.raw_model):
                m = TorchreidModel()
                m.set(raw_m)
                self.model_list.append(m)
    
    def findModelArch(self, model_file):
        """Look up the architecture for a catalog weight filename.

        Parameters
        ----------
        model_file : str
            Filename, not a directory path. Matching is case-insensitive against
            catalog ``model_files`` entries.

        Returns
        -------
        str
            Architecture name, or an empty string when no entry matches.
        """
        model_arch = ""
        for m in self.model_list:
            for mf in m.model_files:
                if mf.lower() == model_file.lower():
                    model_arch = m.arch
                    break
        return model_arch

    def getWH(self, model_file):
        """Look up input dimensions for a catalog weight filename.

        Parameters
        ----------
        model_file : str
            Filename, not a directory path. Matching is case-insensitive against
            catalog ``model_files`` entries.

        Returns
        -------
        tuple[int, int]
            (width, height), or (0, 0) when no entry matches.
        """
        w = 0
        h = 0
        for m in self.model_list:
            for mf in m.model_files:
                if mf.lower() == model_file.lower():
                    w = m.width
                    h = m.height
                    break
        return (w, h)

