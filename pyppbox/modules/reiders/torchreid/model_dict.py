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


default_model_dict_yaml = joinFPathFull(getGlobalRootDir(), "modules/reiders/torchreid/model_dict.yaml")

class TorchreidModel(object):

    """
    A class used to represent configurations of a Torchreid model.

    Attributes
    ----------
    name : str
        Name of the model.
    arch : str
        Architecture of the model.
    height : int
        Image height.
    width : int
        Image height.
    model_files : list[str, ...]
        A list of the all base models and pre-trained model weights.
    """

    def __init__(self):
        pass

    def set(self, mcfg):
        """Set attributes according to the input :obj:`mcfg`.

        Parameters
        ----------
        mcfg : dict
            A configuration dictionary of a single document of the configurations.
        """
        self.name = mcfg['name']
        self.arch = mcfg['arch']
        self.height = mcfg['height']
        self.width = mcfg['width']
        self.model_files = mcfg['model_files']


class TorchreidModelDict(object):

    """
    A class used to store a dictionary for mapping name, arch, height, width, pretrained files, and base files of Torchreid models.

    Attributes
    ----------
    raw_model : dict, auto
        A configuration dictionary of a single document of the Torchreid model configurations.
    model_list : list[TorchreidModel, ...], auto
        A list of all :class:`TorchreidModel` objects.
    """

    def __init__(self, model_dict_yaml=default_model_dict_yaml):
        """Initailize and set attributes according to model_dict_yaml.

        Parameters
        ----------
        model_dict_yaml : str, default='{pyppbox root}/modules/reiders/torchreid/model_dict.yaml'
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
        """Find the arch of a Torchreid model based on the given model file.

        Parameters
        ----------
        model_file : str
            A path of a Torchreid model file.
        
        Return
        ------
        mode_arch : str
            String arch corresponding to the given model.
        """
        model_arch = ""
        for m in self.model_list:
            for mf in m.model_files:
                if mf.lower() == model_file.lower():
                    model_arch = m.arch
                    break
        return model_arch

    def getWH(self, model_file):
        """Find the input image size (width, height) of a Torchreid model based on the given model file.

        Parameters
        ----------
        model_file : str
            A path of a Torchreid model file.
        
        Return
        ------
        (w, h) : tuple(int, int)
            Size (width, height) corresponding to the given model.
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

