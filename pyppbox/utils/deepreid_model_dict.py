from __future__ import division, print_function, absolute_import

import yaml

from yaml.loader import SafeLoader


class MyModel(object):

    def __init__(self):
        pass

    def set(self, mcfg):
        self.name = mcfg['name']
        self.arch = mcfg['arch']
        self.height = mcfg['height']
        self.width = mcfg['width']
        self.model_files = mcfg['model_files']


class ModelDictionary(object):

    def __init__(self):
        pass

    def loadCFG(self, cfg_file):  
        with open(cfg_file) as input_file:
            self.raw_model = yaml.load_all(input_file, Loader=SafeLoader)
            self.raw_model_list = list(self.raw_model)
            self.model_list = []
            for raw_m in self.raw_model_list:
                m = MyModel()
                m.set(raw_m)
                self.model_list.append(m)

    def getModelList(self):
        return self.model_list
    
    def findModelArch(self, model_file):
        model_arch = ""
        for m in self.model_list:
            for mf in m.model_files:
                if mf.lower() == model_file.lower():
                    model_arch = m.arch
                    break
        return model_arch

    def getWH(self, model_file):
        w = 0
        h = 0
        for m in self.model_list:
            for mf in m.model_files:
                if mf.lower() == model_file.lower():
                    w = m.width
                    h = m.height
                    break
        return (w, h)

