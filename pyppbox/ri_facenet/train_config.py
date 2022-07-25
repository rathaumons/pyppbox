"""
    pyppbox: Toolbox for people detecting, tracking, and re-identifying.
    Copyright (C) 2022 UMONS-Numediart

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import yaml
from yaml.loader import SafeLoader

class CFGFacenet(object):

    def __init__(self):
        self.loadCFG("train_config.yaml")

    def set(self, cfg):
        self.gpu_mem = cfg['gpu_mem']
        self.model_det = cfg['model_det']
        self.model_file = cfg['model_file']
        self.classifier_file = cfg['classifier_file']
        self.data_path = cfg['data_path']
        self.batch_size = cfg['batch_size']
        self.min_confidence = cfg['min_confidence']

    def loadCFG(self, cfg_file):
        with open(cfg_file) as rcf:
            cfg = yaml.load_all(rcf, Loader=SafeLoader)
            self.set(next(cfg))
