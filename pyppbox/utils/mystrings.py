from __future__ import division, print_function, absolute_import

import yaml

from yaml.loader import SafeLoader


class MyStrings(object):

    def __init__(self, strings_yaml):
        self.load(strings_yaml)

    def load(self, strings_yaml): 
        with open(strings_yaml, 'r') as str_cfg:
            self.data = yaml.load(str_cfg, Loader=SafeLoader)
        self.set(self.data)

    def set(self, data):
        # module
        self.none = data['none']
        self.detector = data['detector']
        self.tracker = data['tracker']
        self.reider = data['reider']
        # detector
        self.gt = data['gt']
        self.yolo = data['yolo']
        self.openpose = data['openpose']
        # tracker
        self.sort = data['sort']
        self.deepsort = data['deepsort']
        self.centroid = data['centroid']
        # reider
        self.facenet = data['facenet']
        self.deepreid = data['deepreid']
        # internal
        self.dtname_yl = data['dtname_yl']
        self.dtname_op = data['dtname_op']
        self.dtname_gt = data['dtname_gt']
        self.tkname_ct = data['tkname_ct']
        self.tkname_st = data['tkname_st']
        self.tkname_ds = data['tkname_ds']
        self.riname_fn = data['riname_fn']
        self.riname_dr = data['riname_dr']
        self.unk_did = data['unk_did']
        self.unk_fid = data['unk_fid']
        self.err_did = data['err_did']
        self.err_fid = data['err_fid']

