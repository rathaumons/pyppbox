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


import os
import sys

import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
os.environ['PATH']  = os.environ['PATH'] + ';' +  dir_path + ';'

import numpy as np

from . import preprocessing
from . import nn_matching
from . import generate_detections as gdet
from .detection import Detection as DSDetection
from .tracker import Tracker as DSTracker


class MyDeepSort(object):


    def __init__(self, cfg):
        self.plist = []
        self.clist = []
        self.nms_max_overlap = cfg.nms_max_overlap
        self.current_frame = 0
        self.encoder = gdet.create_box_encoder(cfg.model_file, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", cfg.max_cosine_distance, cfg.nn_budget)
        self.tracker = DSTracker(self.metric)


    def getOldIndex(self, box, ppobj):
        index = -1
        box_list = box.tolist()
        spread_list = []

        for p in ppobj:
            max_spread = -1
            pbbox_list = p.getBbox().tolist()
            for i in range(0, 4):
                sub_spread = abs(box_list[i] - pbbox_list[i])
                if sub_spread > max_spread:
                    max_spread = sub_spread
            spread_list.append(max_spread)

        sm_spread = min(spread_list)
        index = spread_list.index(sm_spread)
        return index


    def getCurrentIndexByBbox(self, box):
        box_list = box.tolist()
        spread_list = []

        for p in self.clist:
            max_spread = -1
            pbbox_list = p.getBboxTlbr().tolist()
            for i in range(0, 4):
                sub_spread = abs(box_list[i] - pbbox_list[i])
                if sub_spread > max_spread:
                    max_spread = sub_spread
            spread_list.append(max_spread)

        sm_spread = min(spread_list)
        index = spread_list.index(sm_spread)
        return index


    def getOldIDsByCid(self, cid):
        faceid = "Unknown"
        deepid = "Unknown"

        if self.current_frame > 3:
            for p in self.plist:
                if cid == p.getCid():
                    faceid = p.getFaceid()
                    deepid = p.getDeepid()
                    break

        return faceid, deepid


    def update(self, frame, ppobjlist):
        self.plist = self.clist
        self.clist = ppobjlist

        dboxes = []
        dconfidences = []
        dclasses = []

        for i in range(0, len(ppobjlist)):
            dboxes.append(ppobjlist[i].getBbox())
            dconfidences.append(0.5)
            dclasses.append('person')

        dfeatures = self.encoder(frame, dboxes)
        detections = [DSDetection(dbox, dconfidence, dclass, dfeature) for dbox, dconfidence, dclass, dfeature in
                        zip(dboxes, dconfidences, dclasses, dfeatures)]

        dboxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(dboxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        self.tracker.predict()
        self.tracker.update(detections)

        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlbr()
            new_cid = int(track.track_id)
            cindex = self.getCurrentIndexByBbox(box)
            self.clist[cindex].updateCid(new_cid)
            faceid, deepid = self.getOldIDsByCid(new_cid)
            self.clist[cindex].updateFaceid(faceid)
            self.clist[cindex].updateDeepid(deepid)

        self.current_frame += 1
        return self.clist
