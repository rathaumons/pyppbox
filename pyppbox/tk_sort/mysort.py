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


import sys
import os
import numpy as np

import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
os.environ['PATH']  = os.environ['PATH'] + ';' +  dir_path + ';'

from .sort import *


class MySort(object):


    def __init__(self, cfg):
        self.st = Sort(cfg.max_age, cfg.min_hits, cfg.iou_threshold)
        self.plist = []
        self.clist = []


    def getUpdatedID(self, box, dets):
        id = -1000
        box_list = box.tolist()
        det_list = dets.tolist()
        spread_list = []

        if len(box_list) > 0 and len(det_list) > 0:
            for b in det_list:
                max_spread = -1
                for i in range(0, 4):
                    sub_spread = abs(box_list[i] - b[i])
                    if sub_spread > max_spread:
                        max_spread = sub_spread
                spread_list.append(max_spread)
            
            sm_spread = min(spread_list)
            sm_index = spread_list.index(sm_spread)
            id = int(det_list[sm_index][4])

        return id


    def getOldIDsByCid(self, cid):
        faceid = "Unknown"
        deepid = "Unknown"

        for p in self.plist:
            if cid == p.getCid():
                faceid = p.getFaceid()
                deepid = p.getDeepid()
                break
        
        return faceid, deepid


    def update(self, _, ppobjlist):
        self.plist = self.clist
        self.clist = ppobjlist

        dets = np.empty((0, 5))
        for i in range(0, len(ppobjlist)):
            row = np.append(ppobjlist[i].getBboxTlbr(), 0).reshape(1, 5)
            dets = np.append(dets, row, axis=0)

        dets = self.st.update(dets)
        for i in range (0, len(self.clist)):
            new_cid = self.getUpdatedID(self.clist[i].getBboxTlbr(), dets)
            self.clist[i].updateCid(new_cid)
            if len(self.plist) > 0:
                try:
                    faceid, deepid = self.getOldIDsByCid(new_cid)
                    self.clist[i].updateIDs(new_cid, faceid, deepid)
                except Exception as e:
                    print("TK Sort: " + str(e))

        return self.clist
