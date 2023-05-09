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


from __future__ import division, print_function, absolute_import

class Person(object):

    def __init__(self, init_id, cid, faceid, deepid, repspoint, box_xywh=[], box_xyxy=[], keypoints=[]):
        self.init_id = init_id
        self.cid = cid
        self.faceid = faceid
        self.deepid = deepid
        self.repspoint = repspoint
        self.box_xywh = box_xywh
        self.box_xyxy = box_xyxy
        self.keypoints = keypoints
        self.ontracked = 0

    def updateCid(self, new_cid):
        self.cid = new_cid

    def updateFaceid(self, new_faceid):
        self.faceid = new_faceid

    def updateDeepid(self, new_deepid):
        self.deepid = new_deepid

    def updateRepspoint(self, new_repspoint):
        self.repspoint = new_repspoint

    def updateKeypoints(self, new_keypoints):
        self.keypoints = new_keypoints

    def updateBoxXYWH(self, new_box_xywh):
        self.box_xywh = new_box_xywh

    def updateBoxXYXY(self, new_box_xyxy):
        self.box_xyxy = new_box_xyxy

    def updateOnTracked(self, nframe):
        self.ontracked = self.ontracked + nframe

    def incrementOnTracked(self):
        self.ontracked += 1

    def updateDetails(self, new_id, new_faceid, new_deepid, new_repspoint, new_box_xywh, new_box_xyxy, new_keypoints):
        self.cid = new_id
        self.faceid = new_faceid
        self.deepid = new_deepid
        self.repspoint = new_repspoint
        self.box_xywh = new_box_xywh
        self.box_xyxy = new_box_xyxy
        self.keypoints = new_keypoints

    def updateIDs(self, new_id, new_faceid, new_deepid):
        self.cid = new_id
        self.faceid = new_faceid
        self.deepid = new_deepid

    def getInitid(self):
        return self.init_id

    def getCid(self):
        return self.cid

    def getFaceid(self):
        return self.faceid

    def getDeepid(self):
        return self.deepid

    def getRepspoint(self):
        return self.repspoint

    def getKeypoint(self):
        return self.keypoints

    def getBoxXYWH(self):
        return self.box_xywh

    def getBoxXYXY(self):
        return self.box_xyxy

    def getDetails(self):
        return self.init_id, self.cid, self.faceid, self.deepid, self.repspoint, self.box_xywh, self.box_xyxy, self.keypoints

