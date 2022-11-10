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

    def __init__(self, init_id, cid, faceid, deepid, repspoint, bbox=[], bbox_tlbr=[]):
        self.init_id = init_id
        self.cid = cid
        self.faceid = faceid
        self.deepid = deepid
        self.repspoint = repspoint
        self.bbox = bbox
        self.bbox_tlbr = bbox_tlbr
        self.ontracked = 0

    def updateCid(self, new_cid):
        self.cid = new_cid

    def updateFaceid(self, new_faceid):
        self.faceid = new_faceid

    def updateDeepid(self, new_deepid):
        self.deepid = new_deepid

    def updateRepspoint(self, new_repspoint):
        self.repspoint = new_repspoint

    def updateBbox(self, new_bbox):
        self.bbox = new_bbox

    def updateBboxTlbr(self, new_bbox_tlbr):
        self.bbox_tlbr = new_bbox_tlbr

    def updateOnTracked(self, nframe):
        self.bbox = self.bbox + nframe

    def incrementOnTracked(self):
        self.bbox += 1

    def updateDetails(self, new_id, new_faceid, new_deepid, new_repspoint, new_bbox, new_bbox_tlbr):
        self.cid = new_id
        self.faceid = new_faceid
        self.deepid = new_deepid
        self.repspoint = new_repspoint
        self.bbox = new_bbox
        self.bbox_tlbr = new_bbox_tlbr

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

    def getBbox(self):
        return self.bbox

    def getBboxTlbr(self):
        return self.bbox_tlbr

    def getDetails(self):
        return self.init_id, self.cid, self.faceid, self.deepid, self.repspoint, self.bbox, self.bbox_tlbr

