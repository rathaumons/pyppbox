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

