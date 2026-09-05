# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


import numpy as np

from pyppbox.utils.persontools import Person
from pyppbox.utils.logtools import add_error_log, ignore_this_logger

ignore_this_logger("tensorflow")
ignore_this_logger("preprocessing")
ignore_this_logger("nn_matching")
ignore_this_logger("generate_detections")
ignore_this_logger("detection")
ignore_this_logger("tracker")

from .origin import preprocessing
from .origin import nn_matching
from .origin import generate_detections as gdet
from .origin.detection import Detection as DSDetection
from .origin.tracker import Tracker as DSTracker


class MyDeepSORT(object):

    """Class used as a custom layer or interface for interacting with DeepSORT tracker.
    """

    def __init__(self, cfg):
        """Initialize according to the given ``cfg``.

        Parameters
        ----------
        cfg : pyppbox.config.myconfig.TCFGDeepSORT
            A :class:`~pyppbox.config.myconfig.TCFGDeepSORT` object which manages the configurations of tracker DeepSORT.
        """
        self.previous_list = []
        self.current_list = []
        self._people_by_cid = {}
        self.current_frame = 0
        self.nms_max_overlap = cfg.nms_max_overlap
        self.encoder = gdet.create_box_encoder(cfg.model_file, batch_size=16)
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", cfg.max_cosine_distance, 
                                                                cfg.nn_budget)
        self.tracker = DSTracker(self.metric)


    def __getCurrentIndexByBoxXYXY__(self, box, max_spread=128):
        index = -1
        box_list = box.tolist()
        min_box_spread = 8192
        i = 0
        for p in self.current_list:
            pbbox_list = p.box_xyxy.tolist()
            max_ss = max([abs(box_list[j] - pbbox_list[j]) for j in range(0, 4)])
            if max_ss < min_box_spread:
                min_box_spread = max_ss
                index = i
            i += 1
        if min_box_spread > max_spread: index = -1
        return index


    def __getIndexFromPreviousList__(self, cid):
        pindex = -1
        for i in range(0, len(self.previous_list)):
            if cid == self.previous_list[i].cid:
                pindex = i
                break
        return pindex


    def update(self, person_list, img=None, max_spread=128):
        """Update the tracker and return the updated list of
        :class:`pyppbox.utils.persontools.Person`.

        Parameters
        ----------
        person_list : list[Person, ...]
            A list of :class:`pyppbox.utils.persontools.Person` object which stores
            the detected people in the given ``img``.
        img : object
            Defaults to ``None``.
            BGR frame array required for appearance encoding when detections are present.
        max_spread : int
            Defaults to ``128``.
            Max spread or max margin used to decide whether 2 bounding boxes are the same by comparing
            the differences between the elements in the bounding box given by the embedded DeepSORT and the
            coressponding elements of a person's bounding box in the ``person_list``.

        Returns
        -------
        list[Person, ...]
            The updated list of :class:`pyppbox.utils.persontools.Person` object.

        Notes
        -----
        Call once per input frame and keep one tracker instance per stream. Person IDs,
        identity metadata, and ``misc`` can be updated in place; returned people are not
        independent snapshots. Use a result recorder or copy objects for historical data.
        An empty input advances track age and returns an empty list. Metadata
        is retained only while the underlying track remains alive. Returned values
        represent current detections, not a list of predicted boxes for missing people.
        """
        self.previous_list = list(self._people_by_cid.values())
        self.current_list = []

        if len(person_list) > 0:
            if isinstance(person_list[0], Person):
                self.current_list = person_list
                dboxes = []
                dconfidences = []
                dclasses = []

                for i in range(0, len(person_list)):
                    dboxes.append(person_list[i].box_xywh)
                    dconfidences.append(person_list[i].det_conf)
                    dclasses.append('person')

                dfeatures = self.encoder(img, dboxes)
                detections = [DSDetection(dbox, dconfidence, dclass, dfeature) 
                              for dbox, dconfidence, dclass, dfeature in 
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
                    box_xyxy = track.to_tlbr()
                    new_cid = int(track.track_id)
                    cindex = self.__getCurrentIndexByBoxXYXY__(box_xyxy, max_spread=max_spread)
                    if cindex >= 0:
                        self.current_list[cindex].cid = new_cid
                        pindex = self.__getIndexFromPreviousList__(new_cid)
                        if pindex >= 0:
                            self.current_list[cindex].faceid = self.previous_list[pindex].faceid
                            self.current_list[cindex].deepid = self.previous_list[pindex].deepid
                            self.current_list[cindex].faceid_conf = self.previous_list[pindex].faceid_conf
                            self.current_list[cindex].deepid_conf = self.previous_list[pindex].deepid_conf
                            self.current_list[cindex].misc = self.previous_list[pindex].misc
                        self._people_by_cid[new_cid] = self.current_list[cindex]
            else:
                msg = ("MyDeepSORT : update() -> The element of input 'person_list' list has unsupported type.")
                add_error_log(msg)
                raise ValueError(msg)

        else:
            self.tracker.predict()
            self.tracker.update([])

        active_ids = {track.track_id for track in self.tracker.tracks}
        self._people_by_cid = {cid: person for cid, person in self._people_by_cid.items()
                               if cid in active_ids}
        self.current_frame += 1
        return self.current_list
