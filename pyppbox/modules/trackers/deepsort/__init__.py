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

    TensorFlow selects the appearance encoder's device according to its installed
    build and available hardware. Tracking and assignment run on CPU. There is no
    ``device`` config key; the popup's last parameter row shows ``device`` and
    ``auto by TensorFlow`` before Save.
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
        try:
            self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", cfg.max_cosine_distance,
                                                                    cfg.nn_budget)
            self.tracker = DSTracker(self.metric)
        except BaseException:
            close = getattr(self.encoder, 'close', None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    add_error_log(f'DeepSORT candidate cleanup failed: {error}')
            raise

    def reset(self):
        """Clear tracks/features and metadata while retaining the loaded encoder."""
        tracker = DSTracker(self.metric, self.tracker.max_iou_distance,
                            self.tracker.max_age, self.tracker.n_init)
        self.metric.samples = {}
        self.tracker = tracker
        self.previous_list = []
        self.current_list = []
        self._people_by_cid = {}
        self.current_frame = 0

    def close(self):
        """Close the owned appearance encoder and release tracks/features/people."""
        if self.encoder is not None:
            close = getattr(self.encoder, 'close', None)
            if callable(close):
                close()
            self.encoder = None
        self.metric.samples = {}
        self.tracker.tracks = []
        self.tracker.detection_indices = {}
        self.previous_list = []
        self.current_list = []
        self._people_by_cid = {}


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
            Inclusive maximum coordinate difference between an updated track box
            and its matched detection's box in ``person_list``. Assign the CID and
            cached metadata only when that matched pair passes this check.

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
        The returned list retains input order and objects, including detections
        suppressed by NMS. Only confirmed tracks with a current matched detection
        passing ``max_spread`` assign IDs and metadata. Other people retain their
        supplied CIDs. NMS indices map associations back to the original input list.
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
                    if not track.is_confirmed() or track.time_since_update > 0:
                        continue
                    detection_index = self.tracker.detection_indices.get(track.track_id)
                    if detection_index is None:
                        continue
                    # DeepSORT matches the post-NMS list; retain that association
                    # through NMS's reordering/suppression to the original Person.
                    person = self.current_list[indices[detection_index]]
                    if np.max(np.abs(track.to_tlbr() - person.box_xyxy)) > max_spread:
                        continue
                    new_cid = int(track.track_id)
                    person.cid = new_cid
                    previous = self._people_by_cid.get(new_cid)
                    if previous is not None:
                        person.faceid = previous.faceid
                        person.deepid = previous.deepid
                        person.faceid_conf = previous.faceid_conf
                        person.deepid_conf = previous.deepid_conf
                        person.misc = previous.misc
                    self._people_by_cid[new_cid] = person
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
