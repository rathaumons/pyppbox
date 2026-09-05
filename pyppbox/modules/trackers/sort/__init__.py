# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


from pyppbox.utils.persontools import Person
from pyppbox.utils.logtools import add_error_log, ignore_this_logger

ignore_this_logger("sort")

from .origin.sort import Sort


class MySORT(object):

    """Class used as a custom layer or interface for interacting with SORT tracker.
    """

    def __init__(self, cfg):
        """Create an independent SORT tracker with an empty identity cache.

        Parameters
        ----------
        cfg : pyppbox.config.myconfig.TCFGSORT
            Populated configuration supplying ``max_age``, ``min_hits``, and ``iou_threshold``.
        """
        self._people_by_cid = {}
        self.st = Sort(cfg.max_age, cfg.min_hits, cfg.iou_threshold)
        self.previous_list = []
        self.current_list = []


    def __getIndexFromSORTTracks__(self, box_xyxy, sort_tracks, max_spread=128):
        track_index = -1
        updated_cid = -1
        box_list = box_xyxy.tolist()
        tracks_list = sort_tracks.tolist()
        spread_list = []

        if len(box_list) > 0 and len(tracks_list) > 0:
            for b in tracks_list:
                max_spread = -1
                for i in range(0, 4):
                    sub_spread = abs(box_list[i] - b[i])
                    if sub_spread > max_spread:
                        max_spread = sub_spread
                spread_list.append(max_spread)
            
            if len(spread_list) > 0:
                sm_spread = min(spread_list)
                if sm_spread <= max_spread:
                    track_index = spread_list.index(sm_spread)
                    updated_cid = int(tracks_list[track_index][4])

        return track_index, updated_cid


    def __getIndexFromPreviousList__(self, cid):
        pindex = -1
        for i in range(0, len(self.previous_list)):
            if cid == self.previous_list[i].cid:
                pindex = i
                break
        return pindex


    def update(self, person_list, img=None):
        """Update the tracker and return the updated list of
        :class:`pyppbox.utils.persontools.Person`.

        Parameters
        ----------
        person_list : list[Person, ...]
            A list of :class:`pyppbox.utils.persontools.Person` object which stores
            the detected people in the given ``img``.
        img : object
            Defaults to ``None``.
            Being consistent with other trackers, will be ignored.

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
        if not all(isinstance(person, Person) for person in person_list):
            raise ValueError("MySORT : update() -> Expected a list of Person objects.")
        self.previous_list = list(self._people_by_cid.values())
        self.current_list = self.st.update_pyppbox(person_list)
        for person in self.current_list:
            previous = self._people_by_cid.get(person.cid)
            if previous is not None:
                person.updateIDs(person.cid, previous.faceid, previous.deepid,
                                 previous.faceid_conf, previous.deepid_conf)
                person.misc = previous.misc
            self._people_by_cid[person.cid] = person
        active_ids = {track.id for track in self.st.trackers}
        self._people_by_cid = {cid: person for cid, person in self._people_by_cid.items()
                               if cid in active_ids}
        return self.current_list
