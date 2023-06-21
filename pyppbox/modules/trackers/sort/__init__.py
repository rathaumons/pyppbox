# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                           #
#   pyppbox: Toolbox for people detecting, tracking, and re-identifying.    #
#   Copyright (C) 2022 UMONS-Numediart                                      #
#                                                                           #
#   This program is free software: you can redistribute it and/or modify    #
#   it under the terms of the GNU General Public License as published by    #
#   the Free Software Foundation, either version 3 of the License, or       #
#   (at your option) any later version.                                     #
#                                                                           #
#   This program is distributed in the hope that it will be useful,         #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#   GNU General Public License for more details.                            #
#                                                                           #
#   You should have received a copy of the GNU General Public License       #
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.  #
#                                                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import numpy as np

from pyppbox.config.unifiedstrings import UnifiedStrings
from pyppbox.utils.persontools import Person
from pyppbox.utils.logtools import add_error_log, ignore_this_logger

ignore_this_logger("sort")

from .origin.sort import Sort


__ustrings__ = UnifiedStrings()

class MySORT(object):


    def __init__(self, cfg):
        """Initialize according to the given :obj:`cfg` and :obj:`auto_load`.

        Parameters
        ----------
        cfg : TCFGSORT
            A :class:`TCFGDeepSORT` object which manages the configurations of tracker SORT.
        """
        self.st = Sort(cfg.max_age, cfg.min_hits, cfg.iou_threshold)
        self.previous_list = []
        self.current_list = []


    def __getIndexFromSORTTracks__(self, box_xyxy, sort_tracks, max_spread=50):
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


    def update(self, person_list, img=None, max_spread=5):
        """Update the tracker and return the updated list of :class:`Person`.

        Parameters
        ----------
        person_list : list[Person, ...]
            A list of :class:`Person` object which stores the detected people in the given :obj:`img`.
        img : any, default=None
            Being consistent with other trackers, will be ignored.
        max_spread : int, default=5
            Max spread or max margin used to decide whether 2 bounding boxes are the same by comparing 
            the differences between the elements in the bounding box given by the embedded SORT and the 
            coressponding elements of a person's bounding box in the :obj:`person_list`.

        Returns
        -------
        list[Person, ...]
            The updated list of :class:`Person` object.
        """
        self.previous_list = self.current_list
        self.current_list = []

        if len(person_list) > 0:
            if isinstance(person_list[0], Person):
                self.current_list = person_list

                dets = np.empty((0, 5))
                for i in range(0, len(person_list)):
                    row = np.append(person_list[i].box_xyxy, 0).reshape(1, 5)
                    dets = np.append(dets, row, axis=0)

                sort_tracks = self.st.update(dets)
                for i in range (0, len(self.current_list)):
                    track_index, updated_cid = self.__getIndexFromSORTTracks__(
                        self.current_list[i].box_xyxy, sort_tracks, 
                        max_spread=max_spread
                    )
                    if track_index >= 0:
                        self.current_list[i].cid = updated_cid
                        if len(self.previous_list) > 0:
                            pindex = self.__getIndexFromPreviousList__(updated_cid)
                            if pindex >= 0:
                                self.current_list[i].faceid = self.previous_list[pindex].faceid
                                self.current_list[i].deepid = self.previous_list[pindex].deepid
                                self.current_list[i].faceid_conf = self.previous_list[pindex].faceid_conf
                                self.current_list[i].deepid_conf = self.previous_list[pindex].deepid_conf
            else:
                msg = ("MySORT : update() -> The element of input 'person_list' list " + 
                       "has unsupported type.")
                add_error_log(msg)
                raise ValueError(msg)

        return self.current_list
