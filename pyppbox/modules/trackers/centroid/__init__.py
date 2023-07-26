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


from math import hypot
from pyppbox.utils.persontools import Person
from pyppbox.utils.logtools import add_error_log


class MyCentroid(object):

    """Class reprensented a Centroid tracker.
    """

    def __init__(self, cfg):
        """Initialize according to the given :obj:`cfg` and :obj:`auto_load`.

        Parameters
        ----------
        cfg : TCFGCentroid
            A :class:`TCFGCentroid` object which manages the configurations of tracker Centroid.
        """
        self.max_spread = cfg.max_spread
        self.previous_list = []
        self.current_list = []

    def __generateID__(self):
        self.used_cids = list(set(self.used_cids))
        if len(self.used_cids) == 0: aID = 0
        else: aID = max(self.used_cids) + 1
        self.used_cids.append(aID)
        return aID
    
    def __findPID__(self, point):
        pindex = -1
        min_dist = 8192
        i = 0
        for p in self.previous_list:
            dist = hypot(p.repspoint[0] - point[0], 
                         p.repspoint[1] - point[1])
            if dist < min_dist:
                min_dist = dist
                pindex = i
            i += 1
        if min_dist > self.max_spread: pindex = -1
        return pindex

    def update(self, person_list, img=None):
        """Update the tracker and return the updated list of :class:`Person`.

        Parameters
        ----------
        person_list : list[Person, ...]
            A list of :class:`Person` object which stores the detected people in the given :obj:`img`.
        img : any, default=None
            Being consistent with other trackers, will be ignored.

        Returns
        -------
        list[Person, ...]
            The updated list of :class:`Person` object.
        """
        self.previous_list = self.current_list
        self.current_list = []
        self.used_cids = []

        if len(person_list) > 0:
            if isinstance(person_list[0], Person):
                self.current_list = person_list
                hang_indexes_in_clist = []
                len_clist = len(self.current_list)
                if len_clist > 0:
                    for i in range(0, len_clist):
                        pindex = self.__findPID__(self.current_list[i].repspoint)
                        if pindex >= 0:
                            prev_cid = self.previous_list[pindex].cid
                            if prev_cid in self.used_cids:
                                hang_indexes_in_clist.append(i)
                            else:
                                self.current_list[i].updateIDs(
                                    prev_cid, 
                                    self.previous_list[pindex].faceid, 
                                    self.previous_list[pindex].deepid,
                                    self.previous_list[pindex].faceid_conf,
                                    self.previous_list[pindex].deepid_conf
                                )
                                self.used_cids.append(prev_cid)
                                # self.previous_list.pop(pindex)
                        else:
                            hang_indexes_in_clist.append(i)
                    len_hlist = len(hang_indexes_in_clist)
                    if len_hlist > 0:
                        for index in hang_indexes_in_clist:
                            self.current_list[index].cid = self.__generateID__()
            else:
                msg = ("MyCentroid : update() -> The element of input 'person_list' " + 
                       "list has unsupported type.")
                add_error_log(msg)
                raise ValueError(msg)

        return self.current_list
