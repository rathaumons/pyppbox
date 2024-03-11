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


__ustrings__ = UnifiedStrings()

class Person(object):

    """
    A class used to represent a person.

    Attributes
    ----------
    init_id : int
        Initial ID.
    cid : int
        Current ID.
    box_xywh : ndarray
        Bounding box [x y width height], shape=(4,), dtype=int, ndim=1.
    box_xyxy : ndarray
        Bounding box [x1 y1 x2 y2], shape=(4,), dtype=int, ndim=1.
    keypoints : ndarray
        Keypoints of the body.
    repspoint : tuple(int, int), default=(0, 0)
        Respesented 2D point (x, y).
    det_conf : float, default=0.5
        Confience of detection.
    faceid : str, default="Unknown"
        Face ID.
    deepid : str, default="Unknown"
        Deep ID.
    faceid_conf : float, default=0.0
        Confidence of :attr:`faceid`.
    deepid_conf : float, default=0.0
        Confidence of :attr:`deepid`.
    misc : list[], optional
        Miscellaneous items.
    """

    def __init__(
            self, 
            init_id, 
            cid, 
            box_xywh=[], 
            box_xyxy=[], 
            keypoints=[],
            repspoint=(0, 0), 
            det_conf=0.5,
            faceid=__ustrings__.unk_fid, 
            deepid=__ustrings__.unk_did, 
            faceid_conf=100.0, 
            deepid_conf=100.0
        ):

        """
        Construct a Person.

        Parameters
        ----------
        init_id : int
            Initial ID.
        cid : int
            Current ID.
        box_xywh : ndarray, optional
            Bounding box :code:`[x, y, width, height]`, :code:`shape=(4,)`, 
            :code:`dtype=int`, :code:`ndim=1`.
        box_xyxy : ndarray, optional
            Bounding box :code:`[x1, y1, x2, y2]`, :code:`shape=(4,)`, 
            :code:`dtype=int`, :code:`ndim=1`.
        keypoints : list[], optional
            Keypoints of the body.
        repspoint : tuple(int, int), default=(0, 0)
            Respesented 2D point (x, y).
        det_conf : float, default=0.5
            Confience of detection.
        faceid : str, default="Unknown"
            Face ID.
        deepid : str, default="Unknown"
            Deep ID.
        faceid_conf : float, default=0.0
            Confidence of :attr:`faceid`.
        deepid_conf : float, default=0.0
            Confidence of :attr:`deepid`.
        misc : list[], optional
            Miscellaneous items.
        """
        self.init_id = init_id
        self.cid = cid
        self.box_xywh = box_xywh
        self.box_xyxy = box_xyxy
        self.keypoints = keypoints
        self.repspoint = repspoint
        self.det_conf = det_conf
        self.faceid = faceid
        self.deepid = deepid
        self.faceid_conf = faceid_conf
        self.deepid_conf = deepid_conf
        self.misc = []

    def updateIDs(self, new_cid, new_faceid, new_deepid, 
                  new_faceid_conf=0.0, new_deepid_conf=0.0):
        """
        Update :attr:`cid` with :obj:`new_id`, :attr:`faceid` with :obj:`new_faceid`, 
        and :attr:`deepid` with :obj:`new_deepid`.

        Parameters
        ----------
        new_cid : int
            New current ID.
        new_faceid : str
            New face ID.
        new_deepid : str
            New deep ID.
        new_faceid_conf : float, default=0.0
            New confidence of :attr:`faceid`.
        new_deepid_conf :  float, default=0.0
            New confidence of :attr:`deepid`.
        """
        self.cid = new_cid
        self.faceid = new_faceid
        self.deepid = new_deepid
        self.faceid_conf = new_faceid_conf
        self.deepid_conf = new_deepid_conf

    def getDet(self):
        """Get a numpy array of detection bounding box with confidence in shape (5,).

        Returns
        -------
        ndarray
            Numpy array of x1, y1, x2, y2, and confidence.
        """
        return np.concatenate((np.asarray(self.box_xyxy), [self.det_conf]))
    
    def getDetRS(self):
        """Get a numpy array of detection bounding box with confidence in shape (1, 5).

        Returns
        -------
        ndarray
            Numpy array of x1, y1, x2, y2, and confidence.
        """
        return np.concatenate((np.asarray(self.box_xyxy), [self.det_conf])).reshape(1, 5)

    def __print_self__(self):
        print("Person: " + "\t" + str(self.box_xyxy) + "\t" + str(self.cid) + 
              "\t" + str(self.facid) + "\t" + str(self.deepid))


#####################################################################################


def findRepspoint(box_xyxy, calibrate_weight):
    """Find respesented point :code:`(x, y)` of a :class:`Person` object by its bounding 
    :code:`box_xyxy` of :code:`[x1, y1, x2, y2]`. The :obj:`calibrate_weight` indicates, 
    in between :code:`min(y1, y2)` and :code:`max(y1, y2)`, where the :code:`y` is.

    Parameters
    ----------
    box_xyxy : ndarray, optional
        Bounding box :code:`[x1, y1, x2, y2]`, :code:`shape=(4,)`, :code:`dtype=int`, 
        :code:`ndim=1`.
    calibrate_weight : float
        Calibration weight.

    Returns
    -------
    tuple(int, int)
        Respesented 2D point :code:`(x, y)`.
    """
    x = int((box_xyxy[0] + box_xyxy[2]) / 2)
    y_start = min(box_xyxy[1], box_xyxy[3])
    y_dist = abs(box_xyxy[1] - box_xyxy[3])
    y = int(y_start + calibrate_weight*y_dist)
    return (x, y)


def findRepspointBB(box_xyxy, prefer_top=True):
    """Find respesented point :code:`(x, y)` of a :class:`Person` object by its bounding 
    :code:`box_xyxy` of :code:`[x1, y1, x2, y2]`. :code:`x` is the middle of :code:`x1` 
    and :code:`x2` while :code:`y` is the min or max of :code:`(y1, y2)`.

    Parameters
    ----------
    box_xyxy : ndarray
        Bounding box :code:`[x1, y1, x2, y2]`, :code:`shape=(4,)`, :code:`dtype=int`, 
        :code:`ndim=1`.
    calibrate_weight : float
        Calibration weight.
    prefer_top : bool, default=True
        Decide whether :code:`y` is at the top or bottom of the bounding box.

    Returns
    -------
    tuple(int, int)
        Respesented 2D point :code:`(x, y)`.
    """
    x = int((box_xyxy[0] + box_xyxy[2]) / 2)
    y = 0
    if prefer_top:
        y = min(box_xyxy[1], box_xyxy[3])
    else:
        y = max(box_xyxy[1], box_xyxy[3])
    return (x, y)


def findRepspointUP(keypoint, box_xyxy, calibrate_weight, prefer_box=True):
    """Find respesented point :code:`(x, y)` of a :class:`Person` object by its
    YOLOv8 pose :code:`keypoint` (17 keypoints) or by the bounding :code:`box_xyxy` 
    of :code:`[x1, y1, x2, y2]`. The :obj:`calibrate_weight` indicates, in between 
    :code:`min(y1, y2)` and :code:`max(y1, y2)`, where the :code:`y` is.

    Parameters
    ----------
    keypoint : ndarray
        17 keypoints generated by YOLOv8 (Ultralytics).
    box_xyxy : ndarray
        Bounding box :code:`[x1, y1, x2, y2]`, :code:`shape=(4,)`, :code:`dtype=int`, 
        :code:`ndim=1`.
    calibrate_weight : float
        Calibration weight.
    prefer_box : bool, default=True
        Generate respesented point whether by :code:`box_xyxy` or :code:`keypoint`.

    Returns
    -------
    tuple(int, int)
        Respesented 2D point :code:`(x, y)`.
    """
    x = 0
    y = 0
    if len(keypoint) == 17 and not prefer_box:
        lshd = keypoint[5]
        rshd = keypoint[6]
        x = int((lshd[0] + rshd[0]) / 2) 
        y = int((lshd[1] + rshd[1]) / 2) 
    else:
        (x, y) = findRepspoint(box_xyxy, calibrate_weight)
    return (x, y)

