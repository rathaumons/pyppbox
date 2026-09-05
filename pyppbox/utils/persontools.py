# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


import numpy as np
from pyppbox.config.unifiedstrings import UnifiedStrings


__ustrings__ = UnifiedStrings()

class Person(object):

    """A class used to represent a person.

    Attributes
    ----------
    init_id : int
        Initial ID.
    cid : int
        Current ID.
    box_xywh : ``numpy.ndarray or list``
        Bounding box [x y width height], shape=(4,), dtype=int, ndim=1.
    box_xyxy : ``numpy.ndarray or list``
        Bounding box [x1 y1 x2 y2], shape=(4,), dtype=int, ndim=1.
    keypoints : ``numpy.ndarray or torch.Tensor or list``
        Keypoints of the body.
    repspoint : tuple(int, int)
        Defaults to ``(0, 0)``.
        Representative 2D point (x, y).
    det_conf : float
        Defaults to ``0.5``.
        Confidence of detection on a 0-1 scale.
    faceid : str
        Defaults to ``"Unknown"``.
        Face ID.
    deepid : str
        Defaults to ``"Unknown"``.
        Deep ID.
    faceid_conf : float
        Defaults to ``100.0``.
        Confidence of :attr:`faceid` on a 0-100 scale.
    deepid_conf : float
        Defaults to ``100.0``.
        Confidence of :attr:`deepid` on a 0-100 scale.
    misc : list[]
        Optional.
        Miscellaneous items.

    Notes
    -----
    Omitted box/keypoint values become independent empty lists. Supplied values
    are stored by reference without copying, shape validation, or coordinate conversion.
    """

    def __init__(
            self, 
            init_id, 
            cid, 
            box_xywh=None,
            box_xyxy=None,
            keypoints=None,
            repspoint=(0, 0), 
            det_conf=0.5,
            faceid=__ustrings__.unk_fid, 
            deepid=__ustrings__.unk_did, 
            faceid_conf=100.0, 
            deepid_conf=100.0
        ):

        """Construct a Person.

        Parameters
        ----------
        init_id : int
            Initial ID.
        cid : int
            Current ID.
        box_xywh : ``numpy.ndarray or list``
            Optional.
            Bounding box :code:`[x, y, width, height]`, :code:`shape=(4,)`,
            :code:`dtype=int`, :code:`ndim=1`.
        box_xyxy : ``numpy.ndarray or list``
            Optional.
            Bounding box :code:`[x1, y1, x2, y2]`, :code:`shape=(4,)`,
            :code:`dtype=int`, :code:`ndim=1`.
        keypoints : ``numpy.ndarray or torch.Tensor or list``
            Optional.
            Keypoints of the body.
        repspoint : tuple(int, int)
            Defaults to ``(0, 0)``.
            Representative 2D point (x, y).
        det_conf : float
            Defaults to ``0.5``.
            Confidence of detection on a 0-1 scale.
        faceid : str
            Defaults to ``"Unknown"``.
            Face ID.
        deepid : str
            Defaults to ``"Unknown"``.
            Deep ID.
        faceid_conf : float
            Defaults to ``100.0``.
            Confidence of :attr:`faceid` on a 0-100 scale.
        deepid_conf : float
            Defaults to ``100.0``.
            Confidence of :attr:`deepid` on a 0-100 scale.

        Notes
        -----
        ``misc`` starts as an independent empty list and can be set after construction.
        The default identity confidence is a compatibility value, not a prediction.
        ReID recognition replaces it with the classifier confidence.
        Omitted box/keypoint values become independent empty lists. Supplied values
        are stored by reference without copying, shape validation, or coordinate conversion.
        """
        self.init_id = init_id
        self.cid = cid
        self.box_xywh = [] if box_xywh is None else box_xywh
        self.box_xyxy = [] if box_xyxy is None else box_xyxy
        self.keypoints = [] if keypoints is None else keypoints
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
        Update :attr:`cid` with ``new_cid``, :attr:`faceid` with ``new_faceid``,
        and :attr:`deepid` with ``new_deepid``.

        Parameters
        ----------
        new_cid : int
            New current ID.
        new_faceid : str
            New face ID.
        new_deepid : str
            New deep ID.
        new_faceid_conf : float
            Defaults to ``0.0``.
            New confidence of :attr:`faceid`.
        new_deepid_conf : float
            Defaults to ``0.0``.
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
        ``numpy.ndarray``
            Numpy array of x1, y1, x2, y2, and confidence.
        """
        return np.concatenate((np.asarray(self.box_xyxy), [self.det_conf]))
    
    def getDetRS(self):
        """Get a numpy array of detection bounding box with confidence in shape (1, 5).

        Returns
        -------
        ``numpy.ndarray``
            Numpy array of x1, y1, x2, y2, and confidence.
        """
        return np.concatenate((np.asarray(self.box_xyxy), [self.det_conf])).reshape(1, 5)

    def __print_self__(self):
        print(f"Person: \t{self.cid}\t{self.box_xyxy}\t{self.cid}\t{self.faceid}\t{self.deepid}")


#####################################################################################


def findRepspoint(box_xyxy, calibrate_weight):
    """Find respesented point :code:`(x, y)` of a :class:`pyppbox.utils.persontools.Person` 
    object by its bounding :code:`box_xyxy` of :code:`[x1, y1, x2, y2]`. The ``calibrate_weight``
    indicates, in between :code:`min(y1, y2)` and :code:`max(y1, y2)`, where the :code:`y` is.

    Parameters
    ----------
    box_xyxy : ``ndarray``
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
    """Find respesented point :code:`(x, y)` of a :class:`pyppbox.utils.persontools.Person` 
    object by its bounding :code:`box_xyxy` of :code:`[x1, y1, x2, y2]`. :code:`x` is the middle 
    of :code:`x1` and :code:`x2` while :code:`y` is the min or max of :code:`(y1, y2)`.

    Parameters
    ----------
    box_xyxy : ``ndarray``
        Bounding box :code:`[x1, y1, x2, y2]`, :code:`shape=(4,)`, :code:`dtype=int`, 
        :code:`ndim=1`.
    prefer_top : bool
        Defaults to ``True``.
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
    """Find respesented point :code:`(x, y)` of a :class:`pyppbox.utils.persontools.Person` 
    object by its YOLOv8 pose :code:`keypoint` (17 keypoints) or by the bounding :code:`box_xyxy` 
    of :code:`[x1, y1, x2, y2]`. The ``calibrate_weight`` indicates, in between
    :code:`min(y1, y2)` and :code:`max(y1, y2)`, where the :code:`y` is.

    Parameters
    ----------
    keypoint : ``ndarray``
        17 keypoints generated by YOLOv8 (Ultralytics).
    box_xyxy : ``ndarray``
        Bounding box :code:`[x1, y1, x2, y2]`, :code:`shape=(4,)`, :code:`dtype=int`, 
        :code:`ndim=1`.
    calibrate_weight : float
        Calibration weight.
    prefer_box : bool
        Defaults to ``True``.
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

