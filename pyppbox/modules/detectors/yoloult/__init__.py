# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


from itertools import repeat

import cv2

from pyppbox.utils.persontools import Person, findRepspoint, findRepspointBB
from pyppbox.utils.commontools import to_xywh
from pyppbox.utils.logtools import ignore_this_logger


class MyYOLOULT(object):

    """Class used as a custom layer or interface for interacting with 
    detector module YOLO_Ultralytics which uses .pt model.

    ``cfg.device`` is forwarded to Ultralytics. Use ``cpu``, GPU indices,
    ``cuda:N``, or another device supported by the installed backend. An empty
    string or None uses its automatic selection; the literal ``auto`` is not a
    pyppbox YOLO device value. The settings popup provides an editable dropdown.

    Attributes
    ----------
    cfg : pyppbox.config.myconfig.DCFGYOLOULT
        A :class:`~pyppbox.config.myconfig.DCFGYOLOULT` object which manages the configurations
        of detector YOLO_Ultralytics.
    model : ``ultralytics.YOLO or ultralytics.NAS``
        A detection model object of YOLO_Ultralytics.
    colors : ``ultralytics.utils.plotting.Colors``
        Color palette created for pose models.
    skeleton : list[list[int, int], ...]
        A list used for mapping skeletons of a supported model of YOLO_Ultralytics.
    """

    def __init__(self, cfg):
        """Initialize according to the given configuration ``cfg``
        as :class:`~pyppbox.config.myconfig.DCFGYOLOULT` object.

        Parameters
        ----------
        cfg : pyppbox.config.myconfig.DCFGYOLOULT
            A :class:`~pyppbox.config.myconfig.DCFGYOLOULT` object which manages the configurations
            of detector YOLO_Ultralytics.
        """
        self.model = None
        try:
            self.cfg = cfg
            self.cpu_only = False
            if isinstance(self.cfg.device, str):
                if self.cfg.device.lower() == 'cpu':
                    self.cpu_only = True
            ignore_this_logger("ultralytics")
            ignore_this_logger("pyppbox-ultralytics")
            ignore_this_logger("vsensebox-ultralytics")
            if "nas" in self.cfg.model_file:
                # YOLO NAS isn't stable yet :/
                from ultralytics import NAS
                self.model = NAS(self.cfg.model_file)
            else:
                from ultralytics import YOLO
                self.model = YOLO(self.cfg.model_file)
                if getattr(self.model, 'task', None) == 'pose' or "pose" in self.cfg.model_file.lower():
                    from ultralytics.utils.plotting import Colors
                    self.colors = Colors()
                    self.kpt_color = self.colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
                    self.limb_color = self.colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
                    self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                                    [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        except BaseException:
            self.close()
            raise

    def close(self):
        """Release the owned backend model without clearing global CUDA caches."""
        self.model = None

    def _detections(self, img, classes, min_width_filter, alt_repspoint=False, alt_repspoint_top=True):
        """Yield aligned, width-filtered rows in backend order, with CPU keypoints."""
        result = self.model.predict(
            img,
            imgsz=int(self.cfg.imgsz),
            conf=float(self.cfg.conf),
            iou=float(self.cfg.iou),
            classes=classes,
            show_boxes=self.cfg.show_boxes,
            device=self.cfg.device,
            max_det=int(self.cfg.max_det),
            verbose=False
        )[0]
        # Transfer only the fields used by these APIs. Keep pose tensors on CPU
        # instead of converting them to NumPy and rebuilding a wrapper per row.
        boxes = result.boxes.cpu().numpy()
        keypoints = result.keypoints.cpu().data if result.keypoints is not None else repeat(None)
        for box_xyxy, conf, keypoint in zip(boxes.xyxy, boxes.conf, keypoints):
            # Integer conversion must precede the inclusive width check.
            box_xyxy = box_xyxy.astype(int)
            box_xywh = to_xywh(box_xyxy)
            if box_xywh[2] >= min_width_filter:
                if alt_repspoint:
                    repspoint = findRepspointBB(box_xyxy, prefer_top=alt_repspoint_top)
                else:
                    repspoint = findRepspoint(box_xyxy, self.cfg.repspoint_calibration)
                yield box_xywh, box_xyxy, repspoint, keypoint, float(conf)

    def _draw_detection(self, img, box_xyxy, repspoint, keypoint):
        """Draw one accepted detection in the same order for both public APIs."""
        cv2.circle(img, (repspoint[0], repspoint[1]), 5, (0, 0, 255), -1)
        cv2.rectangle(img, (box_xyxy[0], box_xyxy[1]), (box_xyxy[2], box_xyxy[3]), (255, 255, 0), 2)
        if keypoint is not None:
            self.__kpts__(img, keypoint, kpt_line=True)

    def __kpts__(self, img, kpts, radius=5, kpt_line=True):
        # Internal function
        h, w, c = img.shape
        shape = (h, w)
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim == 3
        kpt_line &= is_pose
        for i, k in enumerate(kpts):
            color_k = [int(x) for x in self.kpt_color[i]] if is_pose else self.colors(i)
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.5:
                        continue
                cv2.circle(img, (int(x_coord), int(y_coord)),radius, color_k, -1, lineType=cv2.LINE_AA)
        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < 0.5 or conf2 < 0.5:
                        continue
                if (pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0):
                    continue
                if (pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0):
                    continue
                cv2.line(img, pos1, pos2, [int(x) for x in self.limb_color[i]], thickness=2, lineType=cv2.LINE_AA)


    def detect(self, img, visual=True, classes=0, min_width_filter=15):
        """Detect general object with object's class filter ``classes``
        in a given ``numpy.ndarray`` like image.

        Parameters
        ----------
        img : ``numpy.ndarray``
            BGR image array; the same array is returned, with drawings applied in place.
        visual : bool
            Defaults to ``True``.
            An indication of whether to visualize the detected objects.
        classes : int or list[int] or None
            Defaults to ``0``.
            Object's class filter; 0 means person only, and None selects all classes.
        min_width_filter : int
            Defaults to ``15``.
            Minimum width filter of a detected object.

        Returns
        -------
        ``numpy.ndarray``
            BGR image array; the same array is returned, with drawings applied in place.
        ``list[ndarray[int, int, int, int], ...]``
            A list of bounding box :code:`ndarray[x, y, width, height]`.
        ``list[ndarray[int, int, int, int], ...]``
            A list of bounding box :code:`ndarray[x1, y1, x2, y2]`.
        list[tuple(int, int)]
            A list of represented 2D point :code:`(x, y)` of every detected object.
        ``list[torch.Tensor]``
            CPU tensors of body keypoints for pose models; empty for detection-only models.
        list[float]
            Detection confidences on a 0-1 scale, in the same order as the boxes.

        Notes
        -----
        Pass one frame array, not a filename. Copy it before calling if the original
        pixels must be preserved with ``visual=True``. The width filter is inclusive
        and measured after integer conversion in original-image pixels. Accepted
        detections retain backend order, with boxes, scores, and pose rows aligned.
        Model/inference errors propagate.
        """
        pboxes_xyxy = []
        pboxes_xywh = []
        repspoints = []
        keypoints = []
        confs = []
        for box_xywh, box_xyxy, repspoint, keypoint, conf in self._detections(img, classes, min_width_filter):
            pboxes_xywh.append(box_xywh)
            pboxes_xyxy.append(box_xyxy)
            repspoints.append(repspoint)
            if keypoint is not None:
                keypoints.append(keypoint)
            confs.append(conf)
            if visual:
                self._draw_detection(img, box_xyxy, repspoint, keypoint)
        return img, pboxes_xywh, pboxes_xyxy, repspoints, keypoints, confs

    def detectPeople(self, img, visual=True, min_width_filter=15, alt_repspoint=False, alt_repspoint_top=True):
        """Detect person(s) in a given ``numpy.ndarray`` like image.

        Parameters
        ----------
        img : ``numpy.ndarray``
            BGR image array; the same array is returned, with drawings applied in place.
        visual : bool
            Defaults to ``True``.
            An indication of whether to visualize the detected people.
        min_width_filter : int
            Defaults to ``15``.
            Minimum width filter of a detected person.
        alt_repspoint : bool
            Defaults to ``False``.
            An indication of whether to use the alternative :func:`~pyppbox.utils.persontools.findRepspointBB`.
        alt_repspoint_top : bool
            Defaults to ``True``.
            A parameter passed to ``prefer_top`` of :func:`~pyppbox.utils.persontools.findRepspointBB`.

        Returns
        -------
        list[Person, ...]
            A list of detected :class:`pyppbox.utils.persontools.Person` object.
        ``numpy.ndarray``
            BGR image array; the same array is returned, with drawings applied in place.

        Notes
        -----
        Pass one frame array, not a filename. Copy it before calling if the original
        pixels must be preserved with ``visual=True``. The width filter is inclusive
        and measured after integer conversion in original-image pixels. Accepted
        detections retain backend order, with boxes, scores, and pose rows aligned.
        Model/inference errors propagate.
        Initial IDs and CIDs start at zero on each call; persistent tracking
        IDs are assigned separately by a tracker. Representative points use box
        calibration unless an alternative box midpoint is requested.
        """
        people = []
        rows = self._detections(img, 0, min_width_filter, alt_repspoint, alt_repspoint_top)
        for i, (box_xywh, box_xyxy, repspoint, keypoint, conf) in enumerate(rows):
            people.append(Person(i, i, box_xywh=box_xywh, box_xyxy=box_xyxy,
                                 keypoints=keypoint, repspoint=repspoint, det_conf=conf))
            if visual:
                self._draw_detection(img, box_xyxy, repspoint, keypoint)
        return people, img
