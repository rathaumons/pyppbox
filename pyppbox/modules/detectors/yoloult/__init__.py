# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                           #
#   pyppbox: Toolbox for people detecting, tracking, and re-identifying.    #
#   Copyright (C) 2025 UMONS-Numediart                                      #
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


import cv2

from pyppbox.utils.persontools import Person, findRepspoint, findRepspointBB
from pyppbox.utils.commontools import to_xywh
from pyppbox.utils.logtools import ignore_this_logger


class MyYOLOULT(object):

    """Class used as a custom layer or interface for interacting with 
    detector module YOLO_Ultralytics which uses .pt model.

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
            if "pose" in self.cfg.model_file.lower():
                from ultralytics.utils.plotting import Colors
                self.colors = Colors()
                self.kpt_color = Colors().pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
                self.limb_color = Colors().pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
                self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                                [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

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
        classes : int
            Defaults to ``0``.
            Object's class filter, 0 means person only
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
        and measured in original-image pixels. Model/inference errors propagate.
        """
        numpy_dets = []
        pboxes_xyxy = []
        pboxes_xywh = []
        repspoints = []
        keypoints = []
        confs = []
        dets = self.model.predict(
            img,
            imgsz=int(self.cfg.imgsz),
            conf=float(self.cfg.conf),
            iou=float(self.cfg.iou),
            classes=classes,
            show_boxes=self.cfg.show_boxes,
            device=self.cfg.device,
            max_det=int(self.cfg.max_det),
            verbose=False
        )
        cpu_dets = dets[0].cpu()
        numpy_dets = cpu_dets.numpy()
        dt_boxes_xyxy = numpy_dets.boxes.xyxy
        dt_confidences = numpy_dets.boxes.conf
        dt_keypoints = cpu_dets.keypoints
        if dt_keypoints is not None:
            for box_xyxy, conf, kp in zip(dt_boxes_xyxy, dt_confidences, dt_keypoints):
                box_xyxy = box_xyxy.astype(int)
                box_xywh = to_xywh(box_xyxy)
                if box_xywh[2] >= min_width_filter:
                    pboxes_xywh.append(box_xywh)
                    pboxes_xyxy.append(box_xyxy)
                    repspoint = findRepspoint(box_xyxy, self.cfg.repspoint_calibration)
                    repspoints.append(repspoint)
                    keypoint = kp.data[0]
                    keypoints.append(keypoint)
                    confs.append(float(conf))
                    if visual:
                        cv2.circle(img, (repspoint[0], repspoint[1]), 5, (0, 0, 255), -1)
                        cv2.rectangle(img, (box_xyxy[0], box_xyxy[1]), (box_xyxy[2], box_xyxy[3]), (255, 255, 0), 2)
                        self.__kpts__(img, keypoint, kpt_line=True)
        elif len(dt_boxes_xyxy) > 0:
            for box_xyxy, conf in zip(dt_boxes_xyxy, dt_confidences):
                box_xyxy = box_xyxy.astype(int)
                box_xywh = to_xywh(box_xyxy)
                if box_xywh[2] >= min_width_filter:
                    pboxes_xywh.append(box_xywh)
                    pboxes_xyxy.append(box_xyxy)
                    repspoint = findRepspoint(box_xyxy, self.cfg.repspoint_calibration)
                    repspoints.append(repspoint)
                    confs.append(float(conf))
                    if visual:
                        cv2.circle(img, (repspoint[0], repspoint[1]), 5, (0, 0, 255), -1)
                        cv2.rectangle(img, (box_xyxy[0], box_xyxy[1]), (box_xyxy[2], box_xyxy[3]), (255, 255, 0), 2)
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
        and measured in original-image pixels. Model/inference errors propagate.
        Initial IDs and CIDs start at zero on each call; persistent tracking
        IDs are assigned separately by a tracker. Representative points use box
        calibration unless an alternative box midpoint is requested.
        """
        numpy_dets = []
        people = []
        dets = self.model.predict(
            img,
            imgsz=int(self.cfg.imgsz),
            conf=float(self.cfg.conf),
            iou=float(self.cfg.iou),
            classes=0,
            show_boxes=self.cfg.show_boxes,
            device=self.cfg.device,
            max_det=int(self.cfg.max_det),
            verbose=False
        )
        cpu_dets = dets[0].cpu()
        numpy_dets = cpu_dets.numpy()
        dt_boxes_xyxy = numpy_dets.boxes.xyxy
        dt_confidences = numpy_dets.boxes.conf
        dt_keypoints = cpu_dets.keypoints
        if dt_keypoints is not None:
            i = 0
            for box_xyxy, conf, kp in zip(dt_boxes_xyxy, dt_confidences, dt_keypoints):
                box_xyxy = box_xyxy.astype(int)
                box_xywh = to_xywh(box_xyxy)
                if box_xywh[2] >= min_width_filter:
                    keypoint = kp.data[0]
                    if alt_repspoint: repspoint = findRepspointBB(box_xyxy, prefer_top=alt_repspoint_top)
                    else: repspoint = findRepspoint(box_xyxy, self.cfg.repspoint_calibration)
                    people.append(Person(i, i, box_xywh=box_xywh, box_xyxy=box_xyxy,
                                  keypoints=keypoint, repspoint=repspoint, det_conf=float(conf)))
                    i += 1
                    if visual:
                        cv2.circle(img, (repspoint[0], repspoint[1]), 5, (0, 0, 255), -1)
                        cv2.rectangle(img, (box_xyxy[0], box_xyxy[1]), (box_xyxy[2], box_xyxy[3]), (255, 255, 0), 2)
                        self.__kpts__(img, keypoint, kpt_line=True)
        elif len(dt_boxes_xyxy) > 0:
            i = 0
            for box_xyxy, conf in zip(dt_boxes_xyxy, dt_confidences):
                box_xyxy = box_xyxy.astype(int)
                box_xywh = to_xywh(box_xyxy)
                if box_xywh[2] >= min_width_filter:
                    if alt_repspoint: repspoint = findRepspointBB(box_xyxy, prefer_top=alt_repspoint_top)
                    else: repspoint = findRepspoint(box_xyxy, self.cfg.repspoint_calibration)
                    people.append(Person(i, i, box_xywh=box_xywh, box_xyxy=box_xyxy, 
                                  repspoint=repspoint, det_conf=float(conf)))
                    i += 1
                    if visual:
                        cv2.circle(img, (repspoint[0], repspoint[1]), 5, (0, 0, 255), -1)
                        cv2.rectangle(img, (box_xyxy[0], box_xyxy[1]), (box_xyxy[2], box_xyxy[3]), (255, 255, 0), 2)
        return people, img
