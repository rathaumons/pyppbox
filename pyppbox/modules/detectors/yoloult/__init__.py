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


import cv2

from pyppbox.utils.persontools import Person, findRepspoint, findRepspointBB
from pyppbox.utils.commontools import to_xywh
from pyppbox.utils.logtools import ignore_this_logger


class MyYOLOULT(object):

    """Class used as a custom layer or interface for interacting with 
    detector module YOLO_Ultralytics which uses .pt model.

    Attributes
    ----------
    cfg : DCFGYOLOULT
        A :class:`DCFGYOLOULT` object which manages the configurations 
        of detector YOLO_Ultralytics.
    model: ultralytics.yolo.engine.YOLO
        A detection model object of YOLO_Ultralytics.
    colors: ultralytics.yolo.utils.Colors
        A hex color object of YOLO_Ultralytics.
    skeleton : list[list[int, int], ...]
        A list used for mapping skeletons of a supported model of YOLO_Ultralytics.
    """

    def __init__(self, cfg):
        """Initialize according to the given configuration :obj:`cfg` 
        as :class:`DCFGYOLOULT` object.

        Parameters
        ----------
        cfg : DCFGYOLOULT
            A :class:`DCFGYOLOULT` object which manages the configurations 
            of detector YOLO_Ultralytics.
        """
        self.cfg = cfg
        self.cpu_only = False
        if isinstance(self.cfg.device, str):
            if self.cfg.device.lower() == 'cpu':
                self.cpu_only = True
        ignore_this_logger("ultralytics")
        if "nas" in self.cfg.model_file:
            # YOLO NAS isn't stable yet :/
            from ultralytics import NAS
            self.model = NAS(self.cfg.model_file)
        else:
            from ultralytics import YOLO
            self.model = YOLO(self.cfg.model_file)
        from ultralytics.utils.plotting import Colors
        self.colors = Colors()
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], 
                         [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                         [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], 
                         [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    def __kpts__(self, img, kpts, radius=5, kpt_line=True):
        # Internal function
        h, w, c = img.shape
        shape = (h, w)
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim == 3
        kpt_line &= is_pose
        kpt_color = self.colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 
                                              0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        for i, k in enumerate(kpts):
            color_k = [int(x) for x in kpt_color[i]
                       ] if is_pose else self.colors(i)
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.5:
                        continue
                cv2.circle(img, (int(x_coord), int(y_coord)),
                           radius, color_k, -1, lineType=cv2.LINE_AA)
        if kpt_line:
            ndim = kpts.shape[-1]
            limb_color = self.colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 
                                                   0, 0, 16, 16, 16, 16, 16, 16, 16]]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < 0.5 or conf2 < 0.5:
                        continue
                if (pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or 
                    pos1[0] < 0 or pos1[1] < 0):
                    continue
                if (pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or 
                    pos2[0] < 0 or pos2[1] < 0):
                    continue
                cv2.line(img, pos1, pos2, [int(x) for x in limb_color[i]], 
                         thickness=2, lineType=cv2.LINE_AA)

    def detect(self, img, visual=True, classes=0, min_width_filter=35):
        """Detect general object with object's class filter :obj:`class_filter` 
        in a given cv :obj:`Mat` image.

        Parameters
        ----------
        img : Mat
            A cv :obj:`Mat` image.
        visual : bool, default=True
            An indication of whether to visualize the detected objects.
        classes : int, defualt=0
            Object's class filter, 0 means person only
        min_width_filter : int, default=35
            Mininum width filter of a detected object.

        Returns
        -------
        Mat
            A cv :obj:`Mat` image.
        list[ndarray[int, int, int, int], ...]
            A list of bounding box :code:`ndarray[x, y, width, height]`.
        list[ndarray[int, int, int, int], ...]
            A list of bounding box :code:`ndarray[x1, y1, x2, y2]`.
        list[tuple(int, int)]
            A lsit of represented 2D point :code:`(x, y)` of every detected object.
        list[ndarray[int, ...], ...]
            A list of :obj:`ndarray` of body keypoints. 
        float
            A list of the detection confidence of every detected object.
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
            classes=classes,
            show_boxes=self.cfg.show_boxes,
            device=self.cfg.device,
            max_det=int(self.cfg.max_det),
            line_width=self.cfg.line_width,
            verbose=False
        )
        if self.cpu_only:
            numpy_dets = dets[0].numpy()
        else:
            numpy_dets = dets[0].cuda().cpu().to("cpu").numpy()
        dt_boxes_xyxy = numpy_dets.boxes.xyxy
        dt_confidences = numpy_dets.boxes.conf
        dt_keypoints = dets[0].keypoints
        if dt_keypoints is not None:
            for box_xyxy, conf, kp in zip(dt_boxes_xyxy, dt_confidences, reversed(dt_keypoints)):
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
                        cv2.circle(img, (repspoint[0], repspoint[1]), 
                                   radius=5, color=(0, 0, 255), thickness=-1)
                        cv2.rectangle(img, (box_xyxy[0], box_xyxy[1]), 
                                      (box_xyxy[2], box_xyxy[3]), (255, 255, 0), 2)
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
                        cv2.circle(img, (repspoint[0], repspoint[1]), 
                                   radius=5, color=(0, 0, 255), thickness=-1)
                        cv2.rectangle(img, (box_xyxy[0], box_xyxy[1]), 
                                      (box_xyxy[2], box_xyxy[3]), (255, 255, 0), 2)
        return img, pboxes_xywh, pboxes_xyxy, repspoints, keypoints, confs

    def detectPeople(self, img, visual=True, min_width_filter=35, alt_repspoint=False, alt_repspoint_top=True):
        """Detect person(s) in a given cv :obj:`Mat` image.

        Parameters
        ----------
        img : Mat
            A cv :obj:`Mat` image.
        visual : bool, default=True
            An indication of whether to visualize the detected people.
        min_width_filter : int, default=35
            Mininum width filter of a detected person.
        alt_repspoint : bool, default=False
            An indication of whether to use the alternative :meth:`findRepspointBB`.
        alt_repspoint_top : bool, default=True
            A parameter passed to :obj:`prefer_top` of :meth:`findRepspointBB`.

        Returns
        -------
        list[Person, ...]
            A list of detected :class:`Person` object.
        Mat
            A cv :obj:`Mat` image.
        """
        numpy_dets = []
        people = []
        dets = self.model.predict(
            img,
            imgsz=int(self.cfg.imgsz),
            conf=float(self.cfg.conf),
            classes=0,
            show_boxes=self.cfg.show_boxes,
            device=self.cfg.device,
            max_det=int(self.cfg.max_det),
            line_width=self.cfg.line_width,
            verbose=False
        )
        if self.cpu_only:
            numpy_dets = dets[0].numpy()
        else:
            numpy_dets = dets[0].cuda().cpu().to("cpu").numpy()
        dt_boxes_xyxy = numpy_dets.boxes.xyxy
        dt_confidences = numpy_dets.boxes.conf
        dt_keypoints = dets[0].keypoints
        if dt_keypoints is not None:
            i = 0
            for box_xyxy, conf, kp in zip(dt_boxes_xyxy, dt_confidences, reversed(dt_keypoints)):
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
                        cv2.circle(img, (repspoint[0], repspoint[1]), 
                                   radius=5, color=(0, 0, 255), thickness=-1)
                        cv2.rectangle(img, (box_xyxy[0], box_xyxy[1]), 
                                      (box_xyxy[2], box_xyxy[3]), (255, 255, 0), 2)
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
                        cv2.circle(img, (repspoint[0], repspoint[1]), 
                                   radius=5, color=(0, 0, 255), thickness=-1)
                        cv2.rectangle(img, (box_xyxy[0], box_xyxy[1]), 
                                      (box_xyxy[2], box_xyxy[3]), (255, 255, 0), 2)
        return people, img
