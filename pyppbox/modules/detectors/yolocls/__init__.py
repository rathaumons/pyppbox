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
from pyppbox.utils.commontools import to_xyxy


class MyYOLOCLS(object):

    """Class used as a custom layer or interface for interacting with detector module 
    YOLO_Classic which uses :code:`.weights` model.
    
    Attributes
    ----------
    cfg : DCFGYOLOCLS
        A :class:`DCFGYOLOCLS` object which manages the configurations of detector 
        YOLO_Classic.
    model: cv::dnn::DetectionModel
        A detection model object of OpenCV's deep learning network.
    """

    def __init__(self, cfg):
        """Initialize according to the given configuration :obj:`cfg` 
        as :class:`DCFGYOLOCLS` object.

        Parameters
        ----------
        cfg : DCFGYOLOCLS
            A :class:`DCFGYOLOCLS` object which manages the configurations of detector 
            YOLO_Classic.
        """
        self.cfg = cfg
        net = cv2.dnn.readNet(cfg.model_weights, cfg.model_cfg_file)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=cfg.model_resolution, scale=1/255.0)

    def detect(self, img, visual=True, class_filter=0, min_width_filter=35):
        """Detect general object with object's class filter :obj:`class_filter` in a 
        given cv :obj:`Mat` image.

        Parameters
        ----------
        img : Mat
            A cv :obj:`Mat` image.
        visual : bool, default=True
            An indication of whether to visualize the detected objects.
        class_filter : int, defualt=0
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
        float
            A list of the detection confidence of every detected object.
        """
        pboxes_xywh = []
        pboxes_xyxy = []
        repspoints = []
        confs = []
        classes, confidences, boxes = self.model.detect(img, 
                                                        confThreshold=float(self.cfg.conf), 
                                                        nmsThreshold=float(self.cfg.nms))
        if len(classes) > 0:
            for class_id, conf, box_xywh in zip(classes.flatten(), confidences, boxes):
                if class_id == class_filter and box_xywh[2] >= min_width_filter:
                    box_xywh = box_xywh.astype(int)
                    pboxes_xywh.append(box_xywh)
                    box_xyxy = to_xyxy(box_xywh)
                    pboxes_xyxy.append(box_xyxy)
                    repspoint = findRepspoint(box_xyxy, self.cfg.repspoint_calibration)
                    repspoints.append(repspoint)
                    confs.append(float(conf))
                    if visual:
                        cv2.circle(img, (repspoint[0], repspoint[1]), radius=5, 
                                   color=(0, 0, 255), thickness=-1)
                        cv2.rectangle(img, (box_xyxy[0], box_xyxy[1]), 
                                      (box_xyxy[2], box_xyxy[3]), (255, 255, 0), 2)
        return img, pboxes_xywh, pboxes_xyxy, repspoints, confs

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
        people = []
        classes, confidences, boxes = self.model.detect(
            img, 
            confThreshold=float(self.cfg.conf), 
            nmsThreshold=float(self.cfg.nms)
        )
        if len(classes) > 0:
            i = 0
            for class_id, conf, box_xywh in zip(classes.flatten(), confidences, boxes):
                if class_id == 0 and box_xywh[2] >= min_width_filter:
                    box_xywh = box_xywh.astype(int)
                    box_xyxy = to_xyxy(box_xywh)
                    if alt_repspoint: repspoint = findRepspointBB(box_xyxy, prefer_top=alt_repspoint_top)
                    else: repspoint = findRepspoint(box_xyxy, self.cfg.repspoint_calibration)
                    people.append(Person(i, i, box_xywh=box_xywh, box_xyxy=box_xyxy, 
                                         repspoint=repspoint, det_conf=float(conf)))
                    i += 1
                    if visual:
                        cv2.circle(img, (repspoint[0], repspoint[1]), radius=5, 
                                   color=(0, 0, 255), thickness=-1)
                        cv2.rectangle(img, (box_xyxy[0], box_xyxy[1]), 
                                      (box_xyxy[2], box_xyxy[3]), (255, 255, 0), 2)
        return people, img
