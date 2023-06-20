"""
    pyppbox: Toolbox for people detecting, tracking, and re-identifying.
    Copyright (C) 2022 UMONS-Numediart

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import cv2


class MyYOLOCV(object):

	def __init__(self, cfg):
		self.nms_threshold = cfg.nms_threshold
		self.conf_threshold = float(cfg.conf_threshold)
		self.classes_path = cfg.class_file
		self.model_image_size = cfg.model_resolution
		net = cv2.dnn.readNet(cfg.model_weights, cfg.model_cfg_file)
		net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
		self.model = cv2.dnn_DetectionModel(net)
		self.model.setInputParams(size=self.model_image_size, scale=1/255.0)

	def to_xyxy(self, box_xywh):
		ret = box_xywh.copy()
		ret[2:] += ret[:2]
		return ret

	def calibrate(self, y1, y2, weight):
		start = min(y1, y2)
		dist = abs(y1 - y2)
		return int(start + weight*dist)


	def detectFrame(self, frame, visual=True, class_filter=0, width_filter_min=35, repspoint_callibration=0):
		pboxes_xywh = []
		pboxes_xyxy = []
		repspoints = []
		classes, confidences, boxes = self.model.detect(frame, confThreshold=self.conf_threshold, nmsThreshold=self.nms_threshold)
		if len(classes) > 0:
			for classId, box_xywh in zip(classes.flatten(), boxes):
				if classId == class_filter and box_xywh[2] >= width_filter_min:
					pboxes_xywh.append(box_xywh)
					box_xyxy = self.to_xyxy(box_xywh)
					pboxes_xyxy.append(box_xyxy)
					repspoint = [int((box_xyxy[0] + box_xyxy[2]) / 2), self.calibrate(box_xyxy[1], box_xyxy[3], repspoint_callibration)]

					repspoints.append(repspoint)
					if visual:
						cv2.circle(frame, (repspoint[0], repspoint[1]), radius=5, color=(0, 0, 255), thickness=-1)
						cv2.rectangle(frame, (int(box_xyxy[0]), int(box_xyxy[1])), (int(box_xyxy[2]), int(box_xyxy[3])), (255, 255, 0), 2)
		return frame, pboxes_xywh, pboxes_xyxy, repspoints
