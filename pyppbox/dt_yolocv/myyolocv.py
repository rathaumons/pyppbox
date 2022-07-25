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


from __future__ import division, print_function, absolute_import

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

	def to_tlbr(self, box):
		ret = box.copy()
		ret[2:] += ret[:2]
		return ret

	def calibrate(self, y1, y2, weight):
		start = min(y1, y2)
		dist = abs(y1 - y2)
		return int(start + weight*dist)


	def detectFrame(self, frame, visual=True, class_filter=0, width_filter_min=35, repspoint_callibration=0):
		pboxes = []
		pboxes_tlbr = []
		repspoints = []
		classes, confidences, boxes = self.model.detect(frame, confThreshold=self.conf_threshold, nmsThreshold=self.nms_threshold)
		if len(classes) > 0:
			for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
				[x, y, w, h] = box
				if classId == class_filter and w > width_filter_min:
					pboxes.append(box)
					rect = self.to_tlbr(box)
					pboxes_tlbr.append(rect)
					repspoint = [int((rect[0] + rect[2]) / 2), self.calibrate(rect[1], rect[3], repspoint_callibration)]

					repspoints.append(repspoint)
					if visual:
						cv2.circle(frame, (repspoint[0], repspoint[1]), radius=5, color=(0, 0, 255), thickness=-1)
						cv2.rectangle(frame, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (255, 255, 0), 2)
		return frame, pboxes, pboxes_tlbr, repspoints
