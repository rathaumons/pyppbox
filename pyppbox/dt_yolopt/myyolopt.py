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
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Colors


class MyYOLOPT(object):

	def __init__(self, cfg):
		self.conf = float(cfg.conf)
		self.iou = float(cfg.iou)
		self.imgsz = int(cfg.imgsz)
		self.classes = cfg.classes
		self.boxes = cfg.boxes
		self.device = cfg.device
		self.max_det = int(cfg.max_det)
		self.hide_labels = cfg.hide_labels
		self.hide_conf = cfg.hide_conf
		self.line_width = cfg.line_width
		self.visualize = cfg.visualize
		self.model_file = cfg.model_file
		self.repspoint_callibration = cfg.repspoint_callibration
		self.model = YOLO(self.model_file)
		self.colors = Colors()
		self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                		 [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
		self.limb_color = self.colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
		self.kpt_color = self.colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

	def to_xywh(self, box_xyxy):
		box_xywh = box_xyxy.copy()
		box_xywh[2] = box_xywh[2] - box_xywh[0]
		box_xywh[3] = box_xywh[3] - box_xywh[1]
		return box_xywh
	
	def to_xyxy(self, box_xywh):
		ret = box_xywh.copy()
		ret[2:] += ret[:2]
		return ret

	def calibrate(self, y1, y2, weight):
		start = min(y1, y2)
		dist = abs(y1 - y2)
		return int(start + weight*dist)

	def kpts_lite(self, frame, kpts, radius=5):
		h, w, c = frame.shape
		shape = (h, w)
		for i, k in enumerate(kpts):
			x_coord, y_coord = k[0], k[1]
			if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
				if len(k) == 3:
					conf = k[2]
					if conf < 0.5:
						continue
				cv2.circle(frame, (int(x_coord), int(y_coord)), radius, (255, 16, 240), -1, lineType=cv2.LINE_AA)

	def kpts(self, frame, kpts, radius=5, kpt_line=True):
		h, w, c = frame.shape
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
				cv2.circle(frame, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)
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
				if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
					continue
				if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
					continue
				cv2.line(frame, pos1, pos2, [int(x) for x in self.limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

	def detectFrame(self, frame, visual=True, width_filter_min=35):
		pboxes_xyxy = []
		pboxes_xywh = []
		repspoints = []
		keypoints = []
		dets = self.model.predict(
			frame, 
			imgsz=self.imgsz, 
			conf=self.conf, 
			classes=self.classes, 
			boxes=self.boxes, 
			device=self.device, 
			max_det=self.max_det, 
			# hide_labels=self.hide_labels,
			# hide_conf=self.hide_conf,
			# visualize=self.visualize,
			line_width=self.line_width
		)
		numpy_dets = dets[0].cuda().cpu().to("cpu").numpy()
		dt_boxes_xyxy = numpy_dets.boxes.xyxy
		# dt_confidences = numpy_dets.boxes.conf
		dt_keypoints = dets[0].keypoints
		if dt_keypoints is not None:
			for box_xyxy, kp in zip(dt_boxes_xyxy, reversed(dt_keypoints)):
				box_xywh = self.to_xywh(box_xyxy)
				if box_xywh[2] >= width_filter_min:
					pboxes_xywh.append(box_xywh)
					pboxes_xyxy.append(box_xyxy)
					repspoint = [int((box_xyxy[0] + box_xyxy[2]) / 2), self.calibrate(box_xyxy[1], box_xyxy[3], self.repspoint_callibration)]
					repspoints.append(repspoint)
					keypoint = kp.data[0]
					keypoints.append(keypoint)
					if visual:
						cv2.circle(frame, (repspoint[0], repspoint[1]), radius=5, color=(0, 0, 255), thickness=-1)
						cv2.rectangle(frame, (int(box_xyxy[0]), int(box_xyxy[1])), (int(box_xyxy[2]), int(box_xyxy[3])), (255, 255, 0), 2)
						# self.kpts_lite(frame, keypoint)
						self.kpts(frame, keypoint, kpt_line=True)
		elif len(dt_boxes_xyxy) > 0:
			for box_xyxy in dt_boxes_xyxy:
				box_xywh = self.to_xywh(box_xyxy)
				if box_xywh[2] >= width_filter_min:
					pboxes_xywh.append(box_xywh)
					pboxes_xyxy.append(box_xyxy)
					repspoint = [int((box_xyxy[0] + box_xyxy[2]) / 2), self.calibrate(box_xyxy[1], box_xyxy[3], self.repspoint_callibration)]
					repspoints.append(repspoint)
					if visual:
						cv2.circle(frame, (repspoint[0], repspoint[1]), radius=5, color=(0, 0, 255), thickness=-1)
						cv2.rectangle(frame, (int(box_xyxy[0]), int(box_xyxy[1])), (int(box_xyxy[2]), int(box_xyxy[3])), (255, 255, 0), 2)
		return frame, pboxes_xywh, pboxes_xyxy, repspoints, keypoints
	