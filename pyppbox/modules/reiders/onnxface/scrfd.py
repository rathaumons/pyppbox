# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
# Adapted from InsightFace's MIT-licensed SCRFD inference code:
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# See NOTICE and licenses/InsightFace-MIT.txt for upstream attribution and terms.
"""Single-resolution SCRFD inference using an already configured ONNX session.

Supports the float32 NCHW SCRFD exports with five landmarks (9 or 15 outputs).
The caller owns provider selection. This adapter neither downloads models nor
creates/reconfigures sessions, and does not depend on the InsightFace package.
"""

import cv2
import numpy as np


class SCRFD:
    def __init__(self, session, input_size=(320, 320), det_thresh=0.5):
        inputs = session.get_inputs()
        shape = inputs[0].shape if len(inputs) == 1 else []
        if (len(shape) != 4 or shape[1] != 3 or inputs[0].type != 'tensor(float)'
                or isinstance(shape[0], int) and shape[0] != 1):
            raise ValueError('SCRFD requires one NCHW float32 input with shape [1,3,H,W].')
        outputs = session.get_outputs()
        if len(outputs) not in (9, 15):
            raise ValueError('SCRFD requires 9 or 15 outputs, including five facial landmarks.')
        self.strides = (8, 16, 32) if len(outputs) == 9 else (8, 16, 32, 64, 128)
        self.num_anchors = 2 if len(outputs) == 9 else 1
        # Honor a fixed-size ONNX input, as the upstream adapter does.
        self.input_size = tuple(shape[axis] if isinstance(shape[axis], int) and shape[axis] > 0
                                else size for axis, size in zip((3, 2), input_size))
        if len(self.input_size) != 2 or any(size <= 0 or size % self.strides[-1]
                                          for size in self.input_size):
            raise ValueError('SCRFD input dimensions must be positive multiples of its largest stride.')
        self.session = session
        self.input_name = inputs[0].name
        self.output_names = [output.name for output in outputs]
        self.det_thresh = det_thresh
        self.nms_thresh = 0.4
        self.center_cache = {}

    @staticmethod
    def _output(value, count, columns):
        value = np.asarray(value)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.shape != (count, columns) or not np.isfinite(value).all():
            raise ValueError('SCRFD returned invalid scores, boxes, or landmarks.')
        return value

    def _forward(self, image):
        blob = cv2.dnn.blobFromImage(image, 1.0 / 128.0, self.input_size,
                                    (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        levels = len(self.strides)
        if len(outputs) != levels * 3:
            raise ValueError('SCRFD returned an unexpected number of outputs.')
        scores_list, boxes_list, points_list = [], [], []
        for index, stride in enumerate(self.strides):
            height, width = image.shape[0] // stride, image.shape[1] // stride
            if stride not in self.center_cache:
                centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                centers = (centers * stride).reshape(-1, 2)
                self.center_cache[stride] = np.repeat(centers, self.num_anchors, axis=0)
            centers = self.center_cache[stride]
            count = len(centers)
            scores = self._output(outputs[index], count, 1)
            distances = self._output(outputs[index + levels], count, 4) * stride
            offsets = self._output(outputs[index + 2 * levels], count, 10) * stride
            selected = np.where(scores >= self.det_thresh)[0]
            boxes = np.stack([centers[:, 0] - distances[:, 0], centers[:, 1] - distances[:, 1],
                              centers[:, 0] + distances[:, 2], centers[:, 1] + distances[:, 3]], axis=-1)
            points = centers[:, None, :] + offsets.reshape(-1, 5, 2)
            scores_list.append(scores[selected])
            boxes_list.append(boxes[selected])
            points_list.append(points[selected])
        return np.vstack(scores_list), np.vstack(boxes_list), np.vstack(points_list)

    def detect(self, image):
        if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
                or not image.size or image.dtype != np.uint8):
            raise ValueError('SCRFD expects a nonempty uint8 BGR image.')
        width, height = self.input_size
        image_ratio = float(image.shape[0]) / image.shape[1]
        if image_ratio > float(height) / width:
            new_height = height
            new_width = max(1, int(new_height / image_ratio))
        else:
            new_width = width
            new_height = max(1, int(new_width * image_ratio))
        scale = float(new_height) / image.shape[0]
        padded = np.zeros((height, width, 3), dtype=np.uint8)
        padded[:new_height, :new_width] = cv2.resize(image, (new_width, new_height))
        scores, boxes, points = self._forward(padded)
        boxes = boxes / scale
        points = points / scale
        detections = np.hstack((boxes, scores)).astype(np.float32, copy=False)
        order = np.argsort(-detections[:, 4], kind='stable')
        detections, points = detections[order], points[order]
        keep = self._nms(detections)
        return detections[keep], points[keep]

    def _nms(self, detections):
        x1, y1, x2, y2, scores = detections.T
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(-scores, kind='stable')
        keep = []
        while order.size:
            index = order[0]
            keep.append(index)
            rest = order[1:]
            width = np.maximum(0.0, np.minimum(x2[index], x2[rest]) - np.maximum(x1[index], x1[rest]) + 1)
            height = np.maximum(0.0, np.minimum(y2[index], y2[rest]) - np.maximum(y1[index], y1[rest]) + 1)
            intersection = width * height
            overlap = intersection / (areas[index] + areas[rest] - intersection)
            order = rest[np.where(overlap <= self.nms_thresh)[0]]
        return keep
