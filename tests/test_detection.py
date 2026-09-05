"""YOLO wrapper regressions B02/B03/B04 without downloaded weights."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from pyppbox.modules.detectors.yoloult import MyYOLOULT
from pyppbox.ppb.mt import MT


class DetectionTests(unittest.TestCase):
    def detector(self, device='cpu', pose=True, empty=False):
        detector = MyYOLOULT.__new__(MyYOLOULT)
        detector.cfg = SimpleNamespace(imgsz=640, conf=.4, iou=.2, show_boxes=True,
                                       device=device, max_det=50, repspoint_calibration=.25)
        detector.cpu_only = device == 'cpu'
        boxes = np.array([[0, 0, 20, 40], [40, 0, 45, 40], [80, 0, 100, 40]], dtype=float)
        confidences = np.array([.91, .82, .73])
        if empty:
            boxes, confidences = boxes[:0], confidences[:0]
        keypoints = [SimpleNamespace(data=np.full((1, 17, 3), marker, dtype=float))
                     for marker in ([10, 50, 90] if not empty else [])] if pose else None
        cpu = SimpleNamespace(boxes=SimpleNamespace(xyxy=boxes, conf=confidences), keypoints=keypoints)
        cpu.numpy = Mock(return_value=cpu)
        result = Mock()
        result.cpu.return_value = cpu
        result.cuda.side_effect = AssertionError('CUDA transfer is forbidden')
        result.numpy.return_value = cpu
        result.keypoints = keypoints
        detector.model = Mock()
        detector.model.predict.return_value = [result]
        detector.__kpts__ = Mock()
        return detector, result, cpu

    def test_pose_alignment_width_filter_and_devices(self):
        for device in ('cpu', 'cuda:1', 'mps', 1):
            for method in ('detect', 'detectPeople'):
                with self.subTest(device=device, method=method):
                    detector, result, cpu = self.detector(device)
                    frame = np.zeros((60, 120, 3), dtype=np.uint8)
                    output = getattr(detector, method)(frame, visual=False)
                    if method == 'detect':
                        self.assertEqual(len(output), 6)
                        self.assertIs(output[0], frame)
                        boxes, keypoints, confidences = output[2], output[4], output[5]
                    else:
                        self.assertEqual(len(output), 2)
                        self.assertIs(output[1], frame)
                        boxes = [p.box_xyxy for p in output[0]]
                        keypoints = [p.keypoints for p in output[0]]
                        confidences = [p.det_conf for p in output[0]]
                    self.assertEqual([int(box[0]) for box in boxes], [0, 80])
                    self.assertEqual([float(kp[0, 0]) for kp in keypoints], [10, 90])
                    np.testing.assert_allclose(confidences, [.91, .73])
                    result.cpu.assert_called_once_with()
                    result.cuda.assert_not_called()
                    result.numpy.assert_not_called()
                    cpu.numpy.assert_called_once_with()
                    self.assertEqual(detector.model.predict.call_args.kwargs['iou'], .2)
                    self.assertEqual(detector.model.predict.call_args.kwargs['device'], device)

    def test_pose_drawing_uses_aligned_keypoints(self):
        for method in ('detect', 'detectPeople'):
            detector, _, _ = self.detector()
            getattr(detector, method)(np.zeros((60, 120, 3), dtype=np.uint8), visual=True)
            self.assertEqual([float(call.args[1][0, 0]) for call in detector.__kpts__.call_args_list], [10, 90])

    def test_iou_configuration_reaches_predictor(self):
        for method in ('detect', 'detectPeople'):
            detector, _, _ = self.detector(pose=False)
            getattr(detector, method)(np.zeros((60, 120, 3), dtype=np.uint8), visual=False)
            self.assertEqual(detector.model.predict.call_args.kwargs.get('iou'), .2)

    def test_empty_and_nonpose_results(self):
        for pose, empty in ((False, False), (False, True), (True, True)):
            for method in ('detect', 'detectPeople'):
                with self.subTest(pose=pose, empty=empty, method=method):
                    detector, _, _ = self.detector(pose=pose, empty=empty)
                    result = getattr(detector, method)(np.zeros((60, 120, 3), dtype=np.uint8), visual=False)
                    count = len(result[1]) if method == 'detect' else len(result[0])
                    self.assertEqual(count, 0 if empty else 2)
                    if method == 'detect':
                        self.assertEqual(result[4], [])

    def test_public_detector_facade(self):
        detector, _, _ = self.detector()
        mt = MT()
        detector.cfg.dt_name = 'YOLO_Ultralytics'
        mt.__dt__, mt.__dt_cfg__, mt.__dt_is_set__ = detector, detector.cfg, True
        people, _ = mt.detectPeople(np.zeros((60, 120, 3), dtype=np.uint8), img_is_mat=True)
        self.assertEqual([float(person.keypoints[0, 0]) for person in people], [10, 90])


if __name__ == '__main__':
    unittest.main()
