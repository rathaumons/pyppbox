# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""YOLO ordering, integer-boundary filtering, predictor calls, and pose setup."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pyppbox.modules.detectors.yoloult import MyYOLOULT
import test_detection


class YOLOProcessingTests(unittest.TestCase):
    def test_backend_order_and_integer_width_boundary_keep_pose_and_scores_aligned(self):
        for method in ('detect', 'detectPeople'):
            with self.subTest(method=method):
                detector, _, cpu = test_detection.DetectionTests().detector()
                cpu.boxes.xyxy = np.array([[80.9, .9, 95.8, 41.9], [10.9, .9, 25.8, 41.9],
                                          [40.1, 0., 54.99, 40.]])
                cpu.boxes.conf = np.array([.11, .99, .6])
                image = np.zeros((60, 120, 3), dtype=np.uint8)
                output = getattr(detector, method)(image, visual=False)
                if method == 'detect':
                    boxes, points, keypoints, scores = output[2:]
                else:
                    people = output[0]
                    boxes = [p.box_xyxy for p in people]
                    points = [p.repspoint for p in people]
                    keypoints = [p.keypoints for p in people]
                    scores = [p.det_conf for p in people]
                    self.assertEqual([(p.init_id, p.cid) for p in people], [(0, 0), (1, 1)])
                self.assertEqual([box.tolist() for box in boxes], [[80, 0, 95, 41], [10, 0, 25, 41]])
                self.assertEqual([point[0] for point in points], [87, 17])
                self.assertEqual([float(kp[0, 0]) for kp in keypoints], [10., 50.])
                np.testing.assert_array_equal(scores, [.11, .99])
                self.assertFalse(image.any())
                self.assertEqual(detector.model.predict.call_count, 1)

    def test_generic_classes_and_people_representative_point_options_are_preserved(self):
        detector, _, _ = test_detection.DetectionTests().detector(pose=False)
        image = np.zeros((60, 120, 3), dtype=np.uint8)
        for classes in ([1, 3], None):
            detector.detect(image, visual=False, classes=classes)
            self.assertEqual(detector.model.predict.call_args.kwargs, dict(
                imgsz=640, conf=.4, iou=.2, classes=classes, show_boxes=True,
                device='cpu', max_det=50, verbose=False))
        for alternative, top, expected_y in ((False, True, 10), (True, True, 0), (True, False, 40)):
            people, returned = detector.detectPeople(image, visual=False, alt_repspoint=alternative,
                                                     alt_repspoint_top=top)
            self.assertIs(returned, image)
            self.assertEqual([p.repspoint[1] for p in people], [expected_y, expected_y])
            self.assertEqual(detector.model.predict.call_args.kwargs['classes'], 0)
            self.assertEqual([p.keypoints for p in people], [[], []])
            self.assertIsNot(people[0].keypoints, people[1].keypoints)

    def test_unused_result_fields_need_not_be_transferred(self):
        detector, result, _ = test_detection.DetectionTests().detector(pose=False)
        # Segmentation/other auxiliary data is irrelevant to these box-only APIs.
        # A provider unable to transfer those fields must not break detection.
        result.cpu.side_effect = AssertionError('Whole-result transfer touches unused data')
        result.numpy.side_effect = AssertionError('Whole-result conversion touches unused data')
        for method in ('detect', 'detectPeople'):
            output = getattr(detector, method)(np.zeros((60, 120, 3), dtype=np.uint8), visual=False)
            self.assertEqual(len(output[1] if method == 'detect' else output[0]), 2)

    def test_pose_task_metadata_initializes_drawing_for_custom_checkpoint_names(self):
        colors = SimpleNamespace(pose_palette=np.zeros((20, 3), dtype=np.uint8))
        cfg = SimpleNamespace(model_file='custom-checkpoint.pt', device='cpu')
        predictor = SimpleNamespace(task='pose')
        module = SimpleNamespace(YOLO=Mock(return_value=predictor))
        with patch.dict('sys.modules', {'ultralytics': module,
                'ultralytics.utils.plotting': SimpleNamespace(Colors=Mock(return_value=colors))}):
            detector = MyYOLOULT(cfg)
        self.assertIs(detector.colors, colors)
        self.assertEqual(detector.kpt_color.shape, (17, 3))
        self.assertEqual(len(detector.skeleton), 19)

    def test_legacy_pose_filename_remains_a_supported_fallback(self):
        cfg = SimpleNamespace(model_file='legacy-pose.pt', device='cpu')
        colors = SimpleNamespace(pose_palette=np.zeros((20, 3), dtype=np.uint8))
        with patch.dict('sys.modules', {
                'ultralytics': SimpleNamespace(YOLO=Mock(return_value=SimpleNamespace())),
                'ultralytics.utils.plotting': SimpleNamespace(Colors=Mock(return_value=colors))}):
            detector = MyYOLOULT(cfg)
        self.assertIs(detector.colors, colors)


if __name__ == '__main__':
    unittest.main()
