"""Run separately with the installed detector backend and PyTorch, without weights.

python -m unittest discover -s tests/integration -v
"""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from ultralytics.engine.results import Results

from pyppbox.modules.detectors.yoloult import MyYOLOULT


class RealResultsTests(unittest.TestCase):
    def check_device(self, device):
        frame = np.zeros((60, 120, 3), dtype=np.uint8)
        for pose, empty in ((True, False), (False, False), (True, True), (False, True)):
            boxes = torch.tensor([[0, 0, 20, 40, .9, 0], [80, 0, 100, 40, .8, 0]], device=device)
            keypoints = torch.ones((2, 17, 3), device=device)
            keypoints[0, :, :2], keypoints[1, :, :2] = 10, 40
            if empty:
                boxes, keypoints = boxes[:0], keypoints[:0]
            result = Results(frame, path='synthetic.png', names={0: 'person'}, boxes=boxes,
                             keypoints=keypoints if pose else None)
            cfg = SimpleNamespace(imgsz=640, conf=.4, iou=.2, show_boxes=True,
                                  device=device, max_det=50, repspoint_calibration=.25,
                                  model_file='synthetic-pose.pt' if pose else 'synthetic.pt')
            predictor = Mock()
            predictor.predict.return_value = [result]
            with patch('ultralytics.YOLO', return_value=predictor):
                detector = MyYOLOULT(cfg)
            for method in ('detect', 'detectPeople'):
                with self.subTest(device=device, pose=pose, empty=empty, method=method):
                    output = getattr(detector, method)(frame.copy(), visual=True)
                    if method == 'detect':
                        boxes_out, keypoints_out = output[2], output[4]
                    else:
                        boxes_out = [p.box_xyxy for p in output[0]]
                        keypoints_out = [p.keypoints for p in output[0]]
                    self.assertEqual(len(boxes_out), 0 if empty else 2)
                    self.assertTrue(all(isinstance(box, np.ndarray) for box in boxes_out))
                    if pose:
                        self.assertEqual([float(kp[0, 0]) for kp in keypoints_out], [] if empty else [10, 40])
                        self.assertTrue(all(isinstance(kp, torch.Tensor) and kp.device.type == 'cpu'
                                            for kp in keypoints_out))
                    self.assertEqual(predictor.predict.call_args.kwargs['iou'], .2)

    def test_cpu_results(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device unavailable')
    def test_cuda_results(self):
        for index in range(torch.cuda.device_count()):
            self.check_device(f'cuda:{index}')


if __name__ == '__main__':
    unittest.main()
