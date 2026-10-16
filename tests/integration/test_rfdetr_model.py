"""Opt-in tests using published Nano weights and actual CPU/CUDA inference."""
import os
from pathlib import Path
import time
import unittest

import cv2
import numpy as np

from pyppbox.ppb.mt import MT


@unittest.skipUnless(os.environ.get('PYPPBOX_RUN_RFDETR') == '1', 'Set PYPPBOX_RUN_RFDETR=1 for model downloads/inference.')
class RFDETRModelTests(unittest.TestCase):
    def run_model(self, device):
        import torch
        video = Path(__file__).resolve().parents[2] / 'examples/data/gta.mp4'
        capture = cv2.VideoCapture(str(video))
        try:
            ok, frame = capture.read()
            self.assertTrue(ok, f'Cannot read {video}')
        finally:
            capture.release()
        original = frame.copy()
        pipeline = MT()
        pipeline.setMainDetector(dict(dt_name='RF_DETR', model_size='nano', model_file='', device=device))
        pipeline.setMainTracker('SORT')
        pipeline.setMainReIDer('None')
        backend = pipeline.__dt__.model
        native = backend.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=.4, include_source_image=False)
        mask = native.class_id == 1
        self.assertTrue(mask.any())
        self.assertTrue(all(name == 'person' for name in native.data['class_name'][mask]))
        counts, elapsed, track_ids = [], [], []
        for index in range(6):
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            started = time.perf_counter()
            people, returned = pipeline.detectPeople(frame, img_is_mat=True)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            if index:
                elapsed.append(time.perf_counter() - started)
            self.assertIs(returned, frame)
            self.assertTrue(people)
            self.assertTrue(all(0 <= person.det_conf <= 1 and not len(person.keypoints) for person in people))
            people = pipeline.trackPeople(frame, people, img_is_mat=True)
            counts.append(len(people))
            track_ids.append([person.cid for person in people])
        np.testing.assert_array_equal(frame, original)
        self.assertTrue(all(ids == track_ids[0] for ids in track_ids))
        print(f'RF-DETR Nano {device}: {counts}, stable SORT IDs {track_ids[0]}, '
              f'mean warmed detection {1000 * np.mean(elapsed):.1f} ms/frame', flush=True)
        artifact_dir = os.environ.get('PYPPBOX_RFDETR_RESULT_DIR')
        if artifact_dir:
            output = Path(artifact_dir)
            output.mkdir(parents=True, exist_ok=True)
            _, annotated = pipeline.detectPeople(frame.copy(), img_is_mat=True, visual=True)
            self.assertTrue(cv2.imwrite(str(output / ('rfdetr-nano-' + device.replace(':', '-') + '.jpg')), annotated))

    def test_cpu(self):
        self.run_model('cpu')

    def test_cuda(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA is unavailable.')
        self.run_model('cuda:0')


if __name__ == '__main__':
    unittest.main()
