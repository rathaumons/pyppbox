"""SCRFD decoding, preprocessing, model validation, and session ownership."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from pyppbox.modules.reiders.onnxface.scrfd import SCRFD


class SCRFDTests(unittest.TestCase):
    def session(self, size=32, levels=3, batched=False, static=False):
        strides = (8, 16, 32) if levels == 3 else (8, 16, 32, 64, 128)
        anchors = 2 if levels == 3 else 1
        outputs = [np.zeros(((size // stride) ** 2 * anchors, columns), np.float32)
                   for columns in (1, 4, 10) for stride in strides]
        if batched:
            outputs = [value[None] for value in outputs]
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(
            name='image', shape=[1, 3, size, size] if static else [1, 3, 'height', 'width'],
            type='tensor(float)')]
        session.get_outputs.return_value = [SimpleNamespace(name=str(index), shape=value.shape)
                                            for index, value in enumerate(outputs)]
        session.run.return_value = outputs
        return session, outputs

    def test_decode_resize_landmarks_and_nms_for_batched_and_unbatched_models(self):
        for batched in (False, True):
            with self.subTest(batched=batched):
                session, outputs = self.session(batched=batched)
                values = [value[0] for value in outputs] if batched else outputs
                # Two anchors at (8, 8) predict the same box; NMS keeps .9.
                values[0][10:12, 0] = [.9, .8]
                values[3][10:12] = [1, .5, 2, 1.5]
                values[6][10:12] = [0, 0, 1, 0, .5, .5, 0, 1, 1, 1]
                detector = SCRFD(session, input_size=(32, 32))
                image = np.full((64, 32, 3), [10, 20, 30], np.uint8)
                boxes, landmarks = detector.detect(image)
                np.testing.assert_allclose(boxes, [[0, 8, 48, 40, .9]])
                np.testing.assert_array_equal(landmarks, [[[16, 16], [32, 16], [24, 24],
                                                         [16, 32], [32, 32]]])
                blob = session.run.call_args.args[1]['image']
                self.assertEqual(blob.shape, (1, 3, 32, 32))
                np.testing.assert_allclose(blob[0, :, 0, 0], (np.array([30, 20, 10]) - 127.5) / 128)
                np.testing.assert_allclose(blob[0, :, :, 16:], -127.5 / 128)
                self.assertIs(detector.session, session)
                session.set_providers.assert_not_called()

    def test_five_levels_and_fixed_model_size(self):
        session, outputs = self.session(size=128, levels=5, static=True)
        outputs[4][0] = .9
        outputs[9][0] = 1
        detector = SCRFD(session, input_size=(320, 320))
        self.assertEqual(detector.input_size, (128, 128))
        boxes, landmarks = detector.detect(np.zeros((128, 128, 3), np.uint8))
        np.testing.assert_allclose(boxes, [[-128, -128, 128, 128, .9]])
        np.testing.assert_array_equal(landmarks, np.zeros((1, 5, 2)))

    def test_threshold_empty_result_and_stable_ties(self):
        session, outputs = self.session()
        outputs[0][10:12] = .5
        outputs[3][10:12] = 1
        outputs[6][11] = 1
        detector = SCRFD(session, input_size=(32, 32), det_thresh=.5)
        image = np.zeros((32, 32, 3), np.uint8)
        boxes, landmarks = detector.detect(image)
        self.assertEqual(boxes.shape, (1, 5))
        np.testing.assert_array_equal(landmarks, np.full((1, 5, 2), 8))
        detector.det_thresh = .51
        boxes, landmarks = detector.detect(image)
        self.assertEqual(boxes.shape, (0, 5))
        self.assertEqual(landmarks.shape, (0, 5, 2))
        self.assertEqual(boxes.dtype, np.float32)

    def test_rejects_unsupported_model_contracts(self):
        for change in ('dtype', 'channels', 'batch', 'outputs', 'resolution'):
            with self.subTest(change=change):
                session, _ = self.session()
                input_size = (32, 32)
                if change == 'dtype':
                    session.get_inputs.return_value[0].type = 'tensor(uint8)'
                elif change == 'channels':
                    session.get_inputs.return_value[0].shape[1] = 1
                elif change == 'batch':
                    session.get_inputs.return_value[0].shape[0] = 2
                elif change == 'outputs':
                    session.get_outputs.return_value = session.get_outputs.return_value[:6]
                else:
                    input_size = (31, 32)
                with self.assertRaises(ValueError):
                    SCRFD(session, input_size=input_size)

    def test_rejects_malformed_outputs(self):
        for change in ('shape', 'nan', 'count'):
            with self.subTest(change=change):
                session, outputs = self.session()
                detector = SCRFD(session, input_size=(32, 32))
                if change == 'shape':
                    outputs[3] = outputs[3][:-1]
                elif change == 'nan':
                    outputs[6][0, 0] = np.nan
                else:
                    session.run.return_value = outputs[:-1]
                with self.assertRaisesRegex(ValueError, 'SCRFD returned'):
                    detector.detect(np.zeros((32, 32, 3), np.uint8))


if __name__ == '__main__':
    unittest.main()
