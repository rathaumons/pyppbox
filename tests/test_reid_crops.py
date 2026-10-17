"""ReID crop pixels, boundary handling, input isolation, and count semantics."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from pyppbox.ppb.mt import MT
from pyppbox.utils.persontools import Person


class ReIDCropTests(unittest.TestCase):
    def pipeline(self, backend, recognize):
        pipeline = MT()
        pipeline.__ri_is_set__ = True
        pipeline.__ri_cfg__ = SimpleNamespace(ri_name=backend, model_wh=(8, 16),
                                             yl_h_calibration=[-5, 3], yl_w_calibration=[-4, 6])
        pipeline.__ri__ = SimpleNamespace(auto_load=True, recognize=Mock(side_effect=recognize))
        return pipeline

    def test_valid_crops_match_legacy_pixels_in_normal_and_duplicate_passes(self):
        frame = np.random.default_rng(17).integers(0, 256, (80, 100, 3), dtype=np.uint8)
        for backend in ('Torchreid', 'FaceNet'):
            for deduplicate in (False, True):
                with self.subTest(backend=backend, deduplicate=deduplicate):
                    captured = []

                    def recognize(crop, is_bgr):
                        self.assertTrue(is_bgr)
                        captured.append(crop.copy())
                        return 'same-id', 87.25

                    pipeline = self.pipeline(backend, recognize)
                    people = [Person(0, 0, box_xyxy=np.array([2, 3, 22, 33]), repspoint=(15, 20)),
                              Person(1, 1, box_xyxy=[40, 10, 60, 40], repspoint=(50, 30))]
                    expected = ([cv2.resize(frame[3:33, 2:22], (8, 16)),
                                 cv2.resize(frame[10:40, 40:60], (8, 16))]
                                if backend == 'Torchreid' else
                                [frame[15:23, 11:21], frame[25:33, 46:56]])
                    result, counts = pipeline.reidPeople(frame, people, img_is_mat=True,
                                                        deduplicate=deduplicate)
                    self.assertIs(result, people)
                    self.assertEqual(counts, (2, 2) if deduplicate else (2, 0))
                    attribute = 'deepid' if backend == 'Torchreid' else 'faceid'
                    self.assertEqual([getattr(p, attribute) for p in result], ['same-id', 'same-id'])
                    self.assertEqual([getattr(p, attribute + '_conf') for p in result], [87.25, 87.25])
                    expected *= 2 if deduplicate else 1
                    self.assertEqual(len(captured), len(expected))
                    for actual, legacy in zip(captured, expected):
                        np.testing.assert_array_equal(actual, legacy)

    def test_negative_and_oversized_bounds_are_clipped_to_visible_pixels(self):
        frame = np.random.default_rng(18).integers(0, 256, (80, 100, 3), dtype=np.uint8)
        for backend in ('Torchreid', 'FaceNet'):
            with self.subTest(backend=backend):
                captured = []

                def recognize(crop, **kwargs):
                    captured.append(crop.copy())
                    return 'same-id', 91.

                pipeline = self.pipeline(backend, recognize)
                people = [Person(0, 0, box_xyxy=[-5, -7, 18, 25], repspoint=(3, 2)),
                          Person(1, 1, box_xyxy=[95, 75, 120, 100], repspoint=(98, 79))]
                expected = ([cv2.resize(frame[:25, :18], (8, 16)),
                             cv2.resize(frame[75:, 95:], (8, 16))]
                            if backend == 'Torchreid' else [frame[:5, :9], frame[74:, 94:]])
                result, counts = pipeline.reidPeople(frame, people, img_is_mat=True, deduplicate=True)
                self.assertEqual(counts, (2, 2))
                self.assertEqual(len(captured), 4)
                for actual, clipped in zip(captured, expected * 2):
                    np.testing.assert_array_equal(actual, clipped)

    def test_invalid_crops_are_skipped_without_stopping_other_people(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        cases = {
            'Torchreid': [dict(box_xyxy=box) for box in (
                [-12, 5, -2, 25], [101, 5, 120, 25], [5, 90, 10, 100],
                [8, 5, 2, 25], [3, 5, 3, 25], [0, 0, float('nan'), 20],
                [0, 0, float('inf'), 20], [0, 0, 2.5, 20], [0, 0, 2])],
            'FaceNet': [dict(repspoint=point) for point in (
                (-100, -100), (200, 100), (float('nan'), 10),
                (10, float('inf')), (10.5, 10), (10,))],
        }
        for backend, invalid in cases.items():
            with self.subTest(backend=backend):
                pipeline = self.pipeline(backend, lambda crop, **kwargs: ('same-id', 85.))
                people = [Person(i, i, **values) for i, values in enumerate(invalid)]
                people.append(Person(99, 99, box_xyxy=[2, 3, 22, 33], repspoint=(15, 20)))
                with patch('pyppbox.ppb.mt.add_warning_log') as warnings:
                    result, counts = pipeline.reidPeople(frame, people, img_is_mat=True, deduplicate=False)
                self.assertEqual(counts, (1, 0))
                self.assertEqual(pipeline.__ri__.recognize.call_count, 1)
                self.assertEqual(warnings.call_count, len(invalid))
                attribute = 'deepid' if backend == 'Torchreid' else 'faceid'
                self.assertTrue(all(getattr(person, attribute) == 'Unknown' for person in result[:-1]))
                self.assertEqual(getattr(result[-1], attribute), 'same-id')

    def test_invalid_duplicate_crop_retains_identity_and_does_not_count_a_call(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        for backend in ('Torchreid', 'FaceNet'):
            with self.subTest(backend=backend):
                pipeline = self.pipeline(backend, lambda crop, **kwargs: ('same-id', 85.))
                people = [Person(0, 0, box_xyxy=[-12, 5, -2, 25], repspoint=(-100, -100),
                                 faceid='same-id', deepid='same-id', faceid_conf=62., deepid_conf=62.),
                          Person(1, 1, box_xyxy=[2, 3, 22, 33], repspoint=(15, 20),
                                 faceid='same-id', deepid='same-id')]
                with patch('pyppbox.ppb.mt.add_warning_log') as warnings:
                    result, counts = pipeline.reidPeople(frame, people, img_is_mat=True)
                self.assertEqual(counts, (0, 1))
                self.assertEqual(warnings.call_count, 1)
                attribute = 'deepid' if backend == 'Torchreid' else 'faceid'
                self.assertEqual([getattr(person, attribute) for person in result], ['same-id', 'same-id'])
                self.assertEqual([getattr(person, attribute + '_conf') for person in result], [62., 85.])

    def test_recognizer_mutation_cannot_change_readonly_or_strided_input_frames(self):
        for backend in ('Torchreid', 'FaceNet'):
            for readonly, strided in ((False, False), (True, False), (False, True)):
                with self.subTest(backend=backend, readonly=readonly, strided=strided):
                    frame = np.random.default_rng(19).integers(0, 256, (80, 100, 3), dtype=np.uint8)
                    if strided:
                        frame = frame[:, ::-1]
                    before = frame.copy()
                    if readonly:
                        frame.flags.writeable = False

                    def recognize(crop, **kwargs):
                        crop[:] = 0
                        return 'same-id', 88.

                    pipeline = self.pipeline(backend, recognize)
                    people = [Person(i, i, box_xyxy=[2, 3, 22, 33], repspoint=(15, 20)) for i in range(2)]
                    _, counts = pipeline.reidPeople(frame, people, img_is_mat=True)
                    self.assertEqual(counts, (2, 2))
                    np.testing.assert_array_equal(frame, before)

    def test_known_identities_and_empty_inputs_do_not_invoke_recognition(self):
        for backend in ('Torchreid', 'FaceNet'):
            with self.subTest(backend=backend):
                pipeline = self.pipeline(backend, lambda crop, **kwargs: self.fail('Unexpected recognition'))
                person = Person(0, 0, box_xyxy=[], repspoint=(), faceid='Alice', deepid='Alice')
                for people in ([], [person]):
                    result, counts = pipeline.reidPeople(np.zeros((80, 100, 3), dtype=np.uint8),
                                                        people, img_is_mat=True, deduplicate=False)
                    self.assertEqual(counts, (0, 0))
                    self.assertEqual(result, people)


if __name__ == '__main__':
    unittest.main()
