# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""DeepSORT's actual associations own identity metadata and Person payloads."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from pyppbox.utils.persontools import Person
import test_tracking


def person(identity, x=10, confidence=.8):
    return Person(identity, -1, box_xywh=np.array([x, 10, 40, 50]),
                  box_xyxy=np.array([x, 10, x + 40, 60]), det_conf=confidence,
                  repspoint=(x + 20, 20), keypoints=np.full((17, 3), identity))


class DeepSORTMetadataTests(unittest.TestCase):
    def make_tracker(self, nms=1.):
        tracker = test_tracking.TrackingTests().make_deepsort()
        tracker.nms_max_overlap = nms
        tracker.encoder.side_effect = None
        return tracker

    def update(self, tracker, people, **options):
        tracker.encoder.return_value = np.eye(4)[[p.init_id for p in people]]
        return tracker.update(people, img=np.zeros((100, 240, 3), dtype=np.uint8), **options)

    def test_identical_boxes_get_distinct_matched_ids(self):
        tracker = self.make_tracker()
        for step in range(4):
            inputs = [person(0, confidence=.9), person(1, confidence=.8)]
            output = self.update(tracker, inputs)
            self.assertIs(output, inputs)
            # Tentative tracks use IOU: an exact-box tie may pick either pairing.
            # Both actual matches must survive the wrapper as distinct CIDs.
            self.assertEqual(sorted(p.cid for p in output), [-1, -1] if step == 0 else [1, 2])
            for source, result in zip(inputs, output):
                self.assertIs(source, result)
                self.assertTrue(np.all(result.keypoints == result.init_id))

    def test_nms_reordering_and_suppression_keep_original_person_index(self):
        tracker = self.make_tracker(nms=.5)
        for step in range(3):
            output = self.update(tracker, [person(0, confidence=.7), person(1, confidence=.9)])
            self.assertEqual([p.cid for p in output], [-1, -1] if step == 0 else [-1, 1])
        self.assertIs(tracker._people_by_cid[1], output[1])
        # NMS sorts by score, while the returned list stays in input order.
        tracker = self.make_tracker()
        for _ in range(2):
            output = self.update(tracker, [person(0, 10, .7), person(1, 110, .9)])
        self.assertEqual([p.cid for p in output], [2, 1])

    def test_missing_track_cannot_steal_detection_or_metadata(self):
        tracker = self.make_tracker()
        for _ in range(3):
            previous = self.update(tracker, [person(0, 10, .9), person(1, 70, .8)])
        for p, name in zip(previous, ('Alice', 'Bob')):
            p.deepid, p.faceid = name, name + '-face'
            p.deepid_conf, p.faceid_conf, p.misc = 83., 91., [{'name': name}]
        result = self.update(tracker, [person(0, 10, .9)])[0]
        self.assertEqual((result.cid, result.deepid, result.faceid), (1, 'Alice', 'Alice-face'))
        self.assertEqual((result.deepid_conf, result.faceid_conf), (83., 91.))
        self.assertIs(result.misc, previous[0].misc)
        self.assertIs(tracker._people_by_cid[2], previous[1])
        returned = self.update(tracker, [person(1, 70, .8), person(0, 10, .9)])
        self.assertEqual([(p.cid, p.deepid) for p in returned], [(2, 'Bob'), (1, 'Alice')])
        self.assertIs(returned[0].misc, previous[1].misc)
        for _ in range(3):
            self.assertEqual(self.update(tracker, []), [])
        self.assertEqual(tracker._people_by_cid, {})
        self.assertEqual(tracker.tracker.tracks, [])

    def test_crossing_keeps_appearance_ids_and_detection_payloads_aligned(self):
        tracker = self.make_tracker()
        for frame in range(9):
            inputs = [person(0, 10 + frame * 10, .9), person(1, 90 - frame * 10, .8)]
            if frame % 2:
                inputs.reverse()
            output = self.update(tracker, inputs)
            self.assertIs(output, inputs)
            for p in output:
                self.assertEqual(p.cid, -1 if frame == 0 else p.init_id + 1)
                self.assertTrue(np.all(p.keypoints == p.init_id))
                self.assertEqual(p.repspoint, (int(p.box_xyxy[0]) + 20, 20))
                self.assertEqual(p.det_conf, .9 if p.init_id == 0 else .8)

    def test_core_associations_reset_on_misses_deletion_and_new_tracks(self):
        tracker = self.make_tracker()
        for _ in range(2):
            self.update(tracker, [person(0, 10, .9), person(1, 110, .8)])
        self.assertEqual(tracker.tracker.detection_indices, {1: 0, 2: 1})
        self.update(tracker, [person(1, 110, .8)])
        self.assertEqual(tracker.tracker.detection_indices, {2: 0})
        self.update(tracker, [])
        self.assertEqual(tracker.tracker.detection_indices, {})
        for _ in range(2):
            self.update(tracker, [])
        for _ in range(2):
            output = self.update(tracker, [person(2, 150, .9)])
        self.assertEqual(tracker.tracker.detection_indices, {3: 0})
        self.assertEqual(output[0].cid, 3)
        # Existing direct callers still receive None from the embedded update.
        tracker.tracker.predict()
        self.assertIsNone(tracker.tracker.update([]))
        self.assertEqual(tracker.tracker.detection_indices, {})

    def test_max_spread_remains_an_inclusive_gate_on_the_matched_detection(self):
        for maximum, expected in ((128, 7), (127, -1), (-1, -1)):
            with self.subTest(maximum=maximum):
                tracker = self.make_tracker()
                track = SimpleNamespace(track_id=7, time_since_update=0,
                    is_confirmed=lambda: True, to_tlbr=lambda: np.array([138., 10., 178., 60.]))
                tracker.tracker = SimpleNamespace(predict=Mock(), update=Mock(), tracks=[track],
                                                  detection_indices={7: 0})
                output = self.update(tracker, [person(0)], max_spread=maximum)
                self.assertEqual(output[0].cid, expected)


if __name__ == '__main__':
    unittest.main()
