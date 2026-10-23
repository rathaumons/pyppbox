"""Real tracker state machines with a deterministic DeepSORT feature encoder."""
import importlib
import itertools
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pyppbox.modules.trackers.sort import MySORT
from pyppbox.modules.trackers.sort.origin import sort
from pyppbox.modules.trackers import centroid
from pyppbox.ppb.mt import MT
from pyppbox.utils.persontools import Person


def person(x=0):
    return Person(0, -1, box_xywh=np.array([x, 0, 20, 40]),
                  box_xyxy=np.array([x, 0, x + 20, 40]), repspoint=(x + 10, 10))


class TrackingTests(unittest.TestCase):
    def make_sort(self):
        return MySORT(SimpleNamespace(max_age=2, min_hits=1, iou_threshold=.3))

    def make_deepsort(self):
        encoder = Mock(side_effect=lambda frame, boxes: np.tile([1., 0., 0.], (len(boxes), 1)))
        module_name = 'pyppbox.modules.trackers.deepsort'
        # Replace only the model-loading boundary. Kalman filter, matching, metric,
        # track lifecycle, and the public wrapper execute their real code.
        with patch.dict(sys.modules, {module_name + '.origin.generate_detections':
                                     SimpleNamespace(create_box_encoder=Mock(return_value=encoder))}):
            module = importlib.import_module(module_name)
            tracker = module.MyDeepSORT(SimpleNamespace(nms_max_overlap=1., model_file='unused.pb',
                                                       max_cosine_distance=.2, nn_budget=100))
        tracker.tracker.max_age = 2
        tracker.tracker.n_init = 2
        return tracker

    def load_deepsort_assignment(self):
        module_name = 'pyppbox.modules.trackers.deepsort.origin'
        with patch.dict(sys.modules, {module_name + '.generate_detections':
                                     SimpleNamespace(create_box_encoder=Mock())}):
            return importlib.import_module(module_name + '.linear_assignment')

    def test_expiry_empty_frames_and_metadata_through_facade(self):
        for factory in (self.make_sort, self.make_deepsort):
            with self.subTest(tracker=factory.__name__):
                tracker = factory()
                mt = MT()
                mt.__tk__, mt.__tk_is_set__ = tracker, True
                frame = np.zeros((60, 120, 3), dtype=np.uint8)
                for _ in range(4):
                    people = mt.trackPeople(frame, [person()], img_is_mat=True)
                cid = people[0].cid
                people[0].deepid, people[0].faceid = 'Alice', 'Alice-face'
                people[0].deepid_conf, people[0].faceid_conf = 83., 91.
                people[0].misc = [{'tag': 1}]
                self.assertEqual(mt.trackPeople(frame, [], img_is_mat=True), [])
                if hasattr(tracker, 'encoder'):
                    self.assertEqual(tracker.encoder.call_count, 4)
                returned = mt.trackPeople(frame, [person()], img_is_mat=True)[0]
                self.assertEqual(returned.cid, cid)
                self.assertEqual((returned.deepid, returned.faceid), ('Alice', 'Alice-face'))
                self.assertEqual((returned.deepid_conf, returned.faceid_conf), (83., 91.))
                self.assertEqual(returned.misc, [{'tag': 1}])
                for _ in range(3):
                    self.assertEqual(mt.trackPeople(frame, [], img_is_mat=True), [])
                tracks = tracker.st.trackers if hasattr(tracker, 'st') else tracker.tracker.tracks
                self.assertEqual(tracks, [])
                for _ in range(2):
                    returned = mt.trackPeople(frame, [person()], img_is_mat=True)
                self.assertNotEqual(returned[0].cid, cid)
                self.assertEqual(returned[0].deepid, 'Unknown')

    def test_direct_empty_update_advances_tracker(self):
        for factory in (self.make_sort, self.make_deepsort):
            tracker = factory()
            tracker.update([person()], img=np.zeros((60, 60, 3), dtype=np.uint8))
            for _ in range(3):
                self.assertEqual(tracker.update([], img=None), [])
            tracks = tracker.st.trackers if hasattr(tracker, 'st') else tracker.tracker.tracks
            self.assertEqual(tracks, [])

    def test_centroid_assignment_fallback(self):
        broken_lap = SimpleNamespace(lapjv=Mock(side_effect=RuntimeError('injected LAP failure')))
        for backend in (None, SimpleNamespace(), broken_lap, centroid.laprust):
            with self.subTest(backend=backend), patch.object(centroid, 'laprust', backend):
                tracker = centroid.MyCentroid(SimpleNamespace(max_spread=30))
                previous = tracker.update([person(0), person(50)])
                cids = [p.cid for p in previous]
                result = tracker.update([person(51), person(1), person(200)])
                self.assertEqual([p.cid for p in result[:2]], cids[::-1])
                self.assertNotIn(result[2].cid, cids)
                remaining = tracker.update([person(2)])
                self.assertEqual(remaining[0].cid, cids[0])
                self.assertEqual(tracker.update([]), [])

    def test_centroid_all_distances_outside_gate(self):
        tracker = centroid.MyCentroid(SimpleNamespace(max_spread=5))
        cid = tracker.update([person()])[0].cid
        self.assertNotEqual(tracker.update([person(100)])[0].cid, cid)

    def test_laprust_pair_outputs_for_square_rectangular_and_empty_costs(self):
        assignment = self.load_deepsort_assignment()
        wide = np.array([[.3, .1, .8], [.05, .4, .9]])
        wide.setflags(write=False)
        cases = [wide[:, :2], wide, wide.T, np.empty((0, 2)),
                 np.empty((2, 0)), np.empty((0, 0))]
        for module, solve, dtype in ((sort, sort.linear_assignment, np.int32),
                                     (centroid, centroid._linear_assignment, np.int32),
                                     (assignment, assignment._linear_assignment, np.int64)):
            for costs in cases:
                with self.subTest(module=module.__name__, shape=costs.shape):
                    solver = Mock(wraps=module.laprust.lapjv)
                    with patch.object(module, 'laprust', SimpleNamespace(lapjv=solver)):
                        pairs = solve(costs)
                    expected = np.array([[0, 1], [1, 0]]) if costs.size else np.empty((0, 2))
                    np.testing.assert_array_equal(pairs, expected)
                    self.assertEqual(pairs.dtype, dtype)
                    solver.assert_called_once()

    def test_sort_assignment_preserves_iou_post_filtering(self):
        # A finite solver cost_limit would select only (0, 0) in the first case.
        # SORT solves the full assignment first, retaining both off-diagonal pairs.
        cases = [(np.array([[.9, .6, 0.], [.6, .1, 0.]], dtype=np.float32),
                  ([[0, 1], [1, 0]], [], [2])),
                 (np.array([[.5, .6], [.4, .9]], dtype=np.float32),
                  ([[0, 0], [1, 1]], [], [])),
                 (np.zeros((2, 3), dtype=np.float32), ([], [0, 1], [0, 1, 2]))]
        for ious, expected in cases:
            with self.subTest(ious=ious), patch.object(sort, 'iou_batch', return_value=ious), \
                    patch.object(sort, 'linear_assignment', wraps=sort.linear_assignment) as solve:
                matches, unmatched_detections, unmatched_tracks = sort.associate_detections_to_trackers(
                    np.zeros((ious.shape[0], 4)), np.zeros((ious.shape[1], 4)), iou_threshold=.5)
                np.testing.assert_array_equal(matches, np.array(expected[0], dtype=np.int32).reshape(-1, 2))
                np.testing.assert_array_equal(unmatched_detections, expected[1])
                np.testing.assert_array_equal(unmatched_tracks, expected[2])
                solve.assert_called_once()

    def test_deepsort_assignment_backends_preserve_matches_and_gating(self):
        assignment = self.load_deepsort_assignment()
        lap_solver = Mock(wraps=assignment.laprust.lapjv)
        backends = [('laprust', SimpleNamespace(lapjv=lap_solver)),
                    ('unavailable', None), ('missing_api', SimpleNamespace()),
                    ('failed', SimpleNamespace(lapjv=Mock(side_effect=RuntimeError('LAP failure'))))]
        wide = np.array([[.3, .1, .8], [.05, .4, .9]], dtype=np.float32)
        cases = [(wide[:, :2], ([(0, 1), (1, 0)], [], [])),
                 (wide, ([(0, 1), (1, 0)], [], [2])),
                 (wide.T, ([(0, 1), (1, 0)], [2], [])),
                 (np.array([[.6, .7], [.8, np.inf]]), ([], [0, 1], [0, 1])),
                 (np.array([[.5]]), ([(0, 0)], [], [])),
                 # A finite .5 cost_limit would reject the second track instead.
                 (np.array([[0., .25], [.250005, 1.]]), ([(0, 1), (1, 0)], [], [])),
                 (np.empty((0, 2)), ([], [], [0, 1])),
                 (np.empty((2, 0)), ([], [0, 1], [])),
                 (np.empty((0, 0)), ([], [], []))]
        for name, backend in backends:
            for costs, expected in cases:
                with self.subTest(backend=name, costs=costs), \
                        patch.object(assignment, 'laprust', backend), \
                        patch.object(assignment, 'linear_sum_assignment',
                                     wraps=assignment.linear_sum_assignment) as fallback:
                    lap_solver.reset_mock()
                    metric = Mock(return_value=costs.copy(order='K'))
                    result = assignment.min_cost_matching(
                        metric, .5, [None] * costs.shape[0], [None] * costs.shape[1])
                    self.assertEqual(tuple(sorted(part) for part in result), expected)
                    has_costs = bool(costs.size)
                    self.assertEqual(metric.call_count, int(has_costs))
                    self.assertEqual(lap_solver.call_count, int(name == 'laprust' and has_costs))
                    self.assertEqual(fallback.call_count, int(name != 'laprust' and has_costs))

    def test_deepsort_assignment_preserves_precision_and_index_mapping(self):
        assignment = self.load_deepsort_assignment()
        costs = np.array([[1., 1. + 1e-9, 3.], [1., 1., 3.]])
        for backend in (assignment.laprust, None):
            with self.subTest(backend=backend), patch.object(assignment, 'laprust', backend):
                result = assignment.min_cost_matching(
                    Mock(return_value=costs.copy()), 2., [None] * 5, [None] * 9,
                    track_indices=[4, 2], detection_indices=[8, 1, 6])
                self.assertEqual(result, ([(4, 8), (2, 1)], [], [6]))

    def test_deepsort_invalid_costs_use_scipy_validation(self):
        assignment = self.load_deepsort_assignment()
        backend = SimpleNamespace(lapjv=Mock())
        for invalid in (np.nan, -np.inf):
            with self.subTest(cost=invalid), patch.object(assignment, 'laprust', backend), \
                    patch.object(assignment, 'linear_sum_assignment',
                                 wraps=assignment.linear_sum_assignment) as fallback:
                with self.assertRaises(ValueError):
                    assignment.min_cost_matching(
                        Mock(return_value=np.array([[invalid]])), .5, [None], [None])
                fallback.assert_called_once()
        backend.lapjv.assert_not_called()

    def test_gt_mode_setter_order_and_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gt.txt'
            path.write_text('0\t(10, 10)\tAlice\t[0 0 20 40]\t[0 0 20 40]\n', encoding='utf-8')
            actions = [('setMainDetector', {'dt_name': 'GT', 'gt_file': str(path), 'gt_map_file': ''}),
                       ('setMainTracker', {'tk_name': 'Centroid', 'max_spread': 30}),
                       ('setMainReIDer', {'ri_name': 'None'})]
            for ordering in itertools.permutations(actions):
                with self.subTest(order=[name for name, _ in ordering]):
                    mt = MT()
                    for name, value in ordering:
                        getattr(mt, name)(value)
                    self.assertTrue(mt.__dt__.detect_only)
                    mt.forceFullGTMode()
                    self.assertFalse(mt.__dt__.detect_only)
                    people, _ = mt.detectPeople(np.zeros((60, 60, 3), dtype=np.uint8))
                    self.assertEqual(people[0].deepid, 'Alice')
                    self.assertFalse(mt.__dt__.detect_only)
                    mt.setMainTracker({'tk_name': 'Centroid', 'max_spread': 30})
                    self.assertTrue(mt.__dt__.detect_only)
                    mt.setMainTracker({'tk_name': 'None'})
                    self.assertFalse(mt.__dt__.detect_only)
                    mt.setMainReIDer({'ri_name': 'None'})
                    self.assertFalse(mt.__dt__.detect_only)

    def test_sort_import_preserves_numpy_random_state(self):
        code = '''import numpy as np
np.random.seed(837)
expected = np.random.random(4)
np.random.seed(837)
import pyppbox.modules.trackers.sort
np.testing.assert_array_equal(np.random.random(4), expected)
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                env=dict(os.environ, PYPPBOX_DISABLE_FILE_LOG='1'))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_gt_mode_default_loaders_and_active_reider(self):
        from pyppbox.config.configtools import internal_cfg_dir, loadDocumentList, dumpDocDict, dumpDocDictList
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = root / 'gt.txt'
            gt.write_text('', encoding='utf-8')
            detectors = loadDocumentList(Path(internal_cfg_dir) / 'detectors.yaml')
            for config in detectors:
                if config['dt_name'] == 'GT':
                    config['gt_file'] = str(gt)
            dumpDocDictList(root / 'detectors.yaml', detectors, '')
            for name in ('trackers', 'reiders'):
                dumpDocDictList(root / (name + '.yaml'),
                               loadDocumentList(Path(internal_cfg_dir) / (name + '.yaml')), '')
            dumpDocDict(root / 'main.yaml', {'detector': 'GT', 'tracker': 'Centroid', 'reider': 'None'}, '')
            mt = MT()
            mt.setConfigDir(str(root), load_all=True)
            self.assertTrue(mt.__dt__.detect_only)
            mt.setMainModules({'detector': 'GT', 'tracker': 'None', 'reider': 'None'})
            self.assertFalse(mt.__dt__.detect_only)
            with patch('pyppbox.modules.reiders.torchreid.deepreid_extractor', return_value=Mock()), \
                    patch('pyppbox.config.myconfig.RCFGTorchreid.validateRuntime'):
                mt.setMainReIDer('Torchreid', auto_load=False)
            self.assertTrue(mt.__dt__.detect_only)
            mt.setMainDetector('GT')
            self.assertTrue(mt.__dt__.detect_only)
            mt.setMainReIDer('None')
            self.assertFalse(mt.__dt__.detect_only)


if __name__ == '__main__':
    unittest.main()
