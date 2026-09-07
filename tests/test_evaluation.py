"""Independently specified evaluation counts for B14."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pyppbox.utils import evatools
from pyppbox.utils.evatools import MyEVA, compareRes2Ref
from pyppbox.utils.persontools import Person


def row(frame, name, x):
    return f'{frame}\t({x + 10}, 10)\t{name}\t[{x} 0 20 40]\t[{x} 0 {x + 20} 40]\n'


def person(name, x):
    return Person(0, 0, box_xyxy=np.array([x, 0, x + 20, 40]), deepid=name, faceid=name)


class EvaluationTests(unittest.TestCase):
    def test_assignment_backends_preserve_counts(self):
        lap_solver = Mock(wraps=evatools.lap.lapjvx)
        backends = [('lapx', SimpleNamespace(lapjvx=lap_solver)),
                    ('unavailable', None), ('missing_api', SimpleNamespace()),
                    ('failed', SimpleNamespace(lapjvx=Mock(side_effect=RuntimeError('LAP failure'))))]
        cases = [([0, 20], ['Alice', 'Bob'], [8, -10], ['Bob', 'Alice'], (0, 0, 0)),
                 ([0, 100], ['Alice', 'Bob'], [100, 0, 200], ['Wrong', 'Alice', 'Extra'], (1, 0, 1)),
                 ([0, 100, 200], ['Alice', 'Bob', 'Carol'], [100, 0], ['Bob', 'Alice'], (0, 1, 0)),
                 ([0, 100], ['Alice', 'Bob'], [300, 400], ['Other', 'Extra'], (0, 2, 2)),
                 ([0], ['Alice'], [16], ['Alice'], (0, 0, 0)),
                 ([], [], [0], ['Alice'], (0, 0, 1)),
                 ([0], ['Alice'], [], [], (0, 1, 0)),
                 ([], [], [], [], (0, 0, 0))]
        for name, backend in backends:
            for ref_x, ref_ids, pred_x, pred_ids, expected in cases:
                with self.subTest(backend=name, references=ref_x, predictions=pred_x), \
                        patch.object(evatools, 'lap', backend), \
                        patch.object(evatools, 'linear_sum_assignment',
                                     wraps=evatools.linear_sum_assignment) as fallback:
                    lap_solver.reset_mock()
                    result = evatools._compare_detections(
                        [[x, 0, x + 20, 40] for x in ref_x], ref_ids,
                        [[x, 0, x + 20, 40] for x in pred_x], pred_ids, 16)
                    self.assertEqual(result, expected)
                    has_costs = bool(ref_x and pred_x)
                    self.assertEqual(lap_solver.call_count, int(name == 'lapx' and has_costs))
                    self.assertEqual(fallback.call_count, int(name != 'lapx' and has_costs))

    def test_assignment_preserves_float64_precision(self):
        # Rounding to float32 turns the unique diagonal optimum into a tie.
        costs = np.array([[1., 1. + 1e-9], [1., 1.]])
        with patch.object(evatools, 'linear_sum_assignment') as fallback:
            rows, cols = evatools._linear_assignment(costs)
        np.testing.assert_array_equal(rows, [0, 1])
        np.testing.assert_array_equal(cols, [0, 1])
        fallback.assert_not_called()

    def evaluate(self, reference, people, id_mode='deepid'):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gt.txt'
            path.write_text(reference, encoding='utf-8')
            evaluator = MyEVA()
            evaluator.setGTByKnownGTFile(str(path), id_mode=id_mode)
            evaluator.validate(people, frame_id=0)
            return evaluator.getSummary(print_summary=False)

    def test_duplicates_do_not_hide_missed_people(self):
        for mode in ('deepid', 'faceid'):
            result = self.evaluate(row(0, 'Alice', 0) + row(0, 'Bob', 100),
                                   [person('Alice', 0), person('Alice', 0)], mode)
            self.assertEqual(result, (0, 1, 1, 0, 2, .5))

    def test_geometry_not_identity_determines_assignment(self):
        result = self.evaluate(row(0, 'Alice', 0) + row(0, 'Bob', 100),
                               [person('Alice', 100), person('Bob', 0)])
        self.assertEqual(result, (2, 0, 0, 0, 2, 0.))

    def test_unmatched_predictions_and_gt_are_counted_independently(self):
        reference = row(0, 'Alice', 0) + row(0, 'Bob', 100)
        cases = [([], (0, 2, 0, 0, 2, 0.)),
                 ([person('Nobody', 200), person('Nobody', 300)], (0, 2, 2, 0, 2, 0.)),
                 ([person('Nobody', 200), person('Bob', 100), person('Alice', 0)], (0, 0, 1, 0, 2, 1.))]
        for predictions, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(self.evaluate(reference, predictions), expected)

    def test_assignment_maximizes_number_of_valid_matches(self):
        # The first prediction can match either reference; the second can only
        # match Alice. A greedy first-match walk loses Bob.
        result = self.evaluate(row(0, 'Alice', 0) + row(0, 'Bob', 20),
                               [person('Bob', 8), person('Alice', -10)])
        self.assertEqual(result, (0, 0, 0, 0, 2, 1.))

    def test_empty_gt_and_unconfigured_summary(self):
        self.assertEqual(self.evaluate('', []), (0, 0, 0, 0, 0, 0.))
        self.assertEqual(self.evaluate('', [person('Alice', 0)]), (0, 0, 1, 0, 0, 0.))
        evaluator = MyEVA()
        evaluator.validate([])
        self.assertEqual(evaluator.getSummary(False), (0, 0, 0, 0, 0, 0.))

    def test_offline_sparse_frames_duplicates_and_extra_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            ref, res = Path(directory) / 'gt.txt', Path(directory) / 'res.txt'
            ref.write_text(row(10, 'Alice', 0) + row(10, 'Bob', 100) + row(20, 'Carol', 200), encoding='utf-8')
            res.write_text(row(5, 'Extra', 0) + row(10, 'Alice', 0) + row(10, 'Alice', 0)
                           + row(20, 'Wrong', 200) + row(40, 'Extra', 0), encoding='utf-8')
            wrong, missed, false, total, score = compareRes2Ref(res, ref, res_box_xyxy_index=4)
            self.assertEqual((wrong, missed, false, total), (1, 1, 3, 3))
            self.assertAlmostEqual(score, 1 / 3)
            ref.write_text('', encoding='utf-8')
            self.assertEqual(compareRes2Ref(res, ref, res_box_xyxy_index=4), (0, 0, 5, 0, 0.))
            res.write_text('', encoding='utf-8')
            self.assertEqual(compareRes2Ref(res, ref, res_box_xyxy_index=4), (0, 0, 0, 0, 0.))

    def test_valid_gt_after_missing_gt_and_id_mode_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = MyEVA()
            evaluator.setGTByKnownGTFile(str(Path(directory) / 'missing.txt'), id_mode='faceid')
            path = Path(directory) / 'gt.txt'
            path.write_text(row(0, 'Alice', 0), encoding='utf-8')
            evaluator.setGTByKnownGTFile(str(path), id_mode='deepid')
            predicted = person('Alice', 0)
            predicted.faceid = 'Wrong'
            evaluator.validate([predicted], frame_id=0)
            self.assertEqual(evaluator.getSummary(False), (0, 0, 0, 0, 1, 1.))


if __name__ == '__main__':
    unittest.main()
