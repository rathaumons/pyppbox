# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""A comparison must reject identity/order/count changes and nonfinite output."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('compare_cpu', Path(__file__).resolve().parents[1] / 'benchmarks/compare_cpu.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class BenchmarkContractTests(unittest.TestCase):
    def test_rejects_changed_labels_order_counts_and_missing_fields(self):
        expected = {'labels':['a', 'b'], 'counts':[2, 0], 'confidence':91.5}
        for actual in ({**expected, 'labels':['b', 'a']}, {**expected, 'counts':[1, 0]},
                       {'labels':['a', 'b'], 'counts':[2, 0]}, {**expected, 'confidence':91.6}):
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                comparison.equivalent(expected, actual)

    def test_allows_small_numerical_drift_but_not_nonfinite_values(self):
        comparison.equivalent([91.5], [91.5000001])
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                comparison.equivalent([value], [value])


if __name__ == '__main__':
    unittest.main()
