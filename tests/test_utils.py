"""State ownership, stream restoration, logging, and image I/O regressions."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from pyppbox.utils import commontools, logtools
from pyppbox.utils.evatools import TKOReider
from pyppbox.utils.persontools import Person
from pyppbox.utils.restools import ResIO
from pyppbox.ppb.mt import MT


class UtilityTests(unittest.TestCase):
    def test_person_defaults_are_independent(self):
        first, second = Person(0, 0), Person(1, 1)
        for name in ('box_xywh', 'box_xyxy', 'keypoints', 'misc'):
            getattr(first, name).append(1)
            self.assertEqual(getattr(second, name), [])
        explicit = np.array([1, 2, 3, 4])
        self.assertIs(Person(0, 0, box_xyxy=explicit).box_xyxy, explicit)

    def test_results_snapshot_both_add_methods(self):
        person = Person(0, 0, box_xyxy=np.array([0, 1, 2, 3]), deepid='Alice')
        person.misc = [{'nested': [1]}]
        results = ResIO()
        results.addPerson(0, person)
        results.addPeople(1, [person])
        person.deepid = 'Bob'
        person.box_xyxy[0] = 99
        person.misc[0]['nested'].append(2)
        for stored in results.people:
            self.assertEqual(stored.deepid, 'Alice')
            self.assertEqual(stored.box_xyxy[0], 0)
            self.assertEqual(stored.misc, [{'nested': [1]}])

    def test_silencer_restores_nested_streams_and_closes_handles(self):
        observed = []

        @commontools.silencer
        def quiet(raise_error=False):
            observed.append(sys.stdout)
            print('hidden')
            if raise_error:
                raise RuntimeError('expected')
            return 42

        @commontools.silencer
        def nested():
            outer = sys.stdout
            self.assertEqual(quiet(), 42)
            self.assertIs(sys.stdout, outer)

        stream = io.StringIO()
        with redirect_stdout(stream):
            nested()
            self.assertIs(sys.stdout, stream)
            with self.assertRaisesRegex(RuntimeError, 'expected'):
                quiet(True)
            self.assertIs(sys.stdout, stream)
            print('visible')
        self.assertEqual(stream.getvalue(), 'visible\n')
        self.assertTrue(all(handle.closed for handle in observed))
        self.assertEqual(quiet.__name__, 'quiet')

    def test_static_ids_are_consumed_and_copied(self):
        supplied = ['Alice', 'Bob']
        first = TKOReider(static=True, static_ids=supplied)
        second = TKOReider(static=True, static_ids=supplied)
        supplied.append('Charlie')
        self.assertEqual([first.recognize(None) for _ in range(3)], ['Alice', 'Bob', 'EoR2'])
        self.assertEqual(second.recognize(None), 'Alice')
        first.setStaticIDs(['Replacement'])
        self.assertEqual(first.recognize(None), 'Replacement')

    def test_default_static_ids_do_not_leak_between_instances(self):
        with patch.object(TKOReider, 'generateID', return_value='Random'):
            first, second = TKOReider(static=True), TKOReider(static=True)
        self.assertIsNot(first.static_ids, second.static_ids)
        self.assertEqual(first.static_ids_len, len(first.static_ids))
        self.assertEqual(first.recognize(None), 'Lester')
        self.assertEqual(second.recognize(None), 'Lester')
        supplied = []
        first.setStaticIDs(supplied, plus_random=0)
        self.assertEqual(supplied, [])
        self.assertEqual(first.static_ids_len, 15)

    def test_image_failures_and_rgb_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / 'invalid.png'
            invalid.write_bytes(b'not an image')
            for value in (invalid, Path(directory) / 'missing.png', None, 'missing.png',
                          np.empty((0, 3, 3)), np.empty((4, 4))):
                with self.subTest(value=str(value)):
                    with self.assertRaises(ValueError):
                        commontools.getCVMat(value)
            bgr = np.full((4, 4, 3), [10, 20, 30], dtype=np.uint8)
            valid = Path(directory) / 'valid.png'
            self.assertTrue(cv2.imwrite(str(valid), bgr))
            np.testing.assert_array_equal(commontools.getCVMat(valid), bgr)
            np.testing.assert_array_equal(commontools.getCVMat(bgr, to_rgb=True), bgr[:, :, ::-1])
            self.assertIs(commontools.getCVMat(bgr), bgr)

    def test_detector_boundary_rejects_invalid_mat_and_failed_save(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        mt = MT()
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        detector = Mock()
        detector.detectPeople.return_value = ([], frame)
        mt.__dt__, mt.__dt_is_set__ = detector, True
        mt.__dt_cfg__ = SimpleNamespace(dt_name='YOLO_Ultralytics')
        with self.assertRaises(ValueError):
            mt.detectPeople(None, img_is_mat=True)
        detector.detectPeople.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            for outcome in (False, cv2.error('injected write error')):
                with self.subTest(outcome=outcome):
                    args = {'side_effect': outcome} if isinstance(outcome, Exception) else {'return_value': outcome}
                    with patch.object(cv2, 'imwrite', **args):
                        with self.assertRaisesRegex(ValueError, 'Cannot save image'):
                            mt.detectPeople(frame, save=True, save_file=str(Path(directory) / 'out.png'))

    def test_cleanup_only_owns_timestamped_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_log = root / 'log_20200101_010101.txt'
            kept = [root / 'notes.txt', root / '.gitignore', root / 'log_other.txt',
                    root / 'log_20260905_010101.txt']
            for path in [old_log] + kept:
                path.write_text('preserve', encoding='utf-8')
            for path in [old_log] + kept[:-1]:
                os.utime(path, (0, 0))
            folder = root / 'log_20200102_010101.txt'
            folder.mkdir()
            with patch.object(logtools, '__log_dir__', directory):
                logtools.cleanup_old_logs()
            self.assertFalse(old_log.exists())
            self.assertTrue(all(path.exists() for path in kept + [folder]))

    def test_disabled_logging_import_has_no_filesystem_side_effects(self):
        code = '''from unittest.mock import patch
with patch('os.listdir', side_effect=AssertionError('cleanup on import')), patch('os.makedirs', side_effect=AssertionError('mkdir on import')):
    from pyppbox.utils import logtools
    assert not logtools.get_terminal_log_status()
    logtools.add_info_log('should be silent')
'''
        env = dict(os.environ, PYPPBOX_DISABLE_FILE_LOG='1', PYPPBOX_DISABLE_TERMINAL_LOG='1')
        result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
