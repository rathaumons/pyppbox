"""GT row ownership/reloading and compatible, complete result-file ordering."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pyppbox.utils.gttools import GTInterpreter
from pyppbox.utils.persontools import Person
from pyppbox.utils.restools import ResIO


def gt_row(frame, name):
    return f'{frame}\t(5, 5)\t{name}\t[0 0 10 20]\t[0 0 10 20]\n'


class GTLookupTests(unittest.TestCase):
    def test_sparse_lookup_keeps_public_lists_and_live_row_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'gt.txt'
            path.write_text(gt_row(10, 'Alice') + gt_row(10, 'Bob') + gt_row(50, 'Carol'))
            gt = GTInterpreter()
            gt.setGT(path)
            self.assertIs(type(gt.gt_frames_list), list)
            self.assertIs(type(gt.gt_frames), list)
            self.assertEqual(gt.gt_frames_list, [10, 50])
            rows = gt.findGTFrame(10)
            self.assertIs(rows, gt.gt_frames[0])
            self.assertIsInstance(rows[0][1], tuple)
            self.assertIsInstance(rows[0][3], str)
            rows[0][2] = 'Edited'
            self.assertEqual(gt.findGTFrame(10)[0][2], 'Edited')
            replacement = [rows[1]]
            gt.gt_frames[0] = replacement
            self.assertIs(gt.findGTFrame(np.int64(10)), replacement)
            self.assertEqual(gt.findGTFrame(11), [])
            self.assertEqual(gt.findGTFrame(49), [])
            self.assertEqual(gt.findGTFrame(51), [])

    def test_reload_replaces_lookup_without_resetting_cursor_or_detection_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'gt.txt'
            path.write_text(gt_row(10, 'Alice') + gt_row(50, 'Bob'))
            gt = GTInterpreter()
            gt.setGT(path)
            gt.current_frame = 123
            gt.setDetectOnly(unknownFID='Face unknown', unknownDID='Body unknown')
            path.write_text(gt_row(3, 'Carol'))
            gt.setGT(path)
            self.assertEqual(gt.gt_frames_list, [3])
            self.assertEqual(gt.findGTFrame(3)[0][2], 'Carol')
            self.assertEqual(gt.findGTFrame(10), [])
            self.assertEqual(gt.findGTFrame(50), [])
            self.assertEqual(gt.current_frame, 123)
            self.assertTrue(gt.detect_only)
            self.assertEqual((gt.unknownFID, gt.unknownDID), ('Face unknown', 'Body unknown'))
            self.assertEqual((gt.init_frame, gt.total_detections), (3, 1))

    def test_failed_reload_keeps_the_previous_rows_and_lookup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'gt.txt'
            path.write_text(gt_row(10, 'Alice'))
            gt = GTInterpreter()
            gt.setGT(path)
            previous = gt.findGTFrame(10)
            with self.assertRaises(ValueError):
                gt.setGT(Path(folder) / 'missing.txt')
            self.assertIs(gt.findGTFrame(10), previous)
            self.assertEqual(gt.gt_frames_list, [10])

    def test_empty_file_and_missing_frames_return_independent_empty_results(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'gt.txt'
            path.write_text('')
            gt = GTInterpreter()
            gt.setGT(path)
            self.assertEqual(gt.gt_frames_list, [0])
            self.assertIs(gt.findGTFrame(0), gt.gt_frames[0])
            missing = gt.findGTFrame(10)
            missing.append('local edit')
            self.assertEqual(gt.findGTFrame(10), [])
            gt.current_frame = 10
            frame = np.zeros((20, 20, 3), dtype=np.uint8)
            people, returned = gt.getPeople(frame)
            self.assertEqual(people, [])
            self.assertIs(returned, frame)
            self.assertEqual(gt.current_frame, 11)

    def test_duplicate_frame_metadata_preserves_first_group_and_loader_ownership(self):
        first, second = [[10, (5, 5), 'First', '', '']], [[10, (5, 5), 'Second', '', '']]
        groups, ids = [first, second], [10, 10]
        gt = GTInterpreter()
        with patch.object(gt.gtIO, 'loadGT', return_value=(groups, ids, 2, 10)):
            gt.setGT('fixture')
        self.assertIs(gt.gt_frames_list, ids)
        self.assertIs(gt.gt_frames, groups)
        self.assertIs(gt.findGTFrame(10), first)

    def test_integer_query_compatibility_and_people_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'gt.txt'
            path.write_text(gt_row(-3, 'Lester') + gt_row(10, 'Michael'))
            gt = GTInterpreter()
            gt.setGT(path)
            self.assertEqual(gt.findGTFrame(-3)[0][2], 'Lester')
            self.assertIs(gt.findGTFrame(10.), gt.findGTFrame(10))
            # Keep the existing distinction between integer membership and exact
            # equality; input normalization is outside this performance change.
            for value in ('10', 10.5):
                with self.subTest(query=value), self.assertRaises(ValueError):
                    gt.findGTFrame(value)
            self.assertEqual(gt.findGTFrame('99'), [])
            frame = np.zeros((20, 20, 3), dtype=np.uint8)
            gt.current_frame = -3
            people, _ = gt.getPeople(frame)
            self.assertEqual((people[0].cid, people[0].faceid, people[0].deepid), (0, 'Lester', 'Lester'))
            gt.setDetectOnly('No face', 'No body')
            gt.current_frame = 10
            people, _ = gt.getPeople(frame)
            self.assertEqual((people[0].cid, people[0].faceid, people[0].deepid), (0, 'No face', 'No body'))
            np.testing.assert_array_equal(people[0].box_xyxy, [0, 0, 10, 20])


class ResultListTests(unittest.TestCase):
    def person(self, cid, x):
        person = Person(cid, cid, box_xywh=[x, 0, 10, 20], box_xyxy=[x, 0, x + 10, 20],
                        repspoint=(x, 5), deepid=f'Body {cid} 90%', faceid=f'Face {cid} 80%')
        person.misc = ['one', ['nested', cid], 'three']
        return person

    def test_sort_keeps_sparse_frame_order_ties_and_recorded_object_ownership(self):
        recorder = ResIO()
        supplied = [self.person(i, x) for i, x in enumerate((30, 10, 10, 70, 20, 5))]
        for frame, person in zip((5, 5, 5, 20, 20, 99), supplied):
            recorder.addPerson(frame, person)
        stored = list(recorder.people)
        recorder.__sort_people_by_x__()
        self.assertEqual([p.cid for p in recorder.sorted_people], [1, 2, 0, 4, 3, 5])
        self.assertEqual(recorder.frames, ['5', '5', '5', '20', '20', '99'])
        for result, position in zip(recorder.sorted_people, (1, 2, 0, 4, 3, 5)):
            self.assertIs(result, stored[position])
            self.assertIsNot(result, supplied[position])
        recorder.sorted_people[0].misc[1].append('stored edit')
        self.assertEqual(supplied[1].misc[1], ['nested', 1])
        recorder.addPeople(100, [self.person(6, 30), self.person(7, 10)])
        recorder.__sort_people_by_x__()
        self.assertEqual([p.cid for p in recorder.sorted_people], [1, 2, 0, 4, 3, 5, 7, 6])

    def test_dump_modes_ids_and_misc_have_the_expected_text(self):
        for mode in (1, 2, 3):
            for identity in ('deepid', 'faceid', 'all'):
                for include_misc in (False, True):
                    with self.subTest(mode=mode, identity=identity, misc=include_misc), \
                            tempfile.TemporaryDirectory() as folder, \
                            patch('pyppbox.utils.restools.getTimestamp', return_value='fixture'):
                        recorder = ResIO()
                        recorder.addPeople(5, [self.person(0, 30), self.person(1, 10)])
                        if identity == 'all':
                            recorder.dumpAll(folder, mode, include_misc=include_misc, max_misc=1)
                        else:
                            recorder.dump(folder, mode, id_mode=identity, include_misc=include_misc, max_misc=1)
                        expected = []
                        for cid, x in ((1, 10), (0, 30)):
                            columns = ['5', str((x, 5))]
                            columns += [f'Body {cid}', f'Face {cid}'] if identity == 'all' else [
                                f'Body {cid}' if identity == 'deepid' else f'Face {cid}']
                            if mode >= 2:
                                columns.append(str([x, 0, 10, 20]))
                            if mode == 3:
                                columns.append(str([x, 0, x + 10, 20]))
                            if include_misc:
                                columns.append('one')
                            expected.append('\t'.join(columns) + '\n')
                        self.assertEqual((Path(folder) / 'res_fixture_full.txt').read_text(), ''.join(expected))

    def test_empty_results_write_an_empty_file(self):
        for method in ('dump', 'dumpAll'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder, \
                    patch('pyppbox.utils.restools.getTimestamp', return_value='fixture'):
                recorder = ResIO()
                getattr(recorder, method)(folder)
                self.assertEqual((Path(folder) / 'res_fixture_full.txt').read_bytes(), b'')
                self.assertEqual(recorder.sorted_people, [])

    def test_out_of_order_frames_fail_before_overwriting_output_or_publishing_partial_sort(self):
        for method in ('dump', 'dumpAll'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder, \
                    patch('pyppbox.utils.restools.getTimestamp', return_value='fixture'):
                path = Path(folder) / 'res_fixture_full.txt'
                path.write_text('existing output')
                recorder = ResIO()
                recorder.addPerson(20, self.person(0, 10))
                recorder.__sort_people_by_x__()
                previous = recorder.sorted_people
                recorder.addPerson(10, self.person(1, 20))
                with self.assertRaisesRegex(ValueError, 'nondecreasing'):
                    getattr(recorder, method)(folder)
                self.assertEqual(path.read_text(), 'existing output')
                self.assertIs(recorder.sorted_people, previous)
                self.assertEqual(len(recorder.people), 2)

    def test_misaligned_public_lists_fail_before_creating_output(self):
        for frames, people in ((['5'], []), ([], [self.person(0, 10)])):
            for method in ('dump', 'dumpAll'):
                with self.subTest(frames=frames, method=method), tempfile.TemporaryDirectory() as folder:
                    recorder = ResIO()
                    recorder.frames, recorder.people = frames, people
                    with self.assertRaisesRegex(ValueError, 'same length'):
                        getattr(recorder, method)(folder)
                    self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
