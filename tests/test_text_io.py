"""Unicode text, concurrent exports, and failure cleanup for result/GT files."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
import tempfile
from threading import Barrier
import unittest
from unittest.mock import patch

from pyppbox.utils.gttools import GTIO
from pyppbox.utils.persontools import Person
from pyppbox.utils.restools import ResIO


def person(name, x=5):
    return Person(0, 0, repspoint=(x, 5), deepid=name, faceid=name,
                  box_xywh=[0, 0, 10, 20], box_xyxy=[0, 0, 10, 20])


class BrokenMisc:
    def __str__(self):
        raise RuntimeError('cannot serialize misc')


class UnicodeInputTests(unittest.TestCase):
    def test_gt_preserves_unicode_with_and_without_bom(self):
        names = ['José', '测试', 'សុខា', 'Zoë 🧑']
        for encoding in ('utf-8', 'utf-8-sig'):
            with self.subTest(encoding=encoding), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / '参照.txt'
                path.write_text(''.join(f'7\t(5, 5)\t{name}\t[0 0 10 20]\t[0 0 10 20]\n'
                                        for name in names), encoding=encoding)
                groups, frames, total, initial = GTIO().loadGT(path)
                self.assertEqual([row[2] for row in groups[0]], names)
                self.assertEqual((frames, total, initial), ([7], 4, 7))

    def test_mapping_preserves_unicode_filenames_with_and_without_bom(self):
        pairs = {'视频.mp4': '参照.txt', 'caméra 🧑.mp4': 'សុខា.txt'}
        for encoding in ('utf-8', 'utf-8-sig'):
            with self.subTest(encoding=encoding), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / '映射.txt'
                path.write_text(''.join(f'{video}:{gt}\n' for video, gt in pairs.items()),
                                encoding=encoding)
                loader = GTIO()
                loader.loadInputGTMap(path)
                self.assertEqual(loader.map_dict, pairs)
                for video, gt in pairs.items():
                    self.assertEqual(loader.getGTFileName(str(Path(folder) / video)), gt)

    def test_invalid_utf8_is_rejected_instead_of_changing_identities(self):
        for method, data in (('loadGT', b'0\t(5, 5)\tJos\xe9\n'),
                             ('loadInputGTMap', b'cam\xe9ra.mp4:gt.txt\n')):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'legacy.txt'
                path.write_bytes(data)
                with self.assertRaisesRegex(ValueError, 'decode'):
                    getattr(GTIO(), method)(path)


class ResultExportTests(unittest.TestCase):
    def test_all_export_modes_write_utf8_ids_and_misc(self):
        for mode in (1, 2, 3):
            for identity in ('deepid', 'faceid', 'all'):
                with self.subTest(mode=mode, identity=identity), \
                        tempfile.TemporaryDirectory() as folder:
                    destination = Path(folder) / '结果'
                    destination.mkdir()
                    supplied = person('测试 90%')
                    supplied.faceid = 'សុខា 🧑 80%'
                    supplied.misc = ['café', '備考']
                    recorder = ResIO()
                    recorder.addPerson(0, supplied)
                    if identity == 'all':
                        result = recorder.dumpAll(destination, mode, include_misc=True)
                    else:
                        result = recorder.dump(destination, mode, id_mode=identity, include_misc=True)
                    self.assertIsNone(result)
                    paths = list(destination.iterdir())
                    self.assertEqual(len(paths), 1)
                    text = paths[0].read_bytes().decode('utf-8')
                    columns = text.strip().split('\t')
                    expected_ids = ['测试', 'សុខា 🧑'] if identity == 'all' else [
                        '测试' if identity == 'deepid' else 'សុខា 🧑']
                    self.assertEqual(columns[2:2 + len(expected_ids)], expected_ids)
                    self.assertEqual(columns[-2:], ['café', '備考'])
                    self.assertEqual(len(recorder.people), 1)

    def test_repeated_exports_keep_existing_files_and_each_recording(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch('pyppbox.utils.restools.getTimestamp', return_value='fixture'):
            destination = Path(folder)
            originals = {'res_fixture_full.txt': b'existing first result',
                         'res_fixture_full_1.txt': b'existing second result'}
            for name, data in originals.items():
                (destination / name).write_bytes(data)
            recorder = ResIO()
            for index, method in enumerate(('dump', 'dumpAll', 'dump', 'dumpAll')):
                recorder.addPerson(index, person(f'person{index}'))
                self.assertIsNone(getattr(recorder, method)(destination, dump_mode=1))
                rows = (destination / f'res_fixture_full_{index + 2}.txt').read_text(
                    encoding='utf-8').splitlines()
                self.assertEqual([row.split('\t')[2] for row in rows],
                                 [f'person{i}' for i in range(index + 1)])
            self.assertEqual(len(list(destination.iterdir())), 6)
            for name, data in originals.items():
                self.assertEqual((destination / name).read_bytes(), data)

    def test_concurrent_recorders_do_not_overwrite_each_other(self):
        workers = 12
        ready = Barrier(workers)
        with tempfile.TemporaryDirectory() as folder, \
                patch('pyppbox.utils.restools.getTimestamp', return_value='fixture'):
            def export(index):
                recorder = ResIO()
                recorder.addPerson(0, person(f'person{index}'))
                ready.wait(timeout=15)
                method = recorder.dump if index % 2 else recorder.dumpAll
                method(folder, dump_mode=1)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(export, range(workers)))
            paths = list(Path(folder).iterdir())
            self.assertEqual(len(paths), workers)
            contents = [path.read_text(encoding='utf-8').strip().split('\t') for path in paths]
            self.assertEqual({row[2] for row in contents}, {f'person{i}' for i in range(workers)})
            for row in contents:
                index = int(row[2][len('person'):])
                self.assertEqual(row, ['0', '(5, 5)', f'person{index}']
                                 + ([f'person{index}'] if index % 2 == 0 else []))

    def test_failed_exports_remove_partial_files_and_preserve_existing_results(self):
        for method in ('dump', 'dumpAll'):
            for failure in ('serialization', 'encoding', 'publication'):
                with self.subTest(method=method, failure=failure), \
                        tempfile.TemporaryDirectory() as folder, \
                        patch('pyppbox.utils.restools.getTimestamp', return_value='fixture'):
                    existing = Path(folder) / 'res_fixture_full.txt'
                    existing.write_bytes(b'previous result')
                    recorder = ResIO()
                    recorder.addPerson(0, person('first'))
                    second = person('second', x=10)
                    if failure == 'serialization':
                        second.misc = [BrokenMisc()]
                    elif failure == 'encoding':
                        second.deepid = second.faceid = '\ud800'
                    recorder.addPerson(0, second)
                    injected = (patch('os.replace', side_effect=OSError('cannot publish'))
                                if failure == 'publication' else nullcontext())
                    error = {'serialization': RuntimeError, 'encoding': UnicodeEncodeError,
                             'publication': OSError}[failure]
                    with injected, self.assertRaises(error):
                        getattr(recorder, method)(folder, include_misc=True)
                    self.assertEqual(existing.read_bytes(), b'previous result')
                    self.assertEqual(list(Path(folder).iterdir()), [existing])
                    self.assertEqual(len(recorder.people), 2)


if __name__ == '__main__':
    unittest.main()
