"""Compatibility and save-integrity regressions for B07/B08 (no model assets)."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from pyppbox.config import configtools as cfg
from pyppbox.config.myconfig import MyConfigurator, MainCFG, TCFGSORT, DCFGYOLOULT
from pyppbox.config.unifiedstrings import UnifiedStrings
from pyppbox.ppb.mt import MT


class ConfigTests(unittest.TestCase):
    def test_case_insensitive_module_name_normalization(self):
        strings = UnifiedStrings()
        for expected in ('FaceNet', 'Torchreid', 'GT', 'Centroid', 'SORT', 'DeepSORT',
                         'YOLO_Classic', 'YOLO_Ultralytics', 'None'):
            for value in (expected.lower(), expected.upper(), expected.swapcase()):
                with self.subTest(value=value):
                    self.assertEqual(strings.getUnifiedFormat(value), expected)
        for value in ('CustomReID', 'FaceNetV2', 17):
            self.assertEqual(strings.getUnifiedFormat(value), str(value))
        for value, expected in (('fAcEnEt', 'FaceNet'), ('tOrChReId', 'Torchreid')):
            with self.subTest(main_reider=value):
                config = MainCFG()
                config.set({'detector': 'none', 'tracker': 'sort', 'reider': value})
                self.assertEqual(config.reider, expected)

    def test_single_document_forms(self):
        expected = {'tk_name': 'SORT'}
        for value in (expected, [expected], json.dumps(expected), str([expected]),
                      'tk_name: SORT', '---\ntk_name: SORT\n---\n'):
            with self.subTest(value=value):
                self.assertEqual(cfg.getCFGDict(value), expected)
                self.assertEqual(cfg.getCFGDictList(value), [expected])

    def test_multi_document_forms(self):
        expected = [{'tk_name': 'SORT'}, {'tk_name': 'DeepSORT'}]
        for value in (expected, str(expected), 'tk_name: SORT\n---\ntk_name: DeepSORT'):
            with self.subTest(value=value):
                self.assertEqual(cfg.getCFGDictList(value), expected)
                with self.assertRaises(ValueError):
                    cfg.getCFGDict(value)

    def test_empty_and_invalid_documents(self):
        for value in ('', '[]', '---\n', []):
            self.assertEqual(cfg.getCFGDict(value), {})
            self.assertEqual(cfg.getCFGDictList(value), [])
        for value in ('[broken', 'tk_name: [SORT', 'x: y\n---\nx: [broken',
                      'plain-scalar', '123', '[1, 2]', '[[{"x": 1}]]', 12):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    cfg.getCFGDictList(value)
        self.assertFalse(cfg.isDictString('[broken'))
        self.assertTrue(cfg.isDictString('{"tk_name": "SORT"}'))

    def test_inline_filenames_are_values(self):
        for raw in ('model_file: file.json', '{"model_file": "file.yaml"}',
                    "[{'model_file': 'file.json'}]"):
            self.assertIn(cfg.getCFGDict(raw)['model_file'], ('file.json', 'file.yaml'))

    def test_file_formats_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            for extension in ('json', 'yaml', 'yml', 'JSON'):
                path = Path(directory) / ('config.' + extension)
                path.write_text('{"tk_name": "SORT"}', encoding='utf-8')
                self.assertEqual(cfg.getCFGDict(path), {'tk_name': 'SORT'})
                self.assertEqual(cfg.getCFGDictList(str(path)), [{'tk_name': 'SORT'}])
                path.write_text('[{"tk_name": "SORT"}, {"tk_name": "None"}]', encoding='utf-8')
                self.assertEqual(len(cfg.getCFGDictList(path)), 2)
                path.write_text('tk_name: [broken', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'config'):
                    cfg.loadDocumentList(path)
            with self.assertRaisesRegex(ValueError, 'absent'):
                cfg.getCFGDict(Path(directory) / 'absent.yaml')

    def test_atomic_round_trip(self):
        values = {'model_file': 'models/a #1.pt', 'boolean_string': 'false',
                  'number_string': '001', 'unicode': '\u00e9\u4eba', 'colon': 'x: y',
                  'flag': False, 'values': [1, 'false', True], 'nested': {'a': 'null'}}
        with tempfile.TemporaryDirectory() as directory:
            for extension in ('yaml', 'json'):
                path = Path(directory) / ('config.' + extension)
                cfg.dumpDocDict(path, values, '# Header\n')
                self.assertEqual(cfg.loadDocument(path), values)
                cfg.dumpDocDictList(path, [values, {'tk_name': 'SORT'}], '# Header\n')
                self.assertEqual(cfg.loadDocumentList(path), [values, {'tk_name': 'SORT'}])
            self.assertEqual(sorted(p.suffix for p in Path(directory).iterdir()), ['.json', '.yaml'])

    def test_failed_save_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text('original: content\n', encoding='utf-8')
            for failure in ('serialization', 'replacement'):
                with self.subTest(failure=failure):
                    if failure == 'serialization':
                        with self.assertRaises(ValueError):
                            cfg.dumpDocDict(path, {'bad': object()}, '')
                    else:
                        with patch.object(cfg.os, 'replace', side_effect=OSError('injected failure')):
                            with self.assertRaisesRegex(ValueError, 'injected failure'):
                                cfg.dumpDocDict(path, {'new': 'content'}, '')
                    self.assertEqual(path.read_text(encoding='utf-8'), 'original: content\n')
                    self.assertEqual(list(Path(directory).iterdir()), [path])

    @unittest.skipIf(os.name == 'nt', 'POSIX file modes and symlinks')
    def test_atomic_save_preserves_symlink_and_file_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'target.yaml'
            target.write_text('before: save\n', encoding='utf-8')
            target.chmod(0o640)
            link = Path(directory) / 'config.yaml'
            link.symlink_to(target)
            cfg.dumpDocDict(link, {'after': 'save'}, '')
            self.assertTrue(link.is_symlink())
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)
            self.assertEqual(cfg.loadDocument(link), {'after': 'save'})

    def test_shipped_configs_and_paths(self):
        configurator = MyConfigurator()
        configurator.setMainModules()
        configurator.setAllDCFG()
        configurator.setAllTCFG()
        configurator.setAllRCFG()
        # All four files must keep their native mapping/multi-document layouts.
        for name in ('main', 'detectors', 'trackers', 'reiders'):
            path = Path(cfg.internal_cfg_dir) / (name + '.yaml')
            expected = list(yaml.safe_load_all(path.read_text(encoding='utf-8')))
            self.assertEqual(cfg.loadDocumentList(path), expected)
        self.assertTrue(Path(configurator.dcfg_yolout.model_file).is_absolute())
        detector = DCFGYOLOULT()
        detector_values = next(doc for doc in cfg.loadDocumentList(Path(cfg.internal_cfg_dir) / 'detectors.yaml')
                               if doc['dt_name'] == 'YOLO_Ultralytics')
        detector_values['model_file'] = 'models/example.pt'
        detector.set(detector_values)
        self.assertEqual(detector.model_file.replace('\\', '/'),
                         str(Path('models/example.pt').absolute()).replace('\\', '/'))
        tracker = TCFGSORT()
        tracker.set("[{'tk_name': 'SORT', 'max_age': 7, 'min_hits': 3, 'iou_threshold': 0.3}]")
        self.assertEqual(tracker.max_age, 7)

    def test_public_setters_accept_same_forms(self):
        with tempfile.TemporaryDirectory() as directory:
            for setter, key in (('setMainDetector', 'dt_name'), ('setMainTracker', 'tk_name'),
                                ('setMainReIDer', 'ri_name')):
                mapping = {key: 'None'}
                path = Path(directory) / (key + '.json')
                path.write_text(json.dumps(mapping), encoding='utf-8')
                for value in (mapping, [mapping], str([mapping]), json.dumps(mapping),
                              f'{key}: None', path, str(path)):
                    with self.subTest(setter=setter, value=value):
                        mt = MT()
                        getattr(mt, setter)(value)
                        self.assertEqual(set(mt.getMainConfig().values()), {'None'})
                with self.assertRaises(ValueError):
                    getattr(MT(), setter)(f'{key}: [broken')

    def test_yolo_show_boxes_normalizes_legacy_strings(self):
        template = next(doc for doc in cfg.loadDocumentList(Path(cfg.internal_cfg_dir) / 'detectors.yaml')
                        if doc['dt_name'] == 'YOLO_Ultralytics')
        with tempfile.TemporaryDirectory() as directory:
            for value, expected in ((False, False), (True, True), ('False', False), ('True', True),
                                    ('false', False), ('true', True), (' FALSE ', False)):
                values = dict(template, show_boxes=value)
                for extension in ('yaml', 'json'):
                    path = Path(directory) / ('detector.' + extension)
                    cfg.dumpDocDict(path, values, '')
                    for source in (values, path, json.dumps(values)):
                        with self.subTest(value=value, extension=extension, source=type(source).__name__):
                            detector = DCFGYOLOULT()
                            detector.set(source)
                            self.assertIs(detector.show_boxes, expected)
                            self.assertIs(detector.getDocument()['show_boxes'], expected)
                            # Normalizing a known boolean field must not rewrite caller data.
                            self.assertEqual(values['show_boxes'], value)

    def test_yolo_show_boxes_rejects_invalid_values_with_context(self):
        template = next(doc for doc in cfg.loadDocumentList(Path(cfg.internal_cfg_dir) / 'detectors.yaml')
                        if doc['dt_name'] == 'YOLO_Ultralytics')
        for value in ('not-a-bool', '', None, 0, 1, [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'show_boxes'):
                DCFGYOLOULT().set(dict(template, show_boxes=value))


if __name__ == '__main__':
    unittest.main()
