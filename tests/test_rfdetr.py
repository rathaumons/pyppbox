"""RF-DETR integration contracts with deterministic model boundaries, no downloads."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from pyppbox.config.configtools import internal_cfg_dir, loadDocumentList, dumpDocDictList
from pyppbox.config.myconfig import DCFGRFDETR, MyConfigurator
from pyppbox.config.unifiedstrings import UnifiedStrings
from pyppbox.modules.detectors.rfdetr import MyRFDETR
from pyppbox.ppb.mt import MT
from pyppbox.utils.persontools import findRepspointBB


class RFDETRTests(unittest.TestCase):
    def make_detector(self, **values):
        cfg = DCFGRFDETR()
        cfg.set(values)
        detector = MyRFDETR.__new__(MyRFDETR)
        detector.cfg = cfg
        detector.model = Mock()
        detector.model.predict.return_value = SimpleNamespace(
            xyxy=np.array([[10, 10, 50, 80], [30, 10, 80, 90], [-10, -20, 200, 160],
                           [0, 0, 2, 20], [60, 30, 90, 50], [np.nan, 0, 30, 40], [80, 50, 70, 90]]),
            confidence=np.array([.5, .99, .9, .8, .98, .7, .6]),
            class_id=np.array([1, 2, 1, 1, 0, 1, 1]))
        return detector

    def test_optional_backend_is_not_imported_by_configs_or_adapter(self):
        code = ('import sys; from pyppbox.config.myconfig import MyConfigurator; '
                'from pyppbox.modules.detectors.rfdetr import MyRFDETR; '
                'cfg=MyConfigurator(set_all_modules=True); '
                'assert cfg.dcfg_rfdetr.person_class_id == 1; assert "rfdetr" not in sys.modules')
        process = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_defaults_aliases_and_config_formats(self):
        for name in ('RF_DETR', 'rf-detr', 'RfDeTr'):
            with self.subTest(name=name):
                cfg = DCFGRFDETR()
                cfg.set({'dt_name': name, 'device': 0, 'optimize': 'False'})
                self.assertEqual(cfg.dt_name, 'RF_DETR')
                self.assertEqual(cfg.device, 0)
                self.assertIs(cfg.optimize, False)
                self.assertEqual(Path(cfg.model_file), Path.cwd() / 'data/modules/rf_detr/rf-detr-nano.pth')
                self.assertEqual(cfg.person_class_id, 1)
        with tempfile.TemporaryDirectory() as directory:
            mapping = dict(dt_name='RF_DETR', model_size='SMALL', person_class_id=0, model_file='weights/model.pth')
            path = Path(directory) / 'detector.json'
            path.write_text(json.dumps(mapping), encoding='utf-8')
            documents = []
            for value in (mapping, json.dumps(mapping), str(path)):
                cfg = DCFGRFDETR()
                cfg.set(value)
                documents.append(cfg.getDocument())
                self.assertEqual(cfg.model_size, 'small')
                self.assertTrue(Path(cfg.model_file).is_absolute())
                restored = DCFGRFDETR()
                restored.set(cfg.getDocument())
                self.assertEqual(restored.model_file, cfg.model_file)
            self.assertEqual(documents[0], documents[1])
            self.assertEqual(documents[0], documents[2])

    def test_local_model_defaults_path_rules_and_cache_opt_in(self):
        bundled = next(doc for doc in loadDocumentList(Path(internal_cfg_dir) / 'detectors.yaml')
                       if doc['dt_name'] == 'RF_DETR')
        self.assertEqual(bundled, DCFGRFDETR.DEFAULTS)
        for size, filename in DCFGRFDETR.MODEL_FILES.items():
            for internal in (False, True):
                with self.subTest(size=size, internal=internal):
                    cfg = DCFGRFDETR(internal)
                    cfg.set({'model_size': size})
                    root = Path(internal_cfg_dir).parents[1] if internal else Path.cwd()
                    self.assertEqual(Path(cfg.model_file), root / 'data/modules/rf_detr' / filename)
                    self.assertEqual(cfg.getDocument()['device'], 0)
        cfg.set({'model_file': '', 'device': 'cpu'})
        self.assertEqual(cfg.model_file, '')
        self.assertEqual(cfg.getDocument()['model_file'], '')
        self.assertEqual(cfg.device, 'cpu')

    def test_invalid_configuration(self):
        for name, value in [('model_size', 'xlarge'), ('model_size', 'seg-nano'), ('max_det', 0),
                            ('max_det', 1.5), ('max_det', float('inf')), ('max_det', None), ('max_det', True), ('person_class_id', -1),
                            ('conf', float('nan')), ('conf', None), ('conf', []), ('conf', 1.1), ('conf', True),
                            ('optimize', 'yes'), ('device', 'cuda:-1'), ('device', True),
                            ('model_file', None), ('iou', .5), ('dt_name', 'YOLO_Classic')]:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                DCFGRFDETR().set({name: value})

    def test_legacy_config_fallback_does_not_rewrite_files(self):
        with tempfile.TemporaryDirectory() as directory:
            docs = [doc for doc in loadDocumentList(Path(internal_cfg_dir) / 'detectors.yaml')
                    if doc['dt_name'] != 'RF_DETR']
            path = Path(directory) / 'detectors.yaml'
            dumpDocDictList(path, docs, '')
            before = path.read_bytes()
            cfg = MyConfigurator(directory)
            cfg.setAllDCFG()
            self.assertEqual(cfg.dt_map, ['YOLO_Classic', 'YOLO_Ultralytics', 'RF_DETR', 'GT'])
            self.assertEqual(cfg.dcfg_rfdetr.model_size, 'nano')
            self.assertEqual(path.read_bytes(), before)
            cfg.setASupportedModuleCFG({'dt_name': 'rf-detr', 'conf': .75})
            self.assertEqual(cfg.dcfg_rfdetr.conf, .75)
        strings = UnifiedStrings()
        legacy = dict(strings.data)
        legacy.pop('rf_detr')
        strings.set(legacy)
        self.assertEqual(strings.getUnifiedFormat('rf-detr'), 'RF_DETR')

    def test_archive_includes_rfdetr_and_matches_defaults(self):
        with zipfile.ZipFile(Path(internal_cfg_dir) / 'cfg.zip') as archive:
            for name in ('main.yaml', 'detectors.yaml'):
                self.assertEqual(archive.read(name).decode().replace('\r\n', '\n'),
                                 (Path(internal_cfg_dir) / name).read_text(encoding='utf-8'))

    def test_initialization_variants_device_and_optimization(self):
        backend = SimpleNamespace(**{name: Mock(return_value=Mock()) for name in
                                    ('RFDETRNano', 'RFDETRSmall', 'RFDETRMedium', 'RFDETRLarge')})
        with patch.dict(sys.modules, {'rfdetr': backend}), patch.object(sys, 'version_info', (3, 11)):
            for size, name in [('nano', 'RFDETRNano'), ('small', 'RFDETRSmall'),
                               ('medium', 'RFDETRMedium'), ('large', 'RFDETRLarge')]:
                cfg = DCFGRFDETR()
                cfg.set({'model_size': size, 'device': 'cuda:1', 'model_file': ''})
                model = MyRFDETR(cfg)
                getattr(backend, name).assert_called_once_with(device='cuda:1')
                model.model.inference.assert_called_once_with(compile=False, inplace=True)
            backend.RFDETRNano.reset_mock()
            cfg.set({'optimize': False, 'device': 'auto', 'model_file': ''})
            MyRFDETR(cfg)
            backend.RFDETRNano.assert_called_once_with()
            backend.RFDETRNano.return_value.inference.assert_not_called()

    def test_optional_dependency_and_missing_checkpoint_errors(self):
        with patch.object(sys, 'version_info', (3, 9)), self.assertRaisesRegex(ImportError, 'Python 3.10'):
            MyRFDETR(DCFGRFDETR())
        with patch.object(sys, 'version_info', (3, 11)):
            automatic = DCFGRFDETR()
            automatic.set({'model_file': ''})
            with patch.dict(sys.modules, {'rfdetr': None}), self.assertRaisesRegex(ImportError, r'pyppbox\[rfdetr\]'):
                MyRFDETR(automatic)
            cfg = DCFGRFDETR()
            cfg.set({'model_file': 'does-not-exist-test.pth'})
            with self.assertRaisesRegex(FileNotFoundError, 'checkpoint not found'):
                MyRFDETR(cfg)

    def test_local_checkpoint_and_integer_device_are_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'nano.pth'
            checkpoint.touch()
            cfg = DCFGRFDETR()
            cfg.set({'model_file': checkpoint.as_posix(), 'device': '0', 'optimize': False})
            backend = SimpleNamespace(RFDETRNano=Mock())
            with patch.dict(sys.modules, {'rfdetr': backend}), patch.object(sys, 'version_info', (3, 11)):
                MyRFDETR(cfg)
            self.assertEqual(cfg.getDocument()['device'], 0)
            backend.RFDETRNano.assert_called_once_with(device='cuda:0', pretrain_weights=checkpoint.as_posix())

    def test_people_color_alignment_filtering_and_max_det(self):
        detector = self.make_detector(max_det=1)
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        frame[:] = [10, 20, 30]
        original = frame.copy()
        people, returned = detector.detectPeople(frame, visual=False)
        self.assertIs(returned, frame)
        np.testing.assert_array_equal(frame, original)
        np.testing.assert_array_equal(detector.model.predict.call_args.args[0][0, 0], [30, 20, 10])
        self.assertEqual(detector.model.predict.call_args.kwargs, {'threshold': .4, 'include_source_image': False})
        self.assertEqual(len(people), 1)
        np.testing.assert_array_equal(people[0].box_xyxy, [0, 0, 120, 100])
        self.assertEqual(people[0].det_conf, .9)
        self.assertEqual(people[0].init_id, 0)
        self.assertEqual(len(people[0].keypoints), 0)
        detector.detectPeople(frame, visual=False)
        self.assertEqual(detector.model.predict.call_count, 2)

    def test_custom_class_general_detection_and_alternative_point(self):
        detector = self.make_detector(person_class_id=0)
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        people, _ = detector.detectPeople(frame, visual=False, alt_repspoint=True, alt_repspoint_top=False)
        self.assertEqual(people[0].det_conf, .98)
        self.assertEqual(people[0].repspoint, findRepspointBB(people[0].box_xyxy, prefer_top=False))
        result = detector.detect(frame, visual=False, classes=[1, 2])
        self.assertEqual(result[4], [.99, .9, .5])
        self.assertEqual([len(items) for items in result[1:]], [3, 3, 3, 3])
        detector.detectPeople(frame, visual=True)
        self.assertTrue(frame.any())

    def test_empty_rejected_and_malformed_results(self):
        detector = self.make_detector(conf=1)
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        self.assertEqual(detector.detectPeople(frame, visual=False)[0], [])
        detector.model.predict.return_value = SimpleNamespace(xyxy=np.empty((0, 4)), confidence=None, class_id=None)
        self.assertEqual(detector.detect(frame, visual=False)[1:], ([], [], [], []))
        for bad in (None, np.zeros((0, 1, 3), dtype=np.uint8), np.zeros((2, 2)),
                    np.zeros((2, 2, 4), dtype=np.uint8), np.zeros((2, 2, 3), dtype=float)):
            with self.subTest(shape=getattr(bad, 'shape', None)), self.assertRaises(ValueError):
                detector.detectPeople(bad)
        detector = self.make_detector()
        detector.model.predict.return_value.confidence = np.array([.5])
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            detector.detectPeople(frame)

    def test_public_pipeline_by_name_dict_json_and_default_main(self):
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        with patch('pyppbox.modules.detectors.rfdetr.MyRFDETR', side_effect=lambda cfg: self.make_detector(**cfg.getDocument())):
            for value in ('RF_DETR', 'rf-detr', {'dt_name': 'RF_DETR'}, '{"dt_name":"RF_DETR"}'):
                with self.subTest(value=value):
                    pipeline = MT()
                    pipeline.setMainDetector(value)
                    people, _ = pipeline.detectPeople(frame, img_is_mat=True)
                    self.assertEqual(len(people), 2)
                    self.assertEqual(pipeline.getMainConfig()['detector'], 'RF_DETR')
            pipeline = MT()
            pipeline.setMainModules({'detector': 'RF_DETR', 'tracker': 'None', 'reider': 'None'})
            people, _ = pipeline.detectPeople(frame, img_is_mat=True)
            self.assertEqual(len(people), 2)

    def test_save_preserves_other_detector_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'detectors.yaml'
            documents = loadDocumentList(Path(internal_cfg_dir) / 'detectors.yaml')
            documents.append({'dt_name': 'FutureDetector', 'custom_value': 'keep_me'})
            dumpDocDictList(path, documents, '')
            cfg = MyConfigurator(directory)
            new = dict(documents[0], conf=.65)
            cfg.dumpDetectorCFG(new)
            expected = [new] + documents[1:]
            self.assertEqual(loadDocumentList(path), expected)
            cfg.dumpDetectorCFG({'dt_name': 'RF_DETR', 'conf': .85})
            self.assertEqual(loadDocumentList(path)[-1], documents[-1])
            self.assertEqual([doc for doc in loadDocumentList(path) if doc['dt_name'] != 'RF_DETR'],
                             [doc for doc in expected if doc['dt_name'] != 'RF_DETR'])


if __name__ == '__main__':
    unittest.main()
