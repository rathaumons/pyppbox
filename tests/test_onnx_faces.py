"""ONNX face configuration, preprocessing, classifier, and pipeline contracts."""
import copy
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from sklearn.svm import SVC

from pyppbox.config.myconfig import RCFGArcFace, RCFGAdaFace, MyConfigurator
from pyppbox.config.unifiedstrings import UnifiedStrings
from pyppbox.modules.reiders.onnxface import MyArcFace, MyAdaFace
from pyppbox.ppb.mt import MT
from pyppbox.utils.persontools import Person


class ONNXFaceTests(unittest.TestCase):
    def test_optional_imports_are_lazy(self):
        code = ('import sys; from pyppbox.config.myconfig import MyConfigurator; '
                'from pyppbox.modules.reiders.onnxface import MyArcFace; '
                'MyConfigurator(set_all_modules=True); '
                'assert not any(x in sys.modules for x in ("tensorflow","onnxruntime","insightface"))')
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_defaults_roundtrip_aliases_and_atomic_validation(self):
        for cls, name in ((RCFGArcFace, 'ArcFace'), (RCFGAdaFace, 'AdaFace')):
            self.assertEqual(UnifiedStrings().getUnifiedFormat(name.swapcase()), name)
            cfg = cls()
            cfg.set({'ri_name': name.lower(), 'device': '0', 'num_threads': '2'})
            self.assertEqual(cfg.device, 'cuda:0')
            restored = cls()
            restored.set(json.dumps(cfg.getDocument()))
            self.assertEqual(cfg.getDocument(), restored.getDocument())
            before = copy.deepcopy(cfg.getDocument())
            for key, value in [('det_size', 321), ('num_threads', True), ('input_std', 0),
                               ('input_mean', float('nan')), ('input_color', 'HSV'),
                               ('device', 'mps'), ('model_file', ''), ('model_det', None),
                               ('det_thresh', 2), ('yl_h_calibration', [0]), ('unknown', 1)]:
                with self.subTest(name=name, field=key), self.assertRaises(ValueError):
                    cfg.set({key: value})
                self.assertEqual(cfg.getDocument(), before)
        configs = MyConfigurator(set_all_modules=True)
        self.assertIn('ArcFace', configs.ri_map)
        self.assertIn('AdaFace', configs.ri_map)

    def fake_backend(self, cls=MyArcFace, cfg=None):
        backend = cls(cfg or (RCFGArcFace() if cls is MyArcFace else RCFGAdaFace()))
        backend.session = Mock()
        backend.input_name = 'images'
        backend.session.run.return_value = [np.array([[3., 4.]], dtype=np.float32)]
        return backend

    def test_preprocessing_channel_order_and_normalization(self):
        for color, expected in [('RGB', [30, 20, 10]), ('BGR', [10, 20, 30])]:
            cfg = RCFGArcFace()
            cfg.set({'input_color': color, 'input_mean': 0, 'input_std': 10})
            backend = self.fake_backend(cfg=cfg)
            bgr = np.full((112, 112, 3), [10, 20, 30], dtype=np.uint8)
            vector = backend.embed_aligned(bgr)
            np.testing.assert_allclose(vector, [.6, .8])
            blob = backend.session.run.call_args.args[1]['images']
            np.testing.assert_allclose(blob[0, :, 0, 0], np.array(expected) / 10)
            backend.embed_aligned(bgr[:, :, ::-1], is_bgr=False)
            np.testing.assert_allclose(backend.session.run.call_args.args[1]['images'], blob)
            for invalid in (np.zeros((10, 10, 3), np.uint8), np.zeros((112, 112)),
                            np.zeros((112, 112, 3), np.float32)):
                with self.assertRaises(ValueError):
                    backend.embed_aligned(invalid)

    def test_bad_features_are_rejected(self):
        backend = self.fake_backend()
        for value in (np.zeros((1, 2)), np.array([[np.nan, 1]]), np.ones((2, 2)), np.ones(2)):
            backend.session.run.return_value = [value]
            with self.assertRaises(ValueError):
                backend.embed_aligned(np.zeros((112, 112, 3), np.uint8))

    def test_recognition_unknown_no_face_and_close(self):
        backend = self.fake_backend()
        with self.assertRaisesRegex(RuntimeError, 'load_classifier'):
            backend.recognize(np.zeros((112, 112, 3), np.uint8))
        backend.model = Mock()
        backend.pnames = ['Zoe', 'Alice']
        backend.model.predict_proba.return_value = np.array([[.9, .1]])
        backend.extract_embedding = Mock(return_value=np.array([.6, .8]))
        self.assertEqual(backend.recognize(None), ('Zoe', 90.))
        backend.model.predict_proba.return_value = np.array([[.6, .4]])
        self.assertEqual(backend.recognize(None), (backend.unk, 60.))
        backend.extract_embedding.return_value = None
        self.assertEqual(backend.recognize(None), (backend.err, 0.))
        backend.close()
        backend.close()
        self.assertIsNone(backend.session)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            backend.recognize(None)

    def test_classifier_fingerprint_and_failed_reload_preserve_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name in ('weights.onnx', 'detector.onnx'):
                (directory / name).write_bytes(name.encode())
            cfg = RCFGArcFace()
            cfg.set(dict(model_file=str(directory / 'weights.onnx'), model_det=str(directory / 'detector.onnx'),
                         classifier_pkl=str(directory / 'svm.pkl')))
            backend = self.fake_backend(cfg=cfg)
            svm = SVC(probability=True, random_state=0).fit([[1, 0], [.9, .1], [0, 1], [.1, .9]], [0, 0, 1, 1])
            payload = dict(format='pyppbox.onnxface.v1', signature=backend._signature(),
                           model=svm, class_names=['Zoe', 'Alice'])
            with open(cfg.classifier_pkl, 'wb') as stream:
                pickle.dump(payload, stream)
            backend.load_classifier()
            previous = backend.model
            self.assertEqual(backend.pnames, ['Zoe', 'Alice'])
            (directory / 'weights.onnx').write_bytes(b'other-checkpoint')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                backend.load_classifier()
            self.assertIs(backend.model, previous)
            with open(cfg.classifier_pkl, 'wb') as stream:
                pickle.dump((svm, ['Zoe', 'Alice']), stream)
            with self.assertRaisesRegex(ValueError, 'legacy'):
                backend.load_classifier()

    def test_pipeline_updates_faceids_and_duplicate_pass_for_both_backends(self):
        for cls, name in ((RCFGArcFace, 'ArcFace'), (RCFGAdaFace, 'AdaFace')):
            with self.subTest(name=name):
                cfg = cls()
                cfg.set(dict(yl_h_calibration=[-4, 4], yl_w_calibration=[-3, 3]))
                fake = SimpleNamespace(auto_load=True, recognize=Mock(side_effect=[
                    ('same', 90), ('same', 91), ('Alice', 92), ('Bob', 93)]), close=Mock())
                with patch.object(cls, 'validateRuntime'), \
                        patch('pyppbox.modules.reiders.onnxface.My' + name, return_value=fake):
                    pipeline = MT()
                    pipeline.setMainReIDer(cfg.getDocument())
                    people = [Person(i, i, box_xyxy=np.array([0, 0, 20, 30]), repspoint=(10, 10)) for i in range(2)]
                    results, counts = pipeline.reidPeople(np.zeros((30, 30, 3), np.uint8), people,
                                                          img_is_mat=True, deduplicate=True)
                    self.assertEqual(counts, (2, 2))
                    self.assertEqual([p.faceid for p in results], ['Alice', 'Bob'])
                    self.assertEqual([p.faceid_conf for p in results], [92, 93])
                    self.assertTrue(all(call.args[0].shape == (8, 6, 3) for call in fake.recognize.call_args_list))
                    pipeline.close()
                    fake.close.assert_called_once()

    def test_training_saves_matching_classifier_and_reports_skipped_faces(self):
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('model.onnx', 'detector.onnx'):
                (root / name).write_bytes(name.encode())
            for name, value in [('Alice', 30), ('Zoe', 220)]:
                folder = root / 'faces' / name
                folder.mkdir(parents=True)
                for i in range(3):
                    cv2.imwrite(str(folder / f'{i}.png'), np.full((20, 20, 3), value + i, np.uint8))
            cv2.imwrite(str(root / 'faces/Alice/no-face.png'), np.zeros((20, 20, 3), np.uint8))
            cfg = RCFGArcFace()
            cfg.set(dict(model_file=str(root / 'model.onnx'), model_det=str(root / 'detector.onnx'),
                         classifier_pkl=str(root / 'classifier.pkl'), train_data=str(root / 'faces')))
            backend = self.fake_backend(cfg=cfg)
            backend.extract_embedding = lambda image, is_bgr=True: (None if not image.any() else
                np.array([image.mean() / 255, 1 - image.mean() / 255], dtype=np.float32))
            backend.train_classifier()
            with open(cfg.classifier_pkl, 'rb') as stream:
                payload = pickle.load(stream)
            self.assertEqual(payload['skipped_no_face'], 1)
            self.assertEqual(payload['samples'], 6)
            self.assertEqual(payload['class_names'], ['Alice', 'Zoe'])
            backend.load_classifier()
            backend.min_confidence = 0
            self.assertEqual(backend.recognize(np.full((20, 20, 3), 30, np.uint8))[0], 'Alice')
            previous = Path(cfg.classifier_pkl).read_bytes()
            backend.extract_embedding = lambda image, is_bgr=True: None
            with self.assertRaisesRegex(ValueError, 'no detectable'):
                backend.train_classifier()
            self.assertEqual(Path(cfg.classifier_pkl).read_bytes(), previous)


if __name__ == '__main__':
    unittest.main()
