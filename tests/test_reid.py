"""ReID confidence, active config, and duplicate-pass regressions B05/B06/B11."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pyppbox.config.configtools import internal_cfg_dir, loadDocumentList
from pyppbox.modules.reiders.torchreid import MyTorchreid
from pyppbox.ppb.mt import MT
from pyppbox.utils.persontools import Person


class ReIDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        name = 'pyppbox.modules.reiders.facenet'
        # Only TensorFlow graph/model dependencies are replaced. Tests below run
        # the production FaceNet recognize() and MT crop/duplicate paths.
        with patch.dict(sys.modules, {'tensorflow': Mock(),
                                     name + '.origin.facenet': Mock(),
                                     name + '.origin.detect_face': Mock()}):
            cls.face_module = importlib.import_module(name)

    def test_confidence_including_unknown_and_unavailable(self):
        for backend in (MyTorchreid, self.face_module.MyFaceNet):
            for prediction, expected in (((0, 25.), ('Unknown', 25.)),
                                         ((0, 80.), ('Alice', 80.)),
                                         ((0, 35.), ('Alice', 35.)),
                                         ((-1, -1), ('Unavailable', 0.))):
                with self.subTest(backend=backend.__name__, prediction=prediction):
                    reider = backend.__new__(backend)
                    reider.unk, reider.err, reider.min_confidence = 'Unknown', 'Unavailable', 35
                    reider.class_names = reider.pnames = ['Alice']
                    reider.prepare_image = Mock(side_effect=lambda img, **kwargs: img)
                    reider.predict = Mock(return_value=prediction)
                    reider.make_facenet_image = Mock(return_value=np.zeros((1, 160, 160, 3)))
                    for name in ('minsize', 'pnet', 'rnet', 'onet', 'threshold', 'factor'):
                        setattr(reider, name, None)
                    with patch.object(self.face_module.df, 'detect_face',
                                      return_value=(np.array([[0, 0, 10, 10, 1.]]), None)):
                        self.assertEqual(reider.recognize(np.zeros((20, 20, 3))), expected)

    def test_no_face_has_zero_confidence(self):
        reider = self.face_module.MyFaceNet.__new__(self.face_module.MyFaceNet)
        reider.err = 'Unavailable'
        reider.prepare_image = Mock(side_effect=lambda img, **kwargs: img)
        reider.predict = Mock()
        for name in ('minsize', 'pnet', 'rnet', 'onet', 'threshold', 'factor'):
            setattr(reider, name, None)
        with patch.object(self.face_module.df, 'detect_face', return_value=(np.empty((0, 5)), None)):
            self.assertEqual(reider.recognize(np.zeros((20, 20, 3))), ('Unavailable', 0.))
        reider.predict.assert_not_called()

    def test_custom_config_and_updated_duplicate_groups(self):
        configs = loadDocumentList(Path(internal_cfg_dir) / 'reiders.yaml')
        for name in ('FaceNet', 'Torchreid'):
            for use_file in (False, True):
                for deduplicate in (False, True):
                    with self.subTest(name=name, file=use_file, deduplicate=deduplicate):
                        values = dict(next(doc for doc in configs if doc['ri_name'] == name))
                        if name == 'FaceNet':
                            values.update(yl_h_calibration=[-2, 2], yl_w_calibration=[-3, 3])
                        fake = SimpleNamespace(auto_load=True, recognize=Mock(side_effect=[
                            ('same-id', 80.), ('same-id', 90.), ('Alice', 81.), ('Bob', 91.)]))
                        module_name = 'pyppbox.modules.reiders.' + name.lower()
                        module = self.face_module if name == 'FaceNet' else importlib.import_module(module_name)
                        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {module_name: module}), \
                                patch.object(module, 'My' + name, return_value=fake):
                            path = Path(directory) / 'custom.json'
                            path.write_text(json.dumps(values), encoding='utf-8')
                            mt = MT()
                            mt.setMainReIDer(str(path) if use_file else values)
                            # The inactive configurator must not supply crop calibration.
                            mt.__cfg__.rcfg_facenet = SimpleNamespace(yl_h_calibration=[0, 1], yl_w_calibration=[0, 1])
                            people = [Person(i, i, box_xyxy=np.array([0, 0, 20, 30]), repspoint=(10, 10))
                                      for i in range(2)]
                            result, counts = mt.reidPeople(np.zeros((40, 40, 3), dtype=np.uint8), people,
                                                          img_is_mat=True, deduplicate=deduplicate)
                            self.assertEqual(counts, (2, 2) if deduplicate else (2, 0))
                            attribute = 'faceid' if name == 'FaceNet' else 'deepid'
                            self.assertEqual([getattr(p, attribute) for p in result],
                                             ['Alice', 'Bob'] if deduplicate else ['same-id', 'same-id'])
                            if name == 'FaceNet':
                                self.assertTrue(all(call.args[0].shape == (4, 6, 3)
                                                    for call in fake.recognize.call_args_list))


if __name__ == '__main__':
    unittest.main()
