# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Session ownership with deterministic model boundaries (no model downloads)."""
from contextlib import nullcontext
import importlib
import importlib.util
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pyppbox.config.configtools import internal_cfg_dir, loadDocumentList
from pyppbox.config.myconfig import RCFGFaceNet


class BackendLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        name='pyppbox.modules.reiders.facenet'
        with patch.dict(sys.modules,{'tensorflow':Mock(),name+'.origin.facenet':Mock(),name+'.origin.detect_face':Mock()}):
            cls.face=importlib.import_module(name)

    def tensorflow(self,sessions):
        tensor=Mock(); tensor.get_shape.return_value=(None,2)
        graph=Mock();graph.as_default.return_value=nullcontext();graph.get_tensor_by_name.return_value=tensor
        for session in sessions: session.as_default.return_value=nullcontext()
        return SimpleNamespace(Graph=Mock(return_value=graph),compat=SimpleNamespace(v1=SimpleNamespace(
            GPUOptions=Mock(),ConfigProto=Mock(),Session=Mock(side_effect=sessions),get_default_graph=Mock(return_value=graph))),autograph=Mock())

    def face_config(self,path):
        cfg=RCFGFaceNet()
        document=next(d for d in loadDocumentList(Path(internal_cfg_dir)/'reiders.yaml') if d['ri_name']=='FaceNet')
        cfg.set(dict(document,classifier_pkl=str(path)))
        return cfg

    def test_failed_facenet_reload_closes_candidate_and_keeps_working_classifier(self):
        sessions=[Mock(),Mock(),Mock()]
        with tempfile.TemporaryDirectory() as directory, patch.object(self.face,'tf',self.tensorflow(sessions)), \
                patch.object(self.face.df,'create_mtcnn',return_value=(None,None,None)), patch.object(self.face.fn,'load_model'):
            path=Path(directory)/'classifier.pkl';path.write_bytes(pickle.dumps(({'classifier':1},['Alice'])))
            reider=self.face.MyFaceNet(self.face_config(path),auto_load=True)
            old_model=reider.model
            with patch.object(self.face.fn,'load_model',side_effect=RuntimeError('broken graph')):
                with self.assertRaisesRegex(RuntimeError,'broken graph'): reider.load_classifier()
            sessions[1].close.assert_called_once();sessions[0].close.assert_not_called()
            self.assertIs(reider.sess,sessions[0]);self.assertIs(reider.model,old_model)
            reider.load_classifier();sessions[0].close.assert_called_once()
            self.assertIs(reider.sess,sessions[2]);reider.close();reider.close()
            sessions[2].close.assert_called_once();self.assertIsNone(reider.model)
            with self.assertRaisesRegex(RuntimeError,'closed'): reider.train_classifier()
            with self.assertRaisesRegex(RuntimeError,'closed'): reider.recognize(None)

    def test_facenet_constructor_and_training_failure_close_their_sessions(self):
        sessions=[Mock(),Mock()]
        with tempfile.TemporaryDirectory() as directory, patch.object(self.face,'tf',self.tensorflow(sessions)), \
                patch.object(self.face.df,'create_mtcnn',side_effect=RuntimeError('MTCNN failed')):
            cfg=self.face_config(Path(directory)/'unused.pkl')
            with self.assertRaisesRegex(RuntimeError,'MTCNN failed'): self.face.MyFaceNet(cfg,auto_load=True)
            sessions[0].close.assert_called_once()
            reider=self.face.MyFaceNet(cfg,auto_load=False)
            with patch.object(self.face.fn,'get_dataset',side_effect=RuntimeError('dataset failed')):
                with self.assertRaisesRegex(RuntimeError,'dataset failed'): reider.train_classifier()
            sessions[1].close.assert_called_once(); reider.close()

    def test_training_cleanup_failure_retains_session_and_original_error(self):
        session=Mock();session.close.side_effect=[RuntimeError('busy'),None]
        with tempfile.TemporaryDirectory() as directory, patch.object(self.face,'tf',self.tensorflow([session])):
            reider=self.face.MyFaceNet(self.face_config(Path(directory)/'unused.pkl'))
            with patch.object(self.face.fn,'get_dataset',side_effect=ValueError('dataset failure')):
                with self.assertRaisesRegex(ValueError,'dataset failure'): reider.train_classifier()
            self.assertEqual(reider._pending_sessions,[session]);reider.close()
            self.assertEqual(session.close.call_count,2);self.assertEqual(reider._pending_sessions,[])

    def test_deepsort_graph_failure_closes_partially_created_session(self):
        # Load an isolated real encoder module, without replacing the module used by
        # tracker tests' deterministic encoding fixture.
        import pyppbox
        path=Path(pyppbox.__file__).parent/'modules/trackers/deepsort/origin/generate_detections.py'
        session=Mock(); tensorflow=self.tensorflow([session])
        tensorflow.io=SimpleNamespace(gfile=SimpleNamespace(GFile=Mock(side_effect=RuntimeError('graph read failed'))))
        spec=importlib.util.spec_from_file_location('lifecycle_encoder_fixture',path)
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'tensorflow':tensorflow}):
            spec.loader.exec_module(module)
            with self.assertRaisesRegex(RuntimeError,'graph read failed'): module.ImageEncoder('unused.pb')
        session.close.assert_called_once()

    def test_deepsort_reset_retains_encoder_and_releases_it_once(self):
        import test_tracking
        tracker=test_tracking.TrackingTests().make_deepsort(); encoder=tracker.encoder
        tracker.metric.samples={7:['feature']}; tracker._people_by_cid={7:object()}
        tracker.current_frame=40;tracker.reset()
        self.assertIs(tracker.encoder,encoder);self.assertEqual(tracker.metric.samples,{})
        self.assertEqual(tracker._people_by_cid,{});self.assertEqual(tracker.current_frame,0)
        tracker.close();tracker.close();encoder.close.assert_called_once()

    def test_deepsort_later_constructor_failure_closes_created_encoder(self):
        import test_tracking
        tracker=test_tracking.TrackingTests().make_deepsort();tracker.close()
        encoder=Mock()
        # The fixture's isolated module may no longer be in sys.modules; patch the
        # constructor's globals directly instead of importing a second real encoder.
        with patch.dict(type(tracker).__init__.__globals__, {
                'gdet':SimpleNamespace(create_box_encoder=Mock(return_value=encoder)),
                'DSTracker':Mock(side_effect=RuntimeError('tracker failed'))}):
            with self.assertRaisesRegex(RuntimeError,'tracker failed'):
                type(tracker)(SimpleNamespace(nms_max_overlap=1.,model_file='unused.pb',max_cosine_distance=.2,nn_budget=100))
        encoder.close.assert_called_once()


if __name__=='__main__': unittest.main()
