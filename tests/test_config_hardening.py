"""Config validation and save failures must be caught without loading models."""
import copy
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from pyppbox.config import myconfig as m
from pyppbox.config import configtools as tools


class ConfigHardeningTests(unittest.TestCase):
    def setUp(self):
        self.docs = {}
        for filename, key in (('detectors.yaml', 'dt_name'), ('trackers.yaml', 'tk_name'),
                              ('reiders.yaml', 'ri_name')):
            self.docs.update({d[key]: d for d in tools.loadDocumentList(Path(tools.internal_cfg_dir)/filename)})

    def copy_configs(self, directory):
        for name in ('main', 'detectors', 'trackers', 'reiders'):
            shutil.copy2(Path(tools.internal_cfg_dir)/(name+'.yaml'), Path(directory)/(name+'.yaml'))
        return m.MyConfigurator(directory)

    def test_invalid_settings_fail_with_field_names_before_changing_object(self):
        cases = ((m.DCFGYOLOULT, 'YOLO_Ultralytics', 'conf', float('nan')),
                 (m.DCFGYOLOULT, 'YOLO_Ultralytics', 'iou', float('inf')),
                 (m.DCFGYOLOULT, 'YOLO_Ultralytics', 'imgsz', 0),
                 (m.DCFGYOLOULT, 'YOLO_Ultralytics', 'max_det', 1.5),
                 (m.DCFGYOLOULT, 'YOLO_Ultralytics', 'device', 'cuda:-2'),
                 (m.DCFGYOLOCLS, 'YOLO_Classic', 'model_image_size', 0),
                 (m.DCFGYOLOCLS, 'YOLO_Classic', 'nms', -1),
                 (m.DCFGGT, 'GT', 'gt_file', None),
                 (m.TCFGCentroid, 'Centroid', 'max_spread', -1),
                 (m.TCFGSORT, 'SORT', 'max_age', -1),
                 (m.TCFGSORT, 'SORT', 'min_hits', True),
                 (m.TCFGSORT, 'SORT', 'iou_threshold', 'garbage'),
                 (m.TCFGDeepSORT, 'DeepSORT', 'nn_budget', -1),
                 (m.TCFGDeepSORT, 'DeepSORT', 'max_cosine_distance', float('nan')),
                 (m.RCFGFaceNet, 'FaceNet', 'batch_size', 0),
                 (m.RCFGFaceNet, 'FaceNet', 'gpu_mem', 1.1),
                 (m.RCFGFaceNet, 'FaceNet', 'yl_h_calibration', [-1]),
                 (m.RCFGFaceNet, 'FaceNet', 'yl_w_calibration', [0, float('inf')]),
                 (m.RCFGTorchreid, 'Torchreid', 'device', 'cuda:-1'),
                 (m.RCFGTorchreid, 'Torchreid', 'min_confidence', -1),
                 (m.RCFGTorchreid, 'Torchreid', 'model_name', ''))
        for cls, name, field, invalid in cases:
            with self.subTest(name=name, field=field):
                obj = cls(); obj.set(self.docs[name]); before = copy.deepcopy(obj.getDocument())
                raw = dict(self.docs[name], **{field: invalid})
                with self.assertRaisesRegex(ValueError, field):
                    obj.set(raw)
                self.assertEqual(obj.getDocument(), before)

    def test_legacy_numbers_offsets_budget_and_input_ownership(self):
        cases = ((m.DCFGYOLOULT, 'YOLO_Ultralytics', dict(conf='.4', imgsz='640', show_boxes=' FALSE ', device='cuda:1')),
                 (m.TCFGCentroid, 'Centroid', dict(max_spread=12.5)),
                 (m.TCFGSORT, 'SORT', dict(max_age='0', min_hits=0, iou_threshold='.3')),
                 (m.TCFGDeepSORT, 'DeepSORT', dict(nn_budget=None, max_cosine_distance=1.25)),
                 (m.RCFGFaceNet, 'FaceNet', dict(batch_size='1000', yl_h_calibration=[-125.5, 75.5])))
        for cls, name, edits in cases:
            raw = dict(self.docs[name], **edits); before=copy.deepcopy(raw)
            obj=cls(); obj.set(raw)
            self.assertEqual(raw, before)
            for key in ('conf', 'iou_threshold'):
                if key in raw:
                    self.assertIsInstance(getattr(obj, key), (float, int))
            for key in ('imgsz', 'batch_size', 'max_age'):
                if key in raw:
                    self.assertIsInstance(getattr(obj, key), int)
        for device in (None, 'None', '', -1, '-1', '0,1', [0, 1], 'mps', 'xpu:0', 'npu:0'):
            cfg=m.DCFGYOLOULT(); cfg.set(dict(self.docs['YOLO_Ultralytics'], device=device))

    def test_required_fields_are_reported_together(self):
        with self.assertRaisesRegex(ValueError, 'max_age.*min_hits.*iou_threshold'):
            m.TCFGSORT().set({'tk_name': 'SORT'})

    def test_extensions_survive_object_roundtrip_and_targeted_saves(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=self.copy_configs(directory)
            for filename, key, name, method, edits in (
                    ('detectors', 'dt_name', 'YOLO_Ultralytics', 'dumpDetectorCFG', {'conf': .55}),
                    ('trackers', 'tk_name', 'SORT', 'dumpTrackerCFG', {'max_age': 7}),
                    ('reiders', 'ri_name', 'Torchreid', 'dumpReiderCFG', {'min_confidence': .65})):
                path=Path(directory)/(filename+'.yaml'); docs=tools.loadDocumentList(path)
                for doc in docs:
                    doc['extension']={'nested': ['001', False, 'é']}
                docs.insert(1, {key:'CustomModule', 'extension':{'opaque':42}})
                tools.dumpDocDictList(path,docs,'')
                getattr(cfg, method)(dict({key:name}, **edits))
                after=tools.loadDocumentList(path)
                expected=copy.deepcopy(docs)
                next(d for d in expected if d[key]==name).update(edits)
                self.assertEqual(after, expected)
            raw=dict(self.docs['SORT'], extension={'unknown': True})
            obj=m.TCFGSORT(); obj.set(raw)
            self.assertEqual(obj.getDocument()['extension'], raw['extension'])
            main=tools.loadDocument(Path(directory)/'main.yaml'); main['custom']={'value':1}
            tools.dumpDocDict(cfg.main_yaml,main,'')
            cfg.dumpMainCFG(dict(main,tracker='Centroid'))
            self.assertEqual(tools.loadDocument(cfg.main_yaml)['custom'],main['custom'])

    def test_invalid_targeted_save_keeps_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=self.copy_configs(directory); path=Path(cfg.detectors_yaml); before=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'conf'):
                cfg.dumpDetectorCFG({'dt_name':'YOLO_Ultralytics','conf':float('nan')})
            self.assertEqual(path.read_bytes(),before)

    def test_multi_file_save_rolls_back_and_cleans_up_on_second_replace_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=self.copy_configs(directory)
            paths=[Path(cfg.main_yaml),Path(cfg.trackers_yaml)]
            before=[p.read_bytes() for p in paths]
            main=tools.loadDocument(paths[0]); main['tracker']='Centroid'
            trackers=tools.loadDocumentList(paths[1]); trackers[1]['max_age']=7
            replace=tools.os.replace; calls=[]
            def failing(source,destination):
                if str(destination)==str(paths[1].resolve()) and not calls:
                    calls.append(True); raise OSError('injected second replacement failure')
                return replace(source,destination)
            with patch.object(tools.os,'replace',side_effect=failing):
                with self.assertRaisesRegex(ValueError,'injected second'):
                    cfg.dumpConfigs(main=main,trackers=trackers)
            self.assertEqual([p.read_bytes() for p in paths],before)
            self.assertFalse(list(Path(directory).glob('.pyppbox-save-*')))
            cfg.dumpConfigs(main=main,trackers=trackers)
            self.assertEqual(tools.loadDocument(paths[0]), main)
            self.assertEqual(tools.loadDocumentList(paths[1]),trackers)

    def test_multi_file_serialization_failure_changes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=self.copy_configs(directory); path=Path(cfg.main_yaml); before=path.read_bytes()
            with self.assertRaises(ValueError):
                cfg.dumpConfigs(main={'detector':'None','tracker':'None','reider':'None'},
                                trackers=[{'tk_name':'Custom','opaque':object()}])
            self.assertEqual(path.read_bytes(),before)

    @unittest.skipIf(os.name == 'nt', 'POSIX file modes and symlinks')
    def test_group_save_preserves_symlinks_and_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=self.copy_configs(directory)
            path=Path(cfg.main_yaml); target=Path(directory)/'actual-main.yaml'
            path.rename(target); target.chmod(0o640); path.symlink_to(target)
            main={'detector':'None','tracker':'None','reider':'None'}
            cfg.dumpConfigs(main=main,trackers=[self.docs['SORT']])
            self.assertTrue(path.is_symlink())
            self.assertEqual(target.stat().st_mode & 0o777,0o640)
            self.assertEqual(tools.loadDocument(path),main)

    def test_empty_main_and_unnamed_documents_cannot_be_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=m.MyConfigurator(directory)
            for call in (lambda:cfg.dumpConfigs(main={}),lambda:cfg.dumpMainCFG({}),
                         lambda:cfg.dumpAllTCFG([{}])):
                with self.assertRaises(ValueError):
                    call()
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_runtime_preflight_does_not_require_training_or_mapping_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            weights=Path(directory)/'model.pb'; weights.touch()
            obj=m.TCFGDeepSORT(); obj.set(dict(self.docs['DeepSORT'],model_file=str(weights)))
            obj.validateRuntime()
            weights.unlink()
            with self.assertRaisesRegex(ValueError,'model_file'):
                obj.validateRuntime()
            obj=m.DCFGGT(); obj.set(dict(self.docs['GT'],gt_file=str(weights),gt_map_file='missing.txt'))
            weights.touch(); obj.validateRuntime()

    def test_missing_selected_config_is_diagnosed_and_old_rf_config_uses_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=self.copy_configs(directory)
            tools.dumpDocDictList(cfg.trackers_yaml, [], '')
            with self.assertRaisesRegex(ValueError,'trackers.yaml.*SORT'):
                cfg.setMainModules({'detector':'None','tracker':'SORT','reider':'None'})
            tools.dumpDocDictList(cfg.detectors_yaml, [], '')
            cfg.setMainModules({'detector':'RF_DETR','tracker':'None','reider':'None'})
            self.assertEqual(cfg.dcfg_rfdetr.model_size,'nano')
            self.assertEqual(tools.loadDocumentList(cfg.detectors_yaml),[])

    def test_interrupted_save_recovers_and_failed_rollback_retains_backups(self):
        for interrupted in (True, False):
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as directory:
                cfg=self.copy_configs(directory)
                original={p.name:p.read_bytes() for p in Path(directory).glob('*.yaml')}
                main=tools.loadDocument(cfg.main_yaml); main['tracker']='Centroid'
                trackers=tools.loadDocumentList(cfg.trackers_yaml); trackers[1]['max_age']=7
                replace=tools.os.replace
                class Interrupted(BaseException):
                    pass
                def fail(source,destination):
                    if str(source).endswith('1.staged'):
                        raise Interrupted() if interrupted else OSError('write failed')
                    if not interrupted and str(source).endswith('.restore'):
                        raise OSError('rollback failed')
                    return replace(source,destination)
                with patch.object(tools.os,'replace',side_effect=fail):
                    with self.assertRaises(Interrupted if interrupted else ValueError):
                        cfg.dumpConfigs(main=main,trackers=trackers)
                self.assertTrue(list(Path(directory).glob('.pyppbox-save-*')))
                with self.assertRaisesRegex(ValueError,'recovery'):
                    cfg.dumpConfigs(main=main)
                with self.assertRaisesRegex(ValueError,'recovery'):
                    cfg.dumpMainCFG(main)
                tools.recoverConfigSave(directory)
                self.assertEqual({p.name:p.read_bytes() for p in Path(directory).glob('*.yaml')},original)
                self.assertFalse(list(Path(directory).glob('.pyppbox-save-*')))

    @unittest.skipIf(os.name == 'nt', 'POSIX read-only files in a writable config directory')
    def test_recovery_retries_readonly_restore_left_by_a_failed_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.copy_configs(directory)
            original = {path.name:path.read_bytes() for path in Path(directory).glob('*.yaml')}
            for path in Path(directory).glob('*.yaml'):
                path.chmod(0o444)
            replace = tools.os.replace
            def fail(source, destination):
                if str(source).endswith(('1.staged', '.restore')):
                    raise OSError('temporary storage error')
                return replace(source, destination)
            with patch.object(tools.os, 'replace', side_effect=fail), self.assertRaises(ValueError):
                cfg.dumpConfigs(main={'detector':'None','tracker':'None','reider':'None'},
                                trackers=[self.docs['SORT']])
            pending = next(Path(directory).glob('.pyppbox-save-*'))
            self.assertTrue(list(pending.glob('*.restore')))
            tools.recoverConfigSave(directory)
            self.assertFalse(list(Path(directory).glob('.pyppbox-save-*')))
            for path in Path(directory).glob('*.yaml'):
                self.assertEqual(path.read_bytes(), original[path.name])
                self.assertEqual(path.stat().st_mode & 0o777, 0o444)

    def test_runtime_preflight_stops_pipeline_before_constructor_and_classifier_load(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock
        from pyppbox.ppb.mt import MT
        backend=Mock()
        with patch.dict(sys.modules, {'pyppbox.modules.trackers.deepsort':SimpleNamespace(MyDeepSORT=backend)}):
            cfg=dict(self.docs['DeepSORT'],model_file='missing-inference-model-test.pb')
            with self.assertRaisesRegex(ValueError,'model_file'):
                MT().setMainTracker(cfg)
        backend.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            model=Path(directory)/'osnet_ain_ms_d_c.pth.tar'; model.touch()
            backend=Mock(return_value=SimpleNamespace(auto_load=False,load_classifier=Mock()))
            with patch.dict(sys.modules, {'pyppbox.modules.reiders.torchreid':SimpleNamespace(MyTorchreid=backend)}):
                cfg=dict(self.docs['Torchreid'],model_path=str(model),classifier_pkl='missing-classifier-test.pkl')
                pipeline=MT(); pipeline.setMainReIDer(cfg,auto_load=False)
                with self.assertRaisesRegex(ValueError,'classifier_pkl'):
                    pipeline.reidPeople(None,[],img_is_mat=True)
                backend.return_value.load_classifier.assert_not_called()

    def test_interrupted_new_files_are_removed_and_completed_groups_remain(self):
        for complete in (False,True):
            with self.subTest(complete=complete), tempfile.TemporaryDirectory() as directory:
                cfg=m.MyConfigurator(directory)
                class Interrupted(BaseException):
                    pass
                if complete:
                    failure=patch.object(tools,'_clean_save',side_effect=Interrupted())
                else:
                    replace=tools.os.replace
                    def fail(source,destination):
                        if str(source).endswith('1.staged'):
                            raise Interrupted()
                        return replace(source,destination)
                    failure=patch.object(tools.os,'replace',side_effect=fail)
                main={'detector':'None','tracker':'None','reider':'None'}
                with failure, self.assertRaises(Interrupted):
                    cfg.dumpConfigs(main=main,trackers=[self.docs['SORT']])
                tools.recoverConfigSave(directory)
                if complete:
                    self.assertEqual(tools.loadDocument(cfg.main_yaml),main)
                else:
                    self.assertEqual(list(Path(directory).iterdir()),[])


if __name__ == '__main__':
    unittest.main()
