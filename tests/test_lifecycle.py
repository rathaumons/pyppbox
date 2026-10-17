# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Pipeline ownership and candidate-first replacement failures."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pyppbox.config.configtools import dumpDocDict, dumpDocDictList, internal_cfg_dir, loadDocumentList
from pyppbox.config.myconfig import RCFGFaceNet, RCFGTorchreid
from pyppbox.ppb.mt import MT
from pyppbox.utils.persontools import Person


class LifecycleTests(unittest.TestCase):
    def person(self,x=10):
        return Person(0,-1,box_xyxy=np.array([x,0,x+20,40]),box_xywh=np.array([x,0,20,40]),repspoint=(x+10,20))

    def test_failed_single_stage_replacement_preserves_pipeline(self):
        for stage, method, module, cls, document in (
                ('dt','setMainDetector','pyppbox.modules.detectors.yoloult','MyYOLOULT',
                 next(d for d in loadDocumentList(Path(internal_cfg_dir)/'detectors.yaml') if d['dt_name']=='YOLO_Ultralytics')),
                ('tk','setMainTracker','pyppbox.modules.trackers.centroid','MyCentroid',{'tk_name':'Centroid','max_spread':30}),
                ('ri','setMainReIDer','pyppbox.modules.reiders.torchreid','MyTorchreid',
                 next(d for d in loadDocumentList(Path(internal_cfg_dir)/'reiders.yaml') if d['ri_name']=='Torchreid'))):
            with self.subTest(stage=stage), patch.object(RCFGTorchreid,'validateRuntime'):
                old=Mock(); pipeline=MT(); factory=Mock(return_value=old)
                with patch.dict(sys.modules,{module:SimpleNamespace(**{cls:factory})}):
                    getattr(pipeline,method)(document)
                    cfg=getattr(pipeline,f'__{stage}_cfg__'); owner=pipeline.getConfig()
                    factory.side_effect=RuntimeError('injected constructor failure')
                    with self.assertRaisesRegex(RuntimeError,'injected'):
                        getattr(pipeline,method)(document)
                    self.assertIs(getattr(pipeline,f'__{stage}__'),old)
                    self.assertIs(getattr(pipeline,f'__{stage}_cfg__'),cfg)
                    self.assertTrue(getattr(pipeline,f'__{stage}_is_set__'))
                    self.assertIs(pipeline.getConfig(),owner)
                    old.close.assert_not_called()

    def test_failed_group_load_disposes_candidates_and_preserves_config_and_gt_mode(self):
        pipeline=MT(); pipeline.setMainModules({'detector':'None','tracker':'Centroid','reider':'None'})
        old_tracker=pipeline.__tk__; owner=pipeline.getConfig(); before=copy.deepcopy(owner.getMCFG())
        detector=Mock(); tracker=Mock()
        with patch('pyppbox.modules.detectors.yoloult.MyYOLOULT',return_value=detector), \
                patch('pyppbox.modules.trackers.centroid.MyCentroid',return_value=tracker), \
                patch.object(RCFGFaceNet,'validateRuntime',side_effect=RuntimeError('last stage failure')):
            with self.assertRaisesRegex(RuntimeError,'last stage'):
                pipeline.setMainModules({'detector':'YOLO_Ultralytics','tracker':'Centroid','reider':'FaceNet'})
        self.assertIs(pipeline.__tk__,old_tracker)
        self.assertEqual(pipeline.getMainConfig(),{'detector':'None','tracker':'Centroid','reider':'None'})
        self.assertIs(pipeline.getConfig(),owner); self.assertEqual(owner.getMCFG(),before)
        detector.close.assert_called_once(); tracker.close.assert_called_once()

    def test_invalid_inputs_and_config_only_failure_preserve_active_state(self):
        pipeline=MT(); pipeline.setMainTracker({'tk_name':'Centroid','max_spread':30})
        old=pipeline.__tk__; owner=pipeline.getConfig()
        for action in (lambda:pipeline.setMainTracker({'tk_name':'Centroid','max_spread':-1}),
                       lambda:pipeline.setMainTracker('unknown-tracker'),
                       lambda:pipeline.setMainModules({'detector':'None'},load_all=False)):
            with self.assertRaises(ValueError): action()
            self.assertIs(pipeline.__tk__,old); self.assertIs(pipeline.getConfig(),owner)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): pipeline.setConfigDir(directory)
            self.assertIs(pipeline.__tk__,old); self.assertIs(pipeline.getConfig(),owner)

    def test_successful_replacement_retires_old_stage_and_preserves_configurator_reference(self):
        pipeline=MT(); owner=pipeline.getConfig(); old=Mock(); new=Mock()
        with patch('pyppbox.modules.trackers.centroid.MyCentroid',side_effect=[old,new]):
            pipeline.setMainTracker({'tk_name':'Centroid','max_spread':30})
            pipeline.setMainTracker({'tk_name':'Centroid','max_spread':31})
        old.close.assert_called_once(); self.assertIs(pipeline.__tk__,new)
        self.assertIs(pipeline.getConfig(),owner)
        pipeline.close(); new.close.assert_called_once(); pipeline.close(); new.close.assert_called_once()

    def test_context_close_and_closed_instance_requires_new_pipeline(self):
        with MT() as pipeline:
            pipeline.setMainTracker({'tk_name':'Centroid','max_spread':30})
        pipeline.close()
        for action in (lambda:pipeline.reset(),lambda:pipeline.setMainTracker('None'),
                       lambda:pipeline.trackPeople(None,[],img_is_mat=True)):
            with self.assertRaisesRegex(RuntimeError,'closed'): action()

    def test_stream_reset_clears_metadata_and_leaves_other_pipeline_alone(self):
        for name,config in (('Centroid',{'tk_name':'Centroid','max_spread':30}),
                            ('SORT',{'tk_name':'SORT','max_age':2,'min_hits':0,'iou_threshold':.3})):
            with self.subTest(name=name):
                one=MT(); two=MT(); one.setMainTracker(config); two.setMainTracker(config)
                first=one.trackPeople(None,[self.person()],img_is_mat=True)[0]; first.deepid='Alice';first.misc='old'
                second=two.trackPeople(None,[self.person()],img_is_mat=True)[0]; second.deepid='Bob'
                backend=one.__tk__; one.reset()
                self.assertIs(one.__tk__,backend)
                self.assertEqual(first.deepid,'Alice'); self.assertEqual(first.misc,'old')
                current=one.trackPeople(None,[self.person()],img_is_mat=True)[0]
                self.assertNotEqual(current.deepid,'Alice'); self.assertNotEqual(current.misc,'old')
                self.assertEqual(two.trackPeople(None,[self.person()],img_is_mat=True)[0].deepid,'Bob')
                one.close();two.close()

    def test_cleanup_failures_retry_without_masking_body_or_undoing_success(self):
        pipeline=MT(); old=Mock(); old.close.side_effect=[RuntimeError('busy'),None]; replacement=Mock()
        with patch('pyppbox.modules.trackers.centroid.MyCentroid',side_effect=[old,replacement]):
            pipeline.setMainTracker({'tk_name':'Centroid','max_spread':30})
            pipeline.setMainTracker({'tk_name':'Centroid','max_spread':31})
        self.assertIs(pipeline.__tk__,replacement)
        pipeline.close(); self.assertEqual(old.close.call_count,2); replacement.close.assert_called_once()
        pipeline=MT(); failing=Mock(); successful=Mock()
        failing.close.side_effect=RuntimeError('busy')
        pipeline.__dt__=failing; pipeline.__tk__=successful
        with self.assertRaisesRegex(ValueError,'body failure'):
            with pipeline: raise ValueError('body failure')
        successful.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError,'could not close'): pipeline.close()
        failing.close.side_effect=None; pipeline.close(); pipeline.close()

    def test_reset_preserves_gt_override_models_and_restarts_fallback_sequence(self):
        from pyppbox.utils.evatools import TKOReider
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'gt.txt'; path.write_text('',encoding='utf-8')
            pipeline=MT(); pipeline.setMainDetector({'dt_name':'GT','gt_file':str(path),'gt_map_file':'unused.txt'})
            pipeline.setMainTracker({'tk_name':'Centroid','max_spread':30})
            pipeline.setMainReIDer({'ri_name':'None'}); pipeline.forceFullGTMode()
            fallback=pipeline.__ri__; self.assertIsInstance(fallback,TKOReider)
            first=fallback.recognize(None); fallback.recognize(None)
            detector=pipeline.__dt__; detector.current_frame=87
            pipeline.reset(frame_index=12)
            self.assertIs(pipeline.__ri__,fallback); self.assertIs(pipeline.__dt__,detector)
            self.assertEqual(fallback.recognize(None),first); self.assertEqual(detector.current_frame,12)
            self.assertFalse(detector.detect_only)
            pipeline.setMainModules({'detector':'None','tracker':'None','reider':'None'},load_all=False)
            self.assertIs(pipeline.__dt__,detector); self.assertFalse(detector.detect_only)
            for invalid in (-1,1.5,True):
                with self.assertRaises(ValueError): pipeline.reset(invalid)
            pipeline.close()

    def test_training_failure_keeps_active_reider_and_constructs_only_once(self):
        cfg=next(d for d in loadDocumentList(Path(internal_cfg_dir)/'reiders.yaml') if d['ri_name']=='Torchreid')
        old=Mock(); candidate=Mock(); candidate.train_classifier.side_effect=RuntimeError('training failed')
        factory=Mock(side_effect=[old,candidate])
        with patch.dict(sys.modules,{'pyppbox.modules.reiders.torchreid':SimpleNamespace(MyTorchreid=factory)}), \
                patch.object(RCFGTorchreid,'validateRuntime'):
            pipeline=MT(); pipeline.setMainReIDer(cfg)
            with self.assertRaisesRegex(RuntimeError,'training failed'): pipeline.trainReIDClassifier(cfg)
            self.assertIs(pipeline.__ri__,old); old.close.assert_not_called(); candidate.close.assert_called_once()
            self.assertEqual(factory.call_count,2)

    def test_named_and_directory_replacements_use_selected_config_and_retire_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for filename in ('detectors','trackers','reiders'):
                dumpDocDictList(root/(filename+'.yaml'),loadDocumentList(Path(internal_cfg_dir)/(filename+'.yaml')),'')
            dumpDocDict(root/'main.yaml',{'detector':'None','tracker':'SORT','reider':'None'},'')
            pipeline=MT(); owner=pipeline.getConfig()
            pipeline.setConfigDir(directory,load_all=True)
            self.assertEqual(pipeline.getMainConfig()['tracker'],'SORT')
            old=pipeline.__tk__
            with patch('pyppbox.modules.trackers.centroid.MyCentroid',side_effect=RuntimeError('failed')):
                with self.assertRaises(RuntimeError): pipeline.setMainTracker('Centroid')
            self.assertIs(pipeline.__tk__,old); self.assertIs(owner,pipeline.getConfig())
            pipeline.setMainTracker('Centroid'); self.assertEqual(pipeline.getMainConfig()['tracker'],'Centroid')
            pipeline.close()

    def test_failed_group_does_not_change_existing_gt_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'gt.txt';path.write_text('',encoding='utf-8')
            pipeline=MT();pipeline.setMainDetector({'dt_name':'GT','gt_file':str(path),'gt_map_file':'unused.txt'})
            pipeline.forceFullGTMode();old=pipeline.__dt__
            with patch.object(RCFGFaceNet,'validateRuntime',side_effect=RuntimeError('last stage failure')):
                with self.assertRaises(RuntimeError):
                    pipeline.setMainModules({'detector':'None','tracker':'Centroid','reider':'FaceNet'})
            self.assertIs(pipeline.__dt__,old);self.assertFalse(old.detect_only);pipeline.close()


if __name__=='__main__': unittest.main()
