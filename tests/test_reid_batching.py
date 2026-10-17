# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Bounded batches must preserve crop bytes, identity mapping, and per-person counts."""
from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from pyppbox.modules.reiders.torchreid import MyTorchreid
from pyppbox.ppb.mt import MT
from pyppbox.utils.persontools import Person


class Features:
    def __init__(self, values): self.values=np.asarray(values)
    def cpu(self): return self
    def numpy(self): return self.values


class Extractor:
    def __init__(self): self.sizes=[];self.pixels=[]
    def __call__(self, images):
        images=images if isinstance(images,list) else [images]
        self.sizes.append(len(images));self.pixels.extend(image.copy() for image in images)
        return Features([[int(image[0,0,0])] for image in images])


class Scripted:
    auto_load=True
    def __init__(self, fail=None):self.seen=Counter();self.batches=[];self.fail=fail;self.scalars=0
    def value(self,image):
        marker=int(image[0,0,0])
        if marker==self.fail: raise RuntimeError('poisoned crop')
        sequence={1:['A','B','C','D'],2:['A','B','C','D'],3:['B','C','D','E'],4:['B','C','D','E']}.get(marker,['Z']*4)
        result=sequence[min(self.seen[marker],len(sequence)-1)];self.seen[marker]+=1
        return result,80.+marker
    def recognize(self,image,is_bgr=True):self.scalars+=1;return self.value(image)
    def recognize_batch(self,images,is_bgr=True,batch_size=16):
        self.batches.append(len(images))
        if any(int(image[0,0,0])==self.fail for image in images):raise RuntimeError('batch failed')
        return [self.value(image) for image in images]


class ReIDBatchingTests(unittest.TestCase):
    def reider(self):
        obj=MyTorchreid.__new__(MyTorchreid);obj._closed=False
        obj.unk='Unknown';obj.err='Unavailable';obj.min_confidence=70
        obj.class_names=['Alpha_identity','Beta_identity']
        obj.extractor=Extractor()
        obj.model=SimpleNamespace(predict_proba=lambda features:np.array([[.8,.2] if int(row[0])%2==0 else [.3,.7] for row in features]))
        return obj

    def pipeline(self,recognizer):
        obj=MT();obj.__ri_is_set__=True;obj.__ri__=recognizer
        obj.__ri_cfg__=SimpleNamespace(ri_name='Torchreid',model_wh=(8,16))
        return obj

    def fixture(self, alias=False, invalid=False):
        frame=np.zeros((32,80,3),dtype=np.uint8)
        people=[]
        for i in range(1,5):
            x=(i-1)*16;frame[:,x:x+16]=i
            people.append(Person(i,i,box_xyxy=[x,0,x+16,32],deepid='B' if i==3 else 'Unknown'))
        if alias:people=[people[0],people[0]]
        if invalid:people[3].box_xyxy=[200,0,220,10]
        return frame,people

    def test_direct_streamed_inputs_are_bounded_and_match_scalar_pixels_and_labels(self):
        images=[np.tile(np.array([i,i+20,i+40],dtype=np.uint8),(10+i,11,1))[:,::-1] for i in range(7)]
        for image in images:image.flags.writeable=False
        batched=self.reider();batched.min_confidence=65 # keep this test away from the decision guard
        scalar=self.reider();scalar.min_confidence=65;expected=[scalar.recognize(image) for image in images]
        consumed=[]
        def inputs():
            for index,image in enumerate(images):
                if index>=3:self.assertTrue(batched.extractor.sizes)
                consumed.append(index);yield image
        actual=batched.recognize_batch(inputs(),batch_size=3)
        self.assertEqual(actual,expected);self.assertEqual(batched.extractor.sizes,[3,3,1])
        for one,two in zip(scalar.extractor.pixels,batched.extractor.pixels):np.testing.assert_array_equal(one,two)
        np.testing.assert_array_equal(batched.extractor.pixels[0],images[0][:,:,::-1])
        self.assertEqual(consumed,list(range(7)))
        rgb=self.reider();rgb.min_confidence=65
        rgb.recognize_batch(images[:2],is_bgr=False,batch_size=2)
        np.testing.assert_array_equal(rgb.extractor.pixels[0],images[0])

    def test_threshold_and_winner_boundaries_recheck_scalar_decisions(self):
        images=[np.zeros((4,4,3),dtype=np.uint8)]*2
        obj=self.reider()
        obj.model.predict_proba=lambda features:np.tile(
            [.70000001,.29999999] if len(features)>1 else [.69999999,.30000001],(len(features),1))
        expected=[obj.recognize(image) for image in images];obj.extractor.sizes.clear()
        self.assertEqual(obj.recognize_batch(images,batch_size=2),expected)
        self.assertEqual(expected[0][0],'Unknown');self.assertEqual(obj.extractor.sizes,[2,1,1])
        obj.min_confidence=35
        obj.model.predict_proba=lambda features:np.tile(
            [.49999999,.50000001] if len(features)>1 else [.50000001,.49999999],(len(features),1))
        expected=[obj.recognize(image) for image in images];obj.extractor.sizes.clear()
        self.assertEqual(obj.recognize_batch(images,batch_size=2),expected)
        self.assertEqual(expected[0][0],'Alpha_identity');self.assertEqual(obj.extractor.sizes,[2,1,1])

    def test_empty_closed_and_invalid_batch_sizes_do_not_run_models(self):
        obj=self.reider();self.assertEqual(obj.recognize_batch(iter(()),batch_size=2),[])
        self.assertEqual(obj.extractor.sizes,[])
        for invalid in (0,-1,True,1.5,'2'):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):obj.recognize_batch([],batch_size=invalid)
            with self.assertRaises(ValueError):self.pipeline(Scripted()).reidPeople(None,[],img_is_mat=True,batch_size=invalid)
        obj.close()
        with self.assertRaisesRegex(RuntimeError,'closed'):obj.recognize_batch([],batch_size=2)

    def test_pipeline_duplicate_cascades_aliases_invalid_crops_and_failures_match_scalar(self):
        for deduplicate in (False,True):
            for alias,invalid,fail in ((False,False,None),(True,False,None),(False,True,None),(False,False,2)):
                with self.subTest(deduplicate=deduplicate,alias=alias,invalid=invalid,fail=fail):
                    left=Scripted(fail);right=Scripted(fail)
                    frame,people=self.fixture(alias,invalid)
                    expected,counts=self.pipeline(left).reidPeople(frame,people,img_is_mat=True,deduplicate=deduplicate)
                    frame,people=self.fixture(alias,invalid);before=frame.copy()
                    actual,actual_counts=self.pipeline(right).reidPeople(frame,people,img_is_mat=True,deduplicate=deduplicate,batch_size=3)
                    self.assertIs(actual,people);self.assertEqual(actual_counts,counts)
                    self.assertEqual([(x.deepid,x.deepid_conf) for x in actual],[(x.deepid,x.deepid_conf) for x in expected])
                    self.assertTrue(all(size<=3 for size in right.batches))
                    if alias:self.assertFalse(right.batches)
                    np.testing.assert_array_equal(frame,before)

    def test_default_remains_scalar_and_facenet_ignores_torchreid_batch_option(self):
        recognizer=Scripted();frame,people=self.fixture()
        self.pipeline(recognizer).reidPeople(frame,people,img_is_mat=True)
        self.assertFalse(recognizer.batches)
        recognizer=SimpleNamespace(auto_load=True,recognize=Mock(return_value=('Alice',80.)),recognize_batch=Mock(side_effect=AssertionError('unexpected batch')))
        pipeline=self.pipeline(recognizer);pipeline.__ri_cfg__=SimpleNamespace(ri_name='FaceNet',yl_h_calibration=[-2,2],yl_w_calibration=[-2,2])
        person=Person(0,0,repspoint=(10,10))
        _,counts=pipeline.reidPeople(frame,[person],img_is_mat=True,batch_size=8)
        self.assertEqual(counts,(1,0));recognizer.recognize_batch.assert_not_called()

    def test_later_duplicate_group_includes_identities_changed_by_earlier_group(self):
        frame,people=self.fixture();recognizer=Scripted()
        result,counts=self.pipeline(recognizer).reidPeople(frame,people,img_is_mat=True,batch_size=3)
        self.assertEqual(counts,(3,6))
        self.assertEqual([p.deepid for p in result],['C','C','B','C'])
        self.assertEqual(recognizer.seen,{1:3,2:3,3:1,4:2})

    def test_inconsistent_batch_outputs_fall_back_without_dropping_other_people(self):
        frame,people=self.fixture();recognizer=Scripted()
        recognizer.recognize_batch=Mock(return_value=[('bad-length',10.)])
        result,counts=self.pipeline(recognizer).reidPeople(frame,people,img_is_mat=True,deduplicate=False,batch_size=3)
        self.assertEqual(counts,(3,0));self.assertEqual([p.deepid for p in result],['A','A','B','B'])
        self.assertEqual(recognizer.scalars,3)
        obj=self.reider();obj.model.predict_proba=lambda features:np.array([[.8,.2]])
        with self.assertRaisesRegex(ValueError,'probability batch'):
            obj.recognize_batch([np.zeros((4,4,3),dtype=np.uint8)]*2,batch_size=2)

    def test_pipeline_clips_crops_and_preserves_readonly_strided_frames(self):
        frame,unused=self.fixture();frame=frame[:,::-1];frame.flags.writeable=False
        def people():return [Person(0,0,box_xyxy=[-4,-3,12,40]),Person(1,1,box_xyxy=[16,0,32,32])]
        before=frame.copy()
        expected,counts=self.pipeline(Scripted()).reidPeople(frame,people(),img_is_mat=True)
        actual,batch_counts=self.pipeline(Scripted()).reidPeople(frame,people(),img_is_mat=True,batch_size=3)
        self.assertEqual(counts,batch_counts);self.assertEqual([p.deepid for p in expected],[p.deepid for p in actual])
        np.testing.assert_array_equal(frame,before)

    def test_deferred_loading_and_facade_keyword_keep_existing_call_behavior(self):
        from unittest.mock import patch
        import pyppbox.ppb as api
        reider=SimpleNamespace(auto_load=False,load_classifier=Mock(),recognize_batch=Mock())
        pipeline=self.pipeline(reider);pipeline.__ri_cfg__.validateRuntime=Mock()
        result,counts=pipeline.reidPeople(None,[],img_is_mat=True,batch_size=4)
        self.assertEqual((result,counts),([],(0,0)));reider.load_classifier.assert_called_once()
        reider.recognize_batch.assert_not_called();self.assertTrue(reider.auto_load)
        with patch.object(api,'_get_mt',return_value=pipeline),patch.object(pipeline,'reidPeople',return_value=([],(0,0))) as method:
            self.assertEqual(api.reidPeople(None,[],img_is_mat=True,batch_size=4),([],(0,0)))
            method.assert_called_once_with(None,[],deduplicate=True,img_is_mat=True,batch_size=4)


if __name__=='__main__':unittest.main()
