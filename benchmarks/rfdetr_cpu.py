# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Bounded real RF-DETR CPU contracts; generated heads test schemas, not quality."""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
import time
from unittest.mock import patch

from pipeline_cpu import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variants',nargs='+',choices=('nano','small','medium','large'),default=['nano','small','medium','large'])
    args=parser.parse_args()
    root=args.package_root.resolve(); sys.path.insert(0,str(root)); sys.dont_write_bytecode=True
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYPPBOX_DISABLE_FILE_LOG='1',PYPPBOX_DISABLE_TERMINAL_LOG='1',MPLBACKEND='Agg')
    assets=json.loads(args.assets.read_text(encoding='utf-8'))['assets']
    for name,entry in assets.items():
        if sha256(entry['path']) != entry['sha256']: raise AssertionError(f'Asset changed: {name}')
    os.environ['RF_HOME']=str(Path(assets['nano']['path']).parent)
    import cv2
    import numpy as np
    import torch
    import pyppbox
    from pyppbox.ppb.mt import MT
    from pyppbox.modules.trackers.sort.origin.sort import KalmanBoxTracker
    from pyppbox.utils.persontools import findRepspoint
    assert Path(pyppbox.__file__).resolve().is_relative_to(root),pyppbox.__file__
    torch.manual_seed(351); torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    capture=cv2.VideoCapture(assets['video']['path'])
    try:
        ok,frame=capture.read(); assert ok
    finally: capture.release()
    original=frame.copy(); blank=np.zeros_like(frame)
    def snapshot(people):
        return [dict(cid=int(p.cid),init_id=int(p.init_id),box=np.asarray(p.box_xyxy).tolist(),
            confidence=float(p.det_conf),point=list(map(int,p.repspoint)),keypoints=np.asarray(p.keypoints).tolist()) for p in people]
    def document(size,asset=None,optimize=True,person=1):
        return dict(dt_name='RF_DETR',model_size=size,model_file=assets[asset or size]['path'],
                    device='cpu',optimize=optimize,person_class_id=person,conf=.4,max_det=100)
    def check_native(detector,image,person):
        native=detector.model.predict(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),threshold=.4,include_source_image=False)
        expected=[]
        for box,score,label in zip(native.xyxy,native.confidence,native.class_id):
            clipped=np.clip(box,[0,0,0,0],[image.shape[1],image.shape[0],image.shape[1],image.shape[0]]).astype(int)
            if label==person and np.isfinite(box).all() and .4<=score<=1 and clipped[2]-clipped[0]>=15 and clipped[3]>clipped[1]:
                expected.append((clipped,float(score)))
        expected=sorted(expected,key=lambda row:-row[1])[:100]
        people,returned=detector.detectPeople(image,visual=False)
        assert returned is image and len(people)==len(expected)
        for index,(actual,(box,score)) in enumerate(zip(people,expected)):
            np.testing.assert_array_equal(actual.box_xyxy,box)
            assert abs(actual.det_conf-score)<=1e-6
            assert actual.init_id==index and not len(actual.keypoints)
            assert tuple(actual.repspoint)==tuple(findRepspoint(box,.25))
        return snapshot(people),native
    cases={}; failures={}
    # All model construction and inference happen with network access rejected.
    # HF offline mode additionally covers backend downloads implemented outside Python sockets.
    def network(*a,**k): raise AssertionError('Unexpected network access during offline validation')
    with patch.object(socket.socket,'connect',network),patch.object(socket.socket,'connect_ex',network), \
         patch.object(socket,'create_connection',network),tempfile.TemporaryDirectory(prefix='rfdetr-validation-') as temporary:
        with MT() as pipeline:
            pipeline.setMainReIDer('None')
            for size in args.variants:
                cases[size]={}
                for optimize in (False,True):
                    previous=pipeline.__dt__
                    pipeline.setMainDetector(document(size,optimize=optimize))
                    if previous is not None: assert previous.model is None
                    detector=pipeline.__dt__
                    started=time.perf_counter()
                    people,native=check_native(detector,frame,1)
                    elapsed=time.perf_counter()-started
                    assert people,f'{size}: published model detected no people'
                    empty,_=check_native(detector,blank,1)
                    labels=native.data['class_name'][native.class_id==1]
                    assert len(labels)>0 and all(name=='person' for name in labels)
                    painted=frame.copy(); detector.detectPeople(painted,visual=True)
                    cases[size][str(optimize)]=dict(people=people,blank=empty,
                        rendered_sha256=__import__('hashlib').sha256(painted.tobytes()).hexdigest(),
                        two_predictions_seconds=elapsed)
                # Explicit empty model_file opts into the upstream cache; it must work offline.
                cached=dict(document(size),model_file='')
                pipeline.setMainDetector(cached)
                cached_output,_=check_native(pipeline.__dt__,frame,1)
                assert cached_output==cases[size]['True']['people']
                pipeline.setMainTracker(dict(tk_name='SORT',max_age=1,min_hits=1,iou_threshold=.3))
                KalmanBoxTracker.count=0
                tracks=[]
                for image in (frame,frame,blank,frame,frame):
                    people,_=pipeline.detectPeople(image,img_is_mat=True,visual=False)
                    tracked=pipeline.trackPeople(image,people,img_is_mat=True)
                    tracked,counts=pipeline.reidPeople(image,tracked,img_is_mat=True)
                    assert counts==(0,0) or list(counts)==[0,0]
                    tracks.append(snapshot(tracked))
                assert [p['cid'] for p in tracks[0]]==[p['cid'] for p in tracks[1]]
                assert [p['cid'] for p in tracks[1]]==[p['cid'] for p in tracks[3]]
                cases[size]['tracks']=tracks
                print(f'{size}: {len(cached_output)} people; native/optimized/cache/SORT checks pass',flush=True)
            for asset in ('custom','lightning'):
                pipeline.setMainDetector(document('nano',asset=asset,person=0))
                people,native=check_native(pipeline.__dt__,frame,0)
                assert people
                assert all(name=='fixture_person' for name in native.data['class_name'][native.class_id==0])
                cases[asset]=dict(people=people,class_names=list(pipeline.__dt__.model.class_names))
                # A wrong-but-valid configured class must follow its numeric ID, never infer "person" by name.
                pipeline.setMainDetector(document('nano',asset=asset,person=1))
                other,_=check_native(pipeline.__dt__,frame,1)
                cases[asset]['class_one']=other
            pipeline.setMainDetector(document('nano',asset='custom',person=0))
            previous=pipeline.__dt__; config=pipeline.getConfig()
            before,_=check_native(previous,frame,0)
            corrupt=Path(temporary)/'corrupt.pth'; corrupt.write_bytes(b'not a torch checkpoint')
            for label,values in (
                    ('missing',dict(document('nano'),model_file=str(Path(temporary)/'missing.pth'))),
                    ('corrupt',dict(document('nano'),model_file=str(corrupt))),
                    ('unavailable_device',dict(document('nano'),device='cuda:99'))):
                try: pipeline.setMainDetector(values)
                except Exception as error:
                    failures[label]=type(error).__name__
                else: raise AssertionError(f'{label}: invalid detector initialization succeeded')
                assert pipeline.__dt__ is previous and pipeline.getConfig() is config
                after,_=check_native(previous,frame,0); assert after==before
            assert corrupt.read_bytes()==b'not a torch checkpoint'
            pipeline.reset(); assert pipeline.__dt__ is previous
        assert previous.model is None
        pipeline.close()
        with MT() as first, MT() as second:
            for stream in (first,second):
                stream.setMainDetector(document('nano'))
                stream.setMainTracker('SORT')
                stream.setMainReIDer('None')
            def track(stream):
                people,_=stream.detectPeople(frame,img_is_mat=True,visual=False)
                return snapshot(stream.trackPeople(frame,people,img_is_mat=True))
            initial=track(first); track(second)
            first_model=first.__dt__; second.reset(); second.close()
            assert first.__dt__ is first_model and first_model.model is not None
            remaining=track(first)
            assert [p['cid'] for p in initial]==[p['cid'] for p in remaining]
            cases['independent_streams']=dict(initial=initial,remaining=remaining)
    np.testing.assert_array_equal(frame,original)
    report=dict(schema=1,version=pyppbox.__version__,package_file=pyppbox.__file__,platform=platform.platform(),
        architecture=platform.machine(),python=sys.version,threads=2,
        packages={name:metadata.version(name) for name in ('rfdetr','torch','torchvision','numpy','transformers','supervision')},
        asset_sha256={key:entry['sha256'] for key,entry in assets.items()},
        frame_sha256=__import__('hashlib').sha256(frame.tobytes()).hexdigest(),
        offline=True,custom_fixture_quality_claim=False,cases=cases,failures=failures)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('RF-DETR CPU/offline/custom model validation passed',flush=True)


if __name__=='__main__': main()
