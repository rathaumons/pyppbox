# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Compare warmed, bounded Torchreid inference with scalar identity/count output.

Supply installed model/classifier paths and an image. --boxes-json optionally
contains a list of xyxy boxes. Without it, a regular grid supplies image crops.
Use --package-root with a frozen package and --batch-sizes 1 for the baseline.
This measures the ReID stage, not full detector/tracker throughput or a soak.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package-root', type=Path, default=ROOT)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--classifier', type=Path, required=True)
parser.add_argument('--image', type=Path, required=True)
parser.add_argument('--boxes-json', type=Path)
parser.add_argument('--device', default='cpu')
parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 16])
parser.add_argument('--counts', type=int, nargs='+', default=[1, 8, 32])
parser.add_argument('--samples', type=int, default=3)
parser.add_argument('--reference', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if any(size < 1 for size in args.batch_sizes) or args.samples < 1 or any(n < 0 for n in args.counts):
    parser.error('Batch sizes/samples must be positive; people counts must be nonnegative.')
sys.path.insert(0, str(args.package_root.resolve()))
os.environ.setdefault('PYPPBOX_DISABLE_FILE_LOG', '1')
os.environ.setdefault('PYPPBOX_DISABLE_TERMINAL_LOG', '1')
import cv2
import numpy as np
import torch
import pyppbox
from pyppbox.config.myconfig import RCFGTorchreid
from pyppbox.config.configtools import internal_cfg_dir, loadDocumentList
from pyppbox.modules.reiders.torchreid import MyTorchreid
from pyppbox.ppb.mt import MT
from pyppbox.utils.persontools import Person

assert Path(pyppbox.__file__).resolve().is_relative_to(args.package_root.resolve())
torch.manual_seed(351)
torch.set_num_threads(1)
cv2.setNumThreads(1)
frame = cv2.imread(str(args.image))
if frame is None:
    parser.error('Cannot read the input image.')
frame_hash = hashlib.sha256(frame.tobytes()).hexdigest()
if args.boxes_json:
    boxes = json.loads(args.boxes_json.read_text(encoding='utf-8'))
else:
    height, width = frame.shape[:2]
    boxes = [[i * width // 5, 0, (i + 1) * width // 5, height] for i in range(5)]
if not boxes:
    parser.error('At least one crop box is required.')
document = next(doc for doc in loadDocumentList(Path(internal_cfg_dir) / 'reiders.yaml')
                if doc['ri_name'] == 'Torchreid')
cfg = RCFGTorchreid()
cfg.set(dict(document, model_path=str(args.model.resolve()),
             classifier_pkl=str(args.classifier.resolve()), device=args.device))
reider = MyTorchreid(cfg, auto_load=True)


class Probe:
    def __init__(self, wrapped): self.wrapped, self.sizes = wrapped, []
    def __call__(self, images):
        self.sizes.append(len(images) if isinstance(images, list) else 1)
        return self.wrapped(images)


probe = Probe(reider.extractor)
reider.extractor = probe
pipeline = MT()
pipeline.__ri__, pipeline.__ri_cfg__, pipeline.__ri_is_set__ = reider, cfg, True
supports_batch = hasattr(reider, 'recognize_batch')
reference = json.loads(args.reference.read_text()) if args.reference else None
records = {}


def sync():
    if args.device.startswith('cuda'): torch.cuda.synchronize()


def people(count, mixed=False):
    result = []
    for i in range(count):
        box = list(boxes[i % len(boxes)])
        identity = 'Known_identity' if mixed and i == 0 else 'Unknown'
        if mixed and i == 1: box = [frame.shape[1] + 1, 0, frame.shape[1] + 20, 40]
        entry = Person(i, i, box_xyxy=box, repspoint=(10, 20), deepid=identity,
                       faceid='Existing_face', faceid_conf=73.)
        entry.misc = {'index': i}
        result.append(entry)
    return result


def run(count, deduplicate, size, mixed=False):
    entries = people(count, mixed)
    kwargs = dict(img_is_mat=True, deduplicate=deduplicate)
    if supports_batch: kwargs['batch_size'] = size
    elif size != 1: raise ValueError('Frozen package does not support batching.')
    sync()
    start = time.perf_counter()
    returned, counts = pipeline.reidPeople(frame, entries, **kwargs)
    sync()
    elapsed = (time.perf_counter() - start) * 1000
    assert returned is entries or count == 0
    snapshot = dict(counts=list(counts), rows=[dict(init_id=p.init_id, cid=p.cid,
        deepid=p.deepid, confidence=p.deepid_conf, faceid=p.faceid, face_conf=p.faceid_conf,
        box=list(p.box_xyxy), misc=p.misc) for p in returned])
    return elapsed, snapshot


def equivalent(actual, expected):
    assert actual['counts'] == expected['counts'], (actual['counts'], expected['counts'])
    assert len(actual['rows']) == len(expected['rows'])
    delta = 0.
    for row, old in zip(actual['rows'], expected['rows']):
        delta = max(delta, abs(row['confidence'] - old['confidence']))
        assert {k:v for k,v in row.items() if k != 'confidence'} == {k:v for k,v in old.items() if k != 'confidence'}
    assert delta <= .1, delta
    return delta


try:
    for count in args.counts:
        for duplicate in (False, True):
            key = f'n{count}-dup{int(duplicate)}'
            _, baseline = run(count, duplicate, 1)
            if reference: equivalent(baseline, reference['workloads'][key]['snapshot'])
            record = dict(snapshot=baseline, batches={})
            for size in args.batch_sizes:
                run(count, duplicate, size)  # Separate shape-specific warmup from timing.
                sync()
                if args.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
                probe.sizes.clear()
                samples, max_delta = [], 0.
                for _ in range(args.samples):
                    elapsed, output = run(count, duplicate, size)
                    max_delta = max(max_delta, equivalent(output, baseline))
                    samples.append(elapsed)
                assert not probe.sizes or max(probe.sizes) <= size
                record['batches'][str(size)] = dict(samples_ms=samples, p50_ms=statistics.median(samples),
                    max_confidence_delta=max_delta, extractor_calls=len(probe.sizes), max_extractor_batch=max(probe.sizes, default=0),
                    peak_cuda_allocated=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None)
            records[key] = record
            print(f'Completed {key}', flush=True)
    mixed = run(8, True, 1, mixed=True)[1]
    if reference: equivalent(mixed, reference['mixed'])
    for size in args.batch_sizes: equivalent(run(8, True, size, mixed=True)[1], mixed)
    guard_checks = 0
    if supports_batch:
        crops = [cv2.resize(frame[y1:y2,x1:x2],cfg.model_wh) for x1,y1,x2,y2 in boxes[:3]]
        original = reider.min_confidence
        for image in crops:
            identity, confidence = reider.recognize(image)
            reider.min_confidence = confidence
            expected = [reider.recognize(image)] * 2
            actual = reider.recognize_batch([image,image],batch_size=2)
            assert actual == expected
            guard_checks += 1
        reider.min_confidence = original
    assert hashlib.sha256(frame.tobytes()).hexdigest() == frame_hash
    report = dict(version=pyppbox.__version__, device=args.device, python=sys.version, torch=torch.__version__,
        package=str(pyppbox.__file__), platform=platform.platform(), cuda=torch.version.cuda,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        classifier_sha256=hashlib.sha256(args.classifier.read_bytes()).hexdigest(), frame_sha256=frame_hash,
        boxes=boxes, workloads=records, mixed=mixed, scalar_boundary_checks=guard_checks)
    if reference:
        for field in ('version', 'device', 'model_sha256', 'classifier_sha256', 'frame_sha256', 'boxes'):
            assert report[field] == reference[field], field
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:{size:round(row['p50_ms'],3) for size,row in value['batches'].items()} for key,value in records.items()}))
finally:
    pipeline.close()
