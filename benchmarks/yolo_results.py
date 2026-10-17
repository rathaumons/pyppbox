# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Benchmark YOLO wrapper processing with real Results and a fixed predictor.

Requires the installed Ultralytics backend and PyTorch, but no model weights::

    python benchmarks/yolo_results.py --output plans/validation/v4-yolo-results/before.json
    python benchmarks/yolo_results.py --output plans/validation/v4-yolo-results/after.json \
        --compare plans/validation/v4-yolo-results/before.json

Use --source for frozen baseline code and --device cuda:0 for synchronized GPU
result-to-CPU conversion measurements. Neural inference is excluded. Rendering
hashes, row fingerprints, model arguments, raw timings, and allocations are saved.
"""
import argparse
from datetime import datetime, timezone
from importlib import util
import json
import os
from pathlib import Path
import sys
import time
import tracemalloc
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'plans/validation/yolo-settings'))
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
from reid_crops import digest, environment, git
import cv2
import numpy as np
import torch
import ultralytics
from ultralytics.engine.results import Results
from pyppbox.modules.detectors.yoloult import MyYOLOULT


def load_detector(source):
    if source is None:
        return MyYOLOULT
    spec = util.spec_from_file_location('_baseline_yoloult', source)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyYOLOULT


class Predictor:
    def __init__(self, result, task):
        self.result, self.task, self.kwargs = result, task, None

    def predict(self, image, **kwargs):
        self.kwargs = kwargs
        return [self.result]


def snapshot(output, method):
    if method == 'detect':
        image, xywh, xyxy, points, keypoints, scores = output
        ids = None
    else:
        people, image = output
        xywh, xyxy = [p.box_xywh for p in people], [p.box_xyxy for p in people]
        points, scores = [p.repspoint for p in people], [p.det_conf for p in people]
        ids = [[p.init_id, p.cid] for p in people]
        keypoints = [p.keypoints for p in people if len(p.keypoints)]
    rows = dict(xywh=[box.tolist() for box in xywh], xyxy=[box.tolist() for box in xyxy],
                points=[[int(value) for value in point] for point in points], scores=scores, ids=ids,
                keypoints=[kp.tolist() for kp in keypoints])
    return dict(rows_sha256=digest(json.dumps(rows, sort_keys=True).encode()),
                drawing_sha256=digest(image.tobytes()), count=len(xyxy),
                keypoint_devices=[str(kp.device) if isinstance(kp, torch.Tensor) else 'numpy' for kp in keypoints])


def run_case(detector_class, device, method, pose, count, visual, alternative, samples, auxiliary=False):
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    rng = np.random.default_rng(351)
    boxes = np.zeros((count, 6), dtype=np.float32)
    if count:
        boxes[:, :2] = rng.integers(20, 280, (count, 2)) + .9
        boxes[:, 2:4] = boxes[:, :2] + [25.2, 60.3]
        boxes[::4, 2] = boxes[::4, 0] + 4.2  # Filtered rows at multiple positions.
        boxes[1::7, 2] = boxes[1::7, 0] + 14.9  # Integer conversion precedes width filtering.
        boxes[:, 4] = rng.uniform(.4, .99, count)  # Order is deliberately not score-sorted.
    keypoints = np.zeros((count, 17, 3), dtype=np.float32)
    if count:
        keypoints[:, :, 0] = boxes[:, 0, None] + np.arange(17)[None, :]
        keypoints[:, :, 1] = boxes[:, 1, None] + np.arange(17)[None, :]
        keypoints[:, :, 2] = .9
        keypoints[:, 0, 2] = .1
    result = Results(frame, path='fixed.png', names={0: 'person'},
                     boxes=torch.as_tensor(boxes, device=device),
                     keypoints=torch.as_tensor(keypoints, device=device) if pose else None,
                     masks=torch.ones((count, 64, 64), device=device) if auxiliary else None)
    predictor = Predictor(result, 'pose' if pose else 'detect')
    cfg = SimpleNamespace(imgsz=640, conf=.4, iou=.7, show_boxes=False, device=device,
                          max_det=100, repspoint_calibration=.25,
                          model_file='fixed-pose.pt' if pose else 'fixed.pt')
    with patch('ultralytics.YOLO', return_value=predictor):
        detector = detector_class(cfg)
    options = dict(visual=visual)
    if method == 'detectPeople':
        options.update(alt_repspoint=alternative != 'default', alt_repspoint_top=alternative == 'top')

    def sync():
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)

    def run():
        image = frame.copy()
        output = getattr(detector, method)(image, **options)
        assert (output[0] if method == 'detect' else output[1]) is image
        return output

    observed = snapshot(run(), method)
    for _ in range(5):
        run()
    sync()
    times = []
    for _ in range(samples):
        # Input allocation is outside timing, including visual=False cases.
        image = frame.copy()
        sync()
        started = time.perf_counter_ns()
        getattr(detector, method)(image, **options)
        sync()
        times.append((time.perf_counter_ns() - started) / 1e6)
    image = frame.copy()
    tracemalloc.start()
    try:
        getattr(detector, method)(image, **options)
        sync()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return dict(snapshot=observed, predict_kwargs=predictor.kwargs, milliseconds=times,
                p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
                traced_peak_bytes=peak, inference=False, samples=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--samples', type=int, default=45)
    args = parser.parse_args()
    if args.samples < 5:
        parser.error('--samples must be at least 5')
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    source = args.source or ROOT / 'pyppbox/modules/detectors/yoloult/__init__.py'
    detector = load_detector(args.source)
    report = dict(schema=1, scope='YOLO wrapper conversion/filtering/rendering; no inference',
                  captured_at_utc=datetime.now(timezone.utc).isoformat(), git_head=git('rev-parse', 'HEAD'),
                  source_sha256=digest(source.read_bytes().replace(b'\r\n', b'\n')),
                  benchmark_sha256=digest(Path(__file__).read_bytes()), environment=environment(),
                  backend_version=ultralytics.__version__, backend_file=ultralytics.__file__,
                  device=args.device, device_name=torch.cuda.get_device_name(args.device)
                  if args.device.startswith('cuda') else 'CPU',
                  config_sha256={path.name: digest(path.read_bytes()) for path in sorted(
                      (ROOT / 'pyppbox/config/cfg').glob('*.yaml'))}, cases={})
    cases = [(method, pose, count, visual, 'default', False)
             for method in ('detect', 'detectPeople') for pose in (False, True)
             for count in (0, 20, 100) for visual in (False, True)]
    cases += [('detectPeople', True, 20, visual, alternative, False)
              for alternative in ('top', 'bottom') for visual in (False, True)]
    cases += [('detectPeople', False, 20, False, 'default', True)]
    for method, pose, count, visual, alternative, auxiliary in cases:
        name = f'{method}/pose-{pose}/rows-{count}/visual-{visual}/{alternative}/aux-{auxiliary}'
        case = run_case(detector, args.device, method, pose, count, visual, alternative, args.samples, auxiliary)
        report['cases'][name] = case
        print(name, f"{case['p50_ms']:.3f} ms", flush=True)
    if args.compare:
        previous = json.loads(args.compare.read_text(encoding='utf-8'))
        for key in ('config_sha256', 'backend_version', 'device', 'device_name'):
            assert previous[key] == report[key], key
        for key in ('python', 'cpu', 'packages', 'opencv_threads', 'thread_environment'):
            assert previous['environment'][key] == report['environment'][key], key
        assert set(previous['cases']) == set(report['cases'])
        report['comparison'] = {}
        for name, current in report['cases'].items():
            before = previous['cases'][name]
            assert before['snapshot'] == current['snapshot'], (name, 'Outputs or drawing changed')
            assert before['predict_kwargs'] == current['predict_kwargs'], (name, 'Predict arguments changed')
            report['comparison'][name] = dict(before_ms=before['p50_ms'], after_ms=current['p50_ms'],
                before_peak_bytes=before['traced_peak_bytes'], after_peak_bytes=current['traced_peak_bytes'])
        print('All row/drawing fingerprints, CPU keypoint types, IDs, and predictor arguments match.', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
