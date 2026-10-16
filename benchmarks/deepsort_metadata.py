# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Measure DeepSORT bookkeeping separately from encoding and neural inference.

    python benchmarks/deepsort_metadata.py --output plans/validation/v4-deepsort/before.json
    python benchmarks/deepsort_metadata.py --output plans/validation/v4-deepsort/after.json \
        --compare plans/validation/v4-deepsort/before.json

Use --source with a frozen DeepSORT package directory to replay old code. Fixed
associations isolate wrapper overhead; real Kalman/matching replays measure the
tracker with deterministic features. Bug fixtures compare actual associations,
not the legacy wrapper's nearest-box guess. No model weights are required.
"""
import argparse
from datetime import datetime, timezone
import importlib
from importlib import util
import json
from pathlib import Path
import sys
import time
import tracemalloc
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from reid_crops import digest, environment, git
import cv2
import numpy as np
from pyppbox.utils.persontools import Person


def load_wrapper(source=None):
    name = '_baseline_deepsort' if source else 'pyppbox.modules.trackers.deepsort'
    stub = SimpleNamespace(create_box_encoder=Mock())
    with patch.dict(sys.modules, {name + '.origin.generate_detections': stub}):
        if source is None:
            return importlib.import_module(name)
        spec = util.spec_from_file_location(name, source / '__init__.py',
                                           submodule_search_locations=[str(source)])
        module = util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


def make_tracker(module, encoder, nms=1.):
    cfg = SimpleNamespace(nms_max_overlap=nms, model_file='deterministic-encoder.pb',
                          max_cosine_distance=.2, nn_budget=100)
    with patch.object(module.gdet, 'create_box_encoder', return_value=encoder):
        tracker = module.MyDeepSORT(cfg)
    tracker.tracker.n_init, tracker.tracker.max_age = 2, 2
    return tracker


def person(identity, x, confidence=.8, width=20):
    box = np.array([x, 10, x + width, 60])
    p = Person(identity, -1, box_xyxy=box, box_xywh=np.array([x, 10, width, 50]),
               det_conf=confidence, repspoint=(x + width // 2, 20),
               keypoints=np.full((17, 3), identity, dtype=np.float32))
    p.fixture_identity = identity
    return p


def rows(people):
    return [[p.init_id, p.cid, p.box_xyxy.tolist(), p.repspoint, p.det_conf,
             p.faceid, p.deepid, p.faceid_conf, p.deepid_conf, p.misc,
             np.asarray(p.keypoints).tolist()] for p in people]


def state(tracker):
    return [[t.track_id, t.hits, t.age, t.time_since_update, t.state,
             t.mean.tolist(), t.covariance.tolist()] for t in tracker.tracks]


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True).encode())


class FixedAssociations:
    """Fixed one-to-one assignments, bypassing only Kalman/metric work."""
    def __init__(self):
        self.tracks, self.detection_indices = [], {}

    def predict(self):
        pass

    def update(self, detections):
        self.tracks, self.detection_indices = [], {}
        for index, detection in enumerate(detections):
            cid = int(round((detection.tlwh[0] - 10) / 100)) + 1
            box = detection.to_tlbr()
            self.tracks.append(SimpleNamespace(track_id=cid, time_since_update=0,
                is_confirmed=lambda: True, to_tlbr=lambda box=box: box))
            self.detection_indices[cid] = index


def measure(prepare, run, snapshot, samples):
    observed = snapshot(run(prepare()))
    for _ in range(2):
        run(prepare())
    timings = []
    for _ in range(samples):
        prepared = prepare()
        started = time.perf_counter_ns()
        run(prepared)
        timings.append((time.perf_counter_ns() - started) / 1e6)
    prepared = prepare()
    tracemalloc.start()
    try:
        run(prepared)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return dict(fingerprint=fingerprint(observed), milliseconds=timings,
                p50_ms=float(np.median(timings)), p95_ms=float(np.percentile(timings, 95)),
                p99_ms=float(np.percentile(timings, 99)), traced_peak_bytes=peak)


def wrapper_case(module, count, samples):
    tracker = make_tracker(module, lambda image, boxes: np.tile([1., 0.], (len(boxes), 1)))
    tracker.tracker = FixedAssociations()
    for identity in range(count):
        p = person(identity, 10 + identity * 100)
        p.cid, p.deepid, p.faceid, p.misc = identity + 1, f'D{identity}', f'F{identity}', [identity]
        tracker._people_by_cid[p.cid] = p

    def prepare():
        return [person(i, 10 + i * 100, .5 + .4 * (i + 1) / (count + 1)) for i in range(count)]

    return measure(prepare, lambda people: tracker.update(people, img=None), rows, samples)


def real_case(module, count, samples):
    frames = [[person(i, 10 + i * 100, .5 + .4 * (i + 1) / (count + 1)) for i in range(count)]
              if frame not in (5, 6) else [] for frame in range(10)]

    def prepare():
        tracker = make_tracker(module, lambda image, boxes: np.tile([1., 0.], (len(boxes), 1)))
        inputs = [[person(p.init_id, int(p.box_xyxy[0]), p.det_conf) for p in frame] for frame in frames]
        return tracker, inputs

    def run(prepared):
        tracker, inputs = prepared
        result = []
        for people in inputs:
            result.append(tracker.update(people, img=None))
        return tracker, result

    def snapshot(value):
        tracker, result = value
        return dict(frames=[rows(people) for people in result], state=state(tracker.tracker))

    return measure(prepare, run, snapshot, samples)


def fixtures(module):
    scenarios = {
        'identical_boxes': (1., [[(0, 10, .9), (1, 10, .8)]] * 4),
        'nms_winner': (.5, [[(0, 10, .7), (1, 10, .9)]] * 4),
        'unmatched_prediction': (1., [[(0, 10, .9), (1, 70, .8)]] * 3
                                 + [[(0, 10, .9)], [(1, 70, .8), (0, 10, .9)]]
                                 + [[], [], [], [(0, 10, .9)]]),
        'crossing': (1., [[(0, 10 + 10 * i, .9), (1, 90 - 10 * i, .8)][::(-1 if i % 2 else 1)]
                          for i in range(9)]),
    }
    report = {}
    for name, (nms, frames) in scenarios.items():
        encoder = Mock()
        tracker = make_tracker(module, encoder, nms)
        last = {}
        original_match = tracker.tracker._match

        def match(detections):
            result = original_match(detections)
            last['matches'] = [(tracker.tracker.tracks[t].track_id, d) for t, d in result[0]]
            return result

        tracker.tracker._match = match
        original_nms = module.preprocessing.non_max_suppression

        def suppress(*args, **kwargs):
            last['indices'] = original_nms(*args, **kwargs)
            return last['indices']

        records, violations = [], 0
        with patch.object(module.preprocessing, 'non_max_suppression', side_effect=suppress):
            for frame in frames:
                people = [person(i, x, confidence, width=40) for i, x, confidence in frame]
                encoder.return_value = np.eye(2)[[p.init_id for p in people]]
                last['indices'] = []
                result = tracker.update(people, img=None)
                expected = [-1] * len(people)
                confirmed = {t.track_id for t in tracker.tracker.tracks if t.is_confirmed()}
                for cid, detection_index in last['matches']:
                    if cid in confirmed:
                        expected[last['indices'][detection_index]] = cid
                actual = [p.cid for p in result]
                violations += sum(a != b for a, b in zip(actual, expected))
                records.append(dict(actual_cids=actual, expected_cids=expected, rows=rows(result),
                                    nms_indices=[int(i) for i in last['indices']],
                                    matches=last['matches'], core_state=state(tracker.tracker)))
        report[name] = dict(association_violations=violations, frames=records)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--samples', type=int, default=9)
    args = parser.parse_args()
    if args.samples < 3:
        parser.error('--samples must be at least 3')
    cv2.setNumThreads(1)
    module = load_wrapper(args.source)
    source = args.source or ROOT / 'pyppbox/modules/trackers/deepsort'
    report = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(), git_head=git('rev-parse', 'HEAD'),
                  environment=environment(), source_sha256={str(path.relative_to(source)): digest(
                      path.read_bytes().replace(b'\r\n', b'\n')) for path in sorted(source.rglob('*.py'))},
                  config_sha256={path.name: digest(path.read_bytes()) for path in sorted(
                      (ROOT / 'pyppbox/config/cfg').glob('*.yaml'))},
                  benchmark_sha256=digest(Path(__file__).read_bytes()), cases={}, fixtures=fixtures(module))
    for count in (0, 10, 100, 500):
        name = f'fixed-associations/{count}'
        report['cases'][name] = wrapper_case(module, count, args.samples)
        print(name, f"{report['cases'][name]['p50_ms']:.3f} ms", flush=True)
    for count in (0, 1, 10, 40):
        name = f'real-tracker-10-frames/{count}'
        report['cases'][name] = real_case(module, count, args.samples)
        print(name, f"{report['cases'][name]['p50_ms']:.3f} ms", flush=True)
    if args.compare:
        before = json.loads(args.compare.read_text(encoding='utf-8'))
        for key in ('config_sha256', 'benchmark_sha256'):
            assert before[key] == report[key], key
        for key in ('python', 'cpu', 'packages', 'opencv_threads', 'thread_environment'):
            assert before['environment'][key] == report['environment'][key], key
        assert set(before['cases']) == set(report['cases'])
        for name, case in report['cases'].items():
            assert case['fingerprint'] == before['cases'][name]['fingerprint'], name
        for name, fixture in report['fixtures'].items():
            assert fixture['association_violations'] == 0, name
            for old, new in zip(before['fixtures'][name]['frames'], fixture['frames']):
                for key in ('expected_cids', 'nms_indices', 'matches', 'core_state'):
                    assert fingerprint(old[key]) == fingerprint(new[key]), (name, key)
        report['comparison'] = 'Equivalent stable outputs/core states; corrected ambiguous detection associations.'
        print(report['comparison'], flush=True)
    print('Fixture violations:', {name: case['association_violations']
                                  for name, case in report['fixtures'].items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
