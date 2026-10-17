# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Measure MT crop/dispatch overhead with deterministic recognizers, without models.

Run from the checkout with its normal test dependencies installed::

    python benchmarks/reid_crops.py --output plans/validation/v4-crops/before.json
    python benchmarks/reid_crops.py --output plans/validation/v4-crops/after.json \
        --compare plans/validation/v4-crops/before.json

These are ReID preparation measurements, not neural-inference or pipeline FPS.
Raw samples, environment, source hashes, input hashes, and crop/output fingerprints
are retained. Allocation probes run separately from timing to avoid tracer overhead.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tracemalloc
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('PYPPBOX_DISABLE_FILE_LOG', '1')
os.environ.setdefault('PYPPBOX_DISABLE_TERMINAL_LOG', '1')
import_started = time.perf_counter()
import cv2
import numpy as np
from pyppbox.ppb.mt import MT
from pyppbox.utils.persontools import Person
IMPORT_SECONDS = time.perf_counter() - import_started


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def load_pipeline(source):
    if source is None:
        return MT
    spec = util.spec_from_file_location('_benchmark_mt', source)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MT


class Recognizer:
    auto_load = True

    def __init__(self):
        self.record = False
        self.inputs = []

    def recognize(self, image, is_bgr=True):
        if self.record:
            self.inputs.append(dict(shape=list(image.shape), dtype=str(image.dtype),
                                    sha256=digest(image.tobytes()), is_bgr=is_bgr))
        # A shared identity ensures every crowded duplicate-pass crop is exercised.
        return 'Benchmark identity', 87.5


def people_for(count, width, height, known=False):
    people = []
    for index in range(count):
        x = 70 + index % 10 * ((width - 240) // 10)
        y = 80 + index // 10 * ((height - 380) // 2)
        person = Person(index, index, box_xyxy=np.array([x, y, x + 100, y + 250]),
                        repspoint=(x + 50, y + 65))
        if known:
            person.faceid = person.deepid = 'Known identity'
        people.append(person)
    return people


def run_case(pipeline_class, frame, backend, count, deduplicate, known, samples):
    pipeline = pipeline_class()
    pipeline.__ri_is_set__ = True
    pipeline.__ri_cfg__ = SimpleNamespace(ri_name=backend, model_wh=(128, 256),
                                         yl_h_calibration=[-50, 50], yl_w_calibration=[-30, 30])
    recognizer = Recognizer()
    pipeline.__ri__ = recognizer
    height, width = frame.shape[:2]

    def prepare():
        return people_for(count, width, height, known)

    def run(people):
        return pipeline.reidPeople(frame, people, img_is_mat=True, deduplicate=deduplicate)

    before_hash = digest(frame.tobytes())
    recognizer.record = True
    people, counts = run(prepare())
    fingerprint = dict(crops=recognizer.inputs, counts=list(counts), people=[
        [p.cid, p.faceid, p.deepid, p.faceid_conf, p.deepid_conf] for p in people])
    recognizer.record = False
    for _ in range(5):
        run(prepare())
    timings = []
    for _ in range(samples):
        people = prepare()
        started = time.perf_counter_ns()
        run(people)
        timings.append((time.perf_counter_ns() - started) / 1e6)
    peaks = []
    for _ in range(3):
        people = prepare()
        tracemalloc.start()
        try:
            run(people)
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
    assert before_hash == digest(frame.tobytes()), 'Input frame was modified'
    return dict(backend=backend, frame_shape=list(frame.shape), people=count,
                deduplicate=deduplicate, known=known, counts=list(counts),
                fingerprint=digest(json.dumps(fingerprint, sort_keys=True).encode()),
                input_sha256=before_hash, milliseconds=timings,
                p50_ms=float(np.percentile(timings, 50)),
                p95_ms=float(np.percentile(timings, 95)),
                p99_ms=float(np.percentile(timings, 99)),
                traced_peak_bytes=max(peaks), warmup=5)


def environment():
    packages = {}
    for name in ('numpy', 'opencv-contrib-python', 'scipy', 'scikit-learn', 'PyYAML',
                 'torch', 'torchvision', 'tensorflow', 'pyppbox-torchreid',
                 'pyppbox-ultralytics', 'vsensebox-ultralytics', 'rfdetr'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    details = dict(python=sys.version, executable=sys.executable, os=platform.platform(),
                cpu=platform.processor(), logical_cpus=os.cpu_count(),
                packages=packages, opencv_threads=cv2.getNumThreads(), device='CPU',
                thread_environment={key: os.environ.get(key) for key in (
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')},
                process_import_seconds=IMPORT_SECONDS)
    try:
        import psutil
    except ImportError:
        details.update(physical_cpus=None, process_rss_bytes=None)
    else:
        details.update(physical_cpus=psutil.cpu_count(logical=False),
                       process_rss_bytes=psutil.Process().memory_info().rss)
    return details


def compare(before, after):
    for key in ('python', 'cpu', 'packages', 'opencv_threads', 'device', 'thread_environment'):
        if before['environment'][key] != after['environment'][key]:
            raise ValueError(f'Benchmark environment differs: {key}')
    if before['config_sha256'] != after['config_sha256']:
        raise ValueError('Bundled configuration changed')
    if set(before['cases']) != set(after['cases']):
        raise ValueError('Benchmark workloads differ')
    rows = {}
    for name, result in after['cases'].items():
        previous = before['cases'][name]
        for key in ('fingerprint', 'input_sha256', 'counts'):
            if previous[key] != result[key]:
                raise ValueError(f'{name}: {key} changed')
        rows[name] = dict(p50_before_ms=previous['p50_ms'], p50_after_ms=result['p50_ms'],
                          p50_change_percent=100 * (result['p50_ms'] / previous['p50_ms'] - 1),
                          peak_before_bytes=previous['traced_peak_bytes'],
                          peak_after_bytes=result['traced_peak_bytes'])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--mt-source', type=Path, help='Optional frozen MT source for paired baseline reruns.')
    parser.add_argument('--samples', type=int, default=90)
    args = parser.parse_args()
    if args.samples < 10:
        parser.error('--samples must be at least 10')
    cv2.setNumThreads(1)
    source = args.mt_source or ROOT / 'pyppbox/ppb/mt.py'
    pipeline_class = load_pipeline(args.mt_source)
    started = time.perf_counter()
    pipeline_class()
    construction_seconds = time.perf_counter() - started
    report = dict(schema=1, scope='ReID crop/dispatch with deterministic recognizers; no inference',
                  captured_at_utc=datetime.now(timezone.utc).isoformat(),
                  git_head=git('rev-parse', 'HEAD'), git_status=git('status', '--porcelain'),
                  mt_source_sha256=digest(source.read_bytes()),
                  mt_code_sha256=digest(source.read_bytes().replace(b'\r\n', b'\n')),
                  benchmark_sha256=digest(Path(__file__).read_bytes()), environment=environment(),
                  mt_first_construction_seconds=construction_seconds, seed=20261016,
                  config_sha256={path.name: digest(path.read_bytes())
                                 for path in sorted((ROOT / 'pyppbox/config/cfg').glob('*.yaml'))},
                  cases={})
    for height, width in ((720, 1280), (1080, 1920)):
        frame = np.random.default_rng(report['seed']).integers(0, 256, (height, width, 3), dtype=np.uint8)
        for backend in ('Torchreid', 'FaceNet'):
            for label, count, duplicate, known in (
                    ('empty', 0, True, False), ('sparse', 1, False, False),
                    ('crowded', 20, False, False), ('duplicates', 20, True, False),
                    ('known', 20, False, True)):
                name = f'{height}p/{backend}/{label}'
                result = run_case(pipeline_class, frame, backend, count, duplicate, known, args.samples)
                report['cases'][name] = result
                print(f"{name}: {result['p50_ms']:.3f} ms median; "
                      f"{result['traced_peak_bytes']} traced peak bytes", flush=True)
    if args.compare:
        report['comparison'] = compare(json.loads(args.compare.read_text(encoding='utf-8')), report)
        print('All crop/output fingerprints, input frames, and ReID counts match.', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
