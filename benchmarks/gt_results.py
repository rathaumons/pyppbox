# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Measure sparse GT lookup and result-list sorting without neural models.

Run from the checkout, using --gt-source/--results-source for frozen baselines::

    python benchmarks/gt_results.py --output plans/validation/v4-gt-results/before.json
    python benchmarks/gt_results.py --output plans/validation/v4-gt-results/after.json \
        --compare plans/validation/v4-gt-results/before.json

Reports retain raw timing/allocation samples and output hashes. Setup/recording,
lookup batches, result sorting, and actual output-format checks are separate.
"""
import argparse
from datetime import datetime, timezone
from importlib import util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from reid_crops import digest, environment, git
import numpy as np
from pyppbox.utils.gttools import GTInterpreter
from pyppbox.utils.restools import ResIO
from pyppbox.utils.persontools import Person


def load_class(source, name, default):
    if source is None:
        return default
    # Relative imports in frozen utility modules still resolve to pyppbox.utils.
    spec = util.spec_from_file_location('pyppbox.utils._benchmark_' + name, source)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def measure(function, samples):
    function()
    timings = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        function()
        timings.append((time.perf_counter_ns() - started) / 1e6)
    tracemalloc.start()
    try:
        function()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return dict(milliseconds=timings, p50_ms=float(np.median(timings)),
                p95_ms=float(np.percentile(timings, 95)), traced_peak_bytes=peak)


def check_formats(result_class):
    outputs = {}
    with tempfile.TemporaryDirectory() as folder:
        recorder = result_class()
        for frame in (5, 5, 20):
            for index, x in enumerate((30, 10, 10, 70)):
                person = Person(index, index, box_xywh=[x, 0, 10, 20], box_xyxy=[x, 0, x + 10, 20],
                                repspoint=(x, 4), faceid=f'Face {index}', deepid=f'Body {index}')
                person.misc = [f'entry {index}', ['nested', index]]
                recorder.addPerson(frame, person)
        for mode in (1, 2, 3):
            for identity in ('deepid', 'faceid', 'all'):
                for misc in (False, True):
                    if identity == 'all':
                        recorder.dumpAll(folder, mode, include_misc=misc, max_misc=1)
                    else:
                        recorder.dump(folder, mode, id_mode=identity, include_misc=misc, max_misc=1)
                    files = list(Path(folder).glob('res_*_full.txt'))
                    path = max(files, key=lambda item: item.stat().st_mtime_ns)
                    outputs[f'{mode}/{identity}/misc-{misc}'] = digest(path.read_bytes())
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--gt-source', type=Path)
    parser.add_argument('--results-source', type=Path)
    parser.add_argument('--samples', type=int, default=9)
    args = parser.parse_args()
    if args.samples < 3:
        parser.error('--samples must be at least 3')
    gt_class = load_class(args.gt_source, 'GTInterpreter', GTInterpreter)
    result_class = load_class(args.results_source, 'ResIO', ResIO)
    sources = {'gttools': args.gt_source or ROOT / 'pyppbox/utils/gttools.py',
               'restools': args.results_source or ROOT / 'pyppbox/utils/restools.py'}
    report = dict(schema=1, captured_at_utc=datetime.now(timezone.utc).isoformat(),
                  scope='GT lookup batches and stored-result sorting; no inference',
                  git_head=git('rev-parse', 'HEAD'), environment=environment(), samples=args.samples,
                  source_sha256={name: digest(path.read_bytes().replace(b'\r\n', b'\n')) for name, path in sources.items()},
                  benchmark_sha256=digest(Path(__file__).read_bytes()),
                  config_sha256={path.name: digest(path.read_bytes()) for path in sorted(
                      (ROOT / 'pyppbox/config/cfg').glob('*.yaml'))}, gt={}, results={})
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'gt.txt'
        for count in (1000, 10000, 50000):
            path.write_text(''.join(f'{10 + 3 * i}\t(5, 5)\tP{i}\t[0 0 10 20]\t[0 0 10 20]\n'
                                    for i in range(count)), encoding='utf-8')
            gt = gt_class()
            # Initial load is captured separately; lookup timing does not parse input.
            started = time.perf_counter()
            gt.setGT(path)
            initial_load_ms = 1000 * (time.perf_counter() - started)
            load = measure(lambda: gt.setGT(path), args.samples)
            queries = np.random.default_rng(20261017).integers(0, count, size=256)
            for label, requested in (
                    ('hits', [10 + 3 * int(i) for i in queries]),
                    ('misses', [11 + 3 * int(i) for i in queries])):
                rows = [gt.findGTFrame(frame) for frame in requested]
                expected = [gt.gt_frames[(frame - 10) // 3] if label == 'hits' else [] for frame in requested]
                assert rows == expected
                timing = measure(lambda: [gt.findGTFrame(frame) for frame in requested], args.samples)
                timing.update(queries=len(requested), frames=count,
                              fingerprint=digest(json.dumps(rows).encode()),
                              input_sha256=digest(path.read_bytes()), initial_load_ms=initial_load_ms,
                              reload=load)
                report['gt'][f'{count}/{label}'] = timing
                print('GT', count, label, f"{timing['p50_ms']:.3f} ms / 256 lookups", flush=True)
    for count, per_frame in ((1, 4), (100, 4), (1000, 4), (5000, 4), (10000, 4),
                              (1, 20), (100, 20), (1000, 20), (1, 1000)):
        coordinates = np.random.default_rng(71).integers(0, 50, size=per_frame)
        fixture = [Person(i, i, repspoint=(int(x), 5)) for i, x in enumerate(coordinates)]
        expected_order = np.argsort(coordinates).tolist()
        recorder = result_class()
        recorder.frames = [str(10 + 3 * frame) for frame in range(count) for _ in fixture]
        recorder.people = fixture * count
        timing = measure(recorder.__sort_people_by_x__, args.samples)
        ids = [person.cid for person in recorder.sorted_people]
        assert ids == expected_order * count and len(ids) == len(recorder.frames)
        timing.update(frames=count, people=len(ids), fingerprint=digest(json.dumps(ids).encode()))
        timing['people_per_frame'] = per_frame
        report['results'][f'{count}/{per_frame}'] = timing
        print('Results', count, 'frames x', per_frame, f"people: {timing['p50_ms']:.3f} ms; "
              f"{timing['traced_peak_bytes']} traced bytes", flush=True)
    report['output_sha256'] = check_formats(result_class)
    if args.compare:
        before = json.loads(args.compare.read_text(encoding='utf-8'))
        for key in ('python', 'cpu', 'packages', 'opencv_threads', 'thread_environment'):
            assert before['environment'][key] == report['environment'][key], key
        assert before['config_sha256'] == report['config_sha256']
        assert before['output_sha256'] == report['output_sha256'], 'Dump output bytes changed'
        comparison = {}
        for category in ('gt', 'results'):
            assert set(before[category]) == set(report[category])
            comparison[category] = {}
            for name, current in report[category].items():
                previous = before[category][name]
                assert previous['fingerprint'] == current['fingerprint'], (category, name)
                comparison[category][name] = dict(before_ms=previous['p50_ms'], after_ms=current['p50_ms'],
                    before_peak_bytes=previous['traced_peak_bytes'], after_peak_bytes=current['traced_peak_bytes'])
        report['comparison'] = comparison
        print('All GT rows, result order, and 18 dump-format hashes match.', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
