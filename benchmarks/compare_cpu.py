# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Run sequential, alternating installed-wheel CPU comparisons on one host.

Correctness differences fail; runner timings are advisory. Compare revisions on
the same runner only. Output samples, provenance, and logs survive failures.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


def equivalent(expected, actual, context='output', tolerance=1e-6):
    """Require exact labels/order/counts and finite, close floating-point values."""
    if type(expected) is not type(actual):
        raise AssertionError(f'{context}: types differ')
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f'{context}: keys differ')
        for key in expected:
            equivalent(expected[key], actual[key], f'{context}/{key}', tolerance)
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise AssertionError(f'{context}: lengths differ')
        for index, (left, right) in enumerate(zip(expected, actual)):
            equivalent(left, right, f'{context}/{index}', tolerance)
    elif isinstance(expected, float):
        if not math.isfinite(expected) or not math.isfinite(actual) or abs(expected-actual) > tolerance:
            raise AssertionError(f'{context}: {expected!r} != {actual!r}')
    elif expected != actual:
        raise AssertionError(f'{context}: {expected!r} != {actual!r}')


def compare_reports(reference, report):
    for key in ('schema', 'python', 'platform', 'architecture', 'frame_shape', 'packages', 'threads', 'batch_size',
                'frame_sha256', 'blank_frames', 'asset_sha256', 'detector', 'tracker', 'reider'):
        equivalent(reference[key], report[key], key)
    equivalent(list(reference['workloads']), list(report['workloads']), 'workloads')
    for name, expected in reference['workloads'].items():
        actual = report['workloads'][name]
        for key in ('outputs', 'reid_counts', 'export_sha256', 'rendered_sha256'):
            equivalent(expected[key], actual[key], f'{name}/{key}')
        if sum(actual['reid_counts']) == 0 or not any(row['people'] for row in actual['outputs']):
            raise AssertionError(f'{name}: benchmark did not exercise recognition')
        for index in reference['blank_frames']:
            if actual['outputs'][index]['people']:
                raise AssertionError(f'{name}: blank frame {index} was not empty')


def summarize(records):
    summary = {}
    for name, rows in records.items():
        summary[name] = dict(import_seconds=statistics.median(r['import_seconds'] for r in rows),
            model_load_seconds=statistics.median(r['model_load_seconds'] for r in rows), workloads={})
        for workload in rows[0]['workloads']:
            summary[name]['workloads'][workload] = {
                metric:statistics.median(r['workloads'][workload][metric] for r in rows)
                for metric in ('mean_ms','p50_ms','p95_ms','p99_ms','processing_fps','peak_sampled_rss')}
            summary[name]['workloads'][workload]['trial_mean_ms'] = [r['workloads'][workload]['mean_ms'] for r in rows]
            summary[name]['workloads'][workload]['counts'] = rows[0]['workloads'][workload]['reid_counts']
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-ref', required=True)
    parser.add_argument('--candidate-ref', required=True)
    parser.add_argument('--frames', type=int, default=24)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('At least one trial is required.')
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    roots = dict(baseline=args.baseline_root.resolve(), candidate=args.candidate_root.resolve())
    records = {key:[] for key in roots}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1')
    # The worker selects the wheel explicitly; inherited checkout paths must not
    # supply an accidental second pyppbox or alternate detector backend.
    env.pop('PYTHONPATH', None)
    for trial in range(args.trials):
        order = ('baseline', 'candidate') if trial % 2 == 0 else ('candidate', 'baseline')
        for name in order:
            output = directory / f'{name}-{trial+1}.json'
            command = [sys.executable, str(Path(__file__).with_name('pipeline_cpu.py').resolve()),
                '--package-root', str(roots[name]), '--assets', str(args.assets.resolve()),
                '--output', str(output), '--frames', str(args.frames), '--threads', str(args.threads)]
            print(f'Trial {trial+1}: {name}', flush=True)
            with (directory / f'{name}-{trial+1}.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=900)
            records[name].append(json.loads(output.read_text(encoding='utf-8')))
    reference = records['baseline'][0]
    for rows in records.values():
        for row in rows:
            compare_reports(reference, row)
    result = dict(outputs_match=True, confidence_absolute_tolerance=1e-6,
        baseline_ref=args.baseline_ref, candidate_ref=args.candidate_ref, summary=summarize(records),
        timing_policy='Advisory medians on this runner; no shared-runner speed gate or cross-OS ranking.',
        scope='24 consecutive predecoded frames by default, two blank frames, scalar CPU YOLO pose/SORT/Torchreid; refresh forces ReID; render_io includes JPEG/result writes. Decode/model load are separate.')
    (directory / 'comparison.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    lines = ['## CPU installed-wheel comparison', '',
        f'Baseline `{args.baseline_ref}`; candidate `{args.candidate_ref}`.', '',
        'Outputs, order, pose, identities, counts, and result exports match. Timings are advisory.', '',
        '| Workload | Baseline FPS | Candidate FPS | Mean latency change |',
        '|---|---:|---:|---:|']
    for workload, baseline in result['summary']['baseline']['workloads'].items():
        candidate = result['summary']['candidate']['workloads'][workload]
        change = 100*(candidate['mean_ms']/baseline['mean_ms']-1)
        lines.append(f'| {workload} | {baseline["processing_fps"]:.2f} | {candidate["processing_fps"]:.2f} | {change:+.1f}% |')
    lines += ['', result['scope'], '', result['timing_policy'], '']
    markdown = '\n'.join(lines)
    (directory / 'summary.md').write_text(markdown, encoding='utf-8')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as stream:
            stream.write(markdown)
    print(markdown)


if __name__ == '__main__':
    main()
