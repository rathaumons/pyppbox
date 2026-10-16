# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Fresh-process startup/configuration benchmarks, without weights or inference.

    python benchmarks/startup_imports.py --output plans/validation/v4-startup/before.json
    python benchmarks/startup_imports.py --output plans/validation/v4-startup/after.json \
        --compare plans/validation/v4-startup/before.json

Use --package-root with a frozen checkout/package parent, and --gui to include
an offscreen Torchreid settings panel. No files are saved by the settings panel.
Raw timings, module inventories, normalized output hashes, and RSS are recorded.
Each sample launches a separate interpreter; OS filesystem caches are not cleared.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = ('root_import', 'config_import', 'config_all', 'torchreid_config', 'legacy_catalog',
         'sort_import', 'standalone_import', 'facade_first_use', 'explicit_MT')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized(value, package):
    if isinstance(value, dict):
        return {str(k): normalized(v, package) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(v, package) for v in value]
    if isinstance(value, str):
        return value.replace('\\', '/').replace(str(package).replace('\\', '/'), '<PACKAGE>')
    return value


def documents(cfg):
    return {name: value.getDocument() for name, value in vars(cfg).items()
            if hasattr(value, 'getDocument')}


def catalog_rows(catalog):
    return [vars(model) for model in catalog.model_list]


def rss():
    # Probe after the operation/module snapshot. ctypes adds small probe overhead
    # to these process RSS values; they are not Python allocation measurements.
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('peak_rss', 'rss', 'peak_paged', 'paged',
                    'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not query(process(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return dict(rss_bytes=counters.rss, peak_rss_bytes=counters.peak_rss)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(rss_bytes=None, peak_rss_bytes=peak if sys.platform == 'darwin' else peak * 1024)


def worker(case, package_root):
    sys.path.insert(0, str(package_root))
    details, warm = {}, None
    started = time.perf_counter_ns()
    if case == 'root_import':
        import pyppbox
        output = dict(version=pyppbox.__version__, exports=sorted(pyppbox._exports))
    elif case == 'config_import':
        from pyppbox.config.myconfig import MyConfigurator, RCFGTorchreid
        output = [MyConfigurator.__name__, RCFGTorchreid.__name__]
    elif case in ('config_all', 'torchreid_config'):
        from pyppbox.config.myconfig import MyConfigurator
        cfg = MyConfigurator(set_all_modules=case == 'config_all')
        if case == 'torchreid_config':
            cfg.setAllRCFG()
        output = dict(documents=documents(cfg), model_wh=cfg.rcfg_torchreid.model_wh,
                      models=catalog_rows(cfg.rcfg_torchreid.model_dict))
    elif case == 'legacy_catalog':
        from pyppbox.modules.reiders.torchreid.model_dict import TorchreidModelDict
        catalog = TorchreidModelDict()
        output = dict(models=catalog_rows(catalog), unknown=catalog.getWH('unknown.pt'),
                      architecture=catalog.findModelArch('OSNET_AIN_MS_D_C.PTH.TAR'))
    elif case == 'sort_import':
        from pyppbox.modules.trackers.sort import MySORT
        from types import SimpleNamespace
        tracker = MySORT(SimpleNamespace(max_age=2, min_hits=1, iou_threshold=.3))
        output = dict(name=type(tracker).__name__, empty=tracker.update([]))
    elif case in ('standalone_import', 'facade_first_use'):
        import pyppbox.standalone as facade
        details['import_ms'] = (time.perf_counter_ns() - started) / 1e6
        import pyppbox.ppb as api
        output = dict(exports=api.__all__)
        if case == 'facade_first_use':
            first = time.perf_counter_ns()
            output['main'] = facade.getMainConfig()
            details['first_call_ms'] = (time.perf_counter_ns() - first) / 1e6
            output['same_config'] = facade.getConfig() is facade.getConfig()
            warm = facade.getMainConfig
    elif case == 'explicit_MT':
        from pyppbox.standalone import MT
        details['import_ms'] = (time.perf_counter_ns() - started) / 1e6
        first = time.perf_counter_ns()
        pipeline = MT()
        details['construct_ms'] = (time.perf_counter_ns() - first) / 1e6
        output = dict(name=MT.__name__, module=MT.__module__, main=pipeline.getMainConfig())
        warm = MT
    elif case == 'torchreid_gui':
        from PyQt6 import QtWidgets
        from pyppbox.gui.ui_torchreid import Ui_Torchreid
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        dialog = QtWidgets.QDialog()
        ui = Ui_Torchreid(0, '')
        ui.setupUi(dialog)
        output = dict(documents=documents(ui.mycfg), models=catalog_rows(ui.md),
                      text=ui.dr_model_name_lineEdit.text(), confidence=ui.dr_min_confidence_spinBox.value(),
                      size=[dialog.width(), dialog.height()])
    else:
        raise ValueError(case)
    elapsed = (time.perf_counter_ns() - started) / 1e6
    import pyppbox
    package = Path(pyppbox.__file__).resolve().parent
    assert package == (package_root / 'pyppbox').resolve(), package
    modules = sorted(name for name, module in sys.modules.items() if module is not None)
    if warm is not None:
        times = []
        for _ in range(9):
            begin = time.perf_counter_ns()
            for _ in range(100):
                warm()
            times.append((time.perf_counter_ns() - begin) / 1e6 / 100)
        details['warm_call_ms'] = statistics.median(times)
    output = normalized(output, package)
    report = dict(operation_ms=elapsed, details=details, output=output,
                  fingerprint=digest(json.dumps(output, sort_keys=True).encode()),
                  modules=modules, module_count=len(modules), package_file=str(package / '__init__.py'),
                  memory=rss())
    print(json.dumps(report))


def environment():
    packages = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'opencv-contrib-python', 'PyYAML',
                 'filterpy', 'lapx', 'matplotlib', 'scikit-image', 'PyQt6', 'torch',
                 'pyppbox-torchreid', 'pyppbox-ultralytics'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, executable=sys.executable, os=platform.platform(),
                cpu=platform.processor(), logical_cpus=os.cpu_count(), packages=packages,
                thread_environment={name: os.environ.get(name) for name in (
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--package-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--worker', choices=CASES + ('torchreid_gui',))
    args = parser.parse_args()
    args.package_root = args.package_root.resolve()
    if args.worker:
        worker(args.worker, args.package_root)
        return
    if args.output is None or args.samples < 3:
        parser.error('--output and at least three samples are required')
    package = args.package_root / 'pyppbox'
    report = dict(schema=1, captured_at_utc=datetime.now(timezone.utc).isoformat(),
        git_head=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        benchmark_sha256=digest(Path(__file__).read_bytes()), environment=environment(),
        source_sha256={str(path.relative_to(package)): digest(path.read_bytes().replace(b'\r\n', b'\n'))
                       for path in sorted(package.rglob('*.py'))},
        config_sha256={str(path.relative_to(package)): digest(path.read_bytes())
                       for path in sorted(package.rglob('*.yaml'))}, cases={})
    child_env = dict(os.environ, PYTHONPATH=str(args.package_root),
        PYPPBOX_DISABLE_FILE_LOG='1', PYPPBOX_DISABLE_TERMINAL_LOG='1',
        QT_QPA_PLATFORM='offscreen', MPLBACKEND='Agg')
    with tempfile.TemporaryDirectory(prefix='pyppbox-startup-') as temporary:
        for case in CASES + (('torchreid_gui',) if args.gui else ()):
            samples = []
            # One separate-process priming run; never reuse an interpreter.
            for _ in range(args.samples + 1):
                started = time.perf_counter_ns()
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    '--worker', case, '--package-root', str(args.package_root)],
                    cwd=temporary, env=child_env, capture_output=True, text=True)
                if completed.returncode:
                    raise RuntimeError(f'{case} failed:\n{completed.stderr}')
                elapsed = (time.perf_counter_ns() - started) / 1e6
                sample = json.loads(completed.stdout)
                sample.update(process_wall_ms=elapsed, stderr=completed.stderr)
                samples.append(sample)
            samples = samples[1:]
            assert len({sample['fingerprint'] for sample in samples}) == 1, case
            record = dict(samples=samples, fingerprint=samples[0]['fingerprint'],
                p50_operation_ms=statistics.median(s['operation_ms'] for s in samples),
                p50_process_wall_ms=statistics.median(s['process_wall_ms'] for s in samples),
                p50_rss_bytes=statistics.median(s['memory']['rss_bytes'] for s in samples)
                    if samples[0]['memory']['rss_bytes'] is not None else None)
            report['cases'][case] = record
            print(case, f"{record['p50_operation_ms']:.3f} ms", flush=True)
    if args.compare:
        before = json.loads(args.compare.read_text(encoding='utf-8'))
        assert before['environment'] == report['environment'], 'Environment changed'
        assert before['benchmark_sha256'] == report['benchmark_sha256'], 'Benchmark changed'
        assert before['config_sha256'] == report['config_sha256'], 'Config/catalog YAML changed'
        assert set(before['cases']) == set(report['cases'])
        for name, case in report['cases'].items():
            assert before['cases'][name]['fingerprint'] == case['fingerprint'], (name, 'Output changed')
        report['comparison'] = 'All normalized public outputs/settings/catalog fingerprints match.'
        print(report['comparison'], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
