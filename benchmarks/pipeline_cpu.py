# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Bounded real CPU detection/tracking/ReID comparisons, using an explicit package.

Run each revision in a fresh process with identical assets/dependencies. Input
frames are decoded before timing: results measure processing, rendering, and output
I/O, not capture/decode FPS. Models are warmed before timed, fresh-tracker streams.
The refresh workload deliberately requests ReID on each visible person; it is a
stress case, not normal tracking policy. Default inference remains scalar.
"""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True, help='JSON with an assets mapping of path/sha256 entries.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=24)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--workloads', nargs='+', choices=('stream', 'refresh', 'render_io'),
                        default=['stream', 'refresh', 'render_io'])
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    if args.frames < 12 or args.threads < 1 or args.batch_size < 1:
        parser.error('Use at least 12 frames, positive CPU threads, and a positive batch limit.')
    root = args.package_root.resolve()
    sys.path.insert(0, str(root))
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['PYPPBOX_DISABLE_FILE_LOG'] = '1'
    os.environ['PYPPBOX_DISABLE_TERMINAL_LOG'] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    sys.dont_write_bytecode = True
    assets = json.loads(args.assets.read_text(encoding='utf-8'))['assets']
    paths = {}
    for key in ('video', 'detector', 'embedding', 'classifier', 'base'):
        entry = assets[key]
        path = Path(entry['path'])
        path = path if path.is_absolute() else args.assets.resolve().parent / path
        if not path.is_file() or sha256(path) != entry['sha256']:
            raise ValueError(f'Asset missing or checksum differs: {key}: {path}')
        paths[key] = path.resolve()

    import_start = time.perf_counter()
    import cv2
    import numpy as np
    import psutil
    import torch
    import pyppbox
    from pyppbox.ppb.mt import MT
    from pyppbox.modules.trackers.sort.origin.sort import KalmanBoxTracker
    from pyppbox.utils.restools import ResIO
    from pyppbox.utils.visualizetools import visualizePeople
    import_seconds = time.perf_counter() - import_start
    if not Path(pyppbox.__file__).resolve().is_relative_to(root):
        raise RuntimeError(f'Wrong package imported: {pyppbox.__file__}')
    torch.manual_seed(351)
    np.random.seed(351)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    process = psutil.Process()
    detector = dict(dt_name='YOLO_Ultralytics', model_file=str(paths['detector']), device='cpu',
                    conf=.4, iou=.7, imgsz=640, max_det=100, show_boxes=False, repspoint_calibration=.25)
    reider = dict(ri_name='Torchreid', model_name='osnet_ain_x1_0', model_path=str(paths['embedding']),
                  classifier_pkl=str(paths['classifier']), train_data=str(paths['classifier'].parent),
                  min_confidence=.35, device='cpu')
    tracker = dict(tk_name='SORT', max_age=1, min_hits=3, iou_threshold=.3)
    pipeline = MT()
    started = time.perf_counter()
    pipeline.setMainDetector(detector)
    pipeline.setMainReIDer(reider, auto_load=True)
    model_load_seconds = time.perf_counter() - started
    source = cv2.VideoCapture(str(paths['video']))
    frames = []
    decode_started = time.perf_counter()
    try:
        for index in range(args.frames):
            valid, frame = source.read()
            if not valid:
                raise ValueError(f'Video ended before frame {index}.')
            if index in (8, 9):
                frame = np.zeros_like(frame)
            frames.append(frame)
    finally:
        source.release()
    decode_seconds = time.perf_counter() - decode_started
    hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]

    def snapshot(people, counts):
        return dict(counts=list(counts), people=[dict(cid=int(person.cid), init_id=int(person.init_id),
            box=np.asarray(person.box_xyxy).tolist(), point=[int(v) for v in person.repspoint],
            keypoints=np.asarray(person.keypoints).tolist(), confidence=float(person.det_conf),
            faceid=str(person.faceid), deepid=str(person.deepid),
            face_confidence=float(person.faceid_conf), deep_confidence=float(person.deepid_conf)) for person in people])

    def reid(frame, people):
        options = dict(img_is_mat=True, deduplicate=True)
        if args.batch_size != 1:
            options['batch_size'] = args.batch_size
        return pipeline.reidPeople(frame, people, **options)

    records = {}
    try:
        # Shape/model warmup is separate from all timed streams and their counters.
        pipeline.setMainTracker(tracker)
        for _ in range(2):
            people, _ = pipeline.detectPeople(frames[0], img_is_mat=True, visual=False)
            people = pipeline.trackPeople(frames[0], people, img_is_mat=True)
            for person in people:
                person.deepid = 'Unknown'
            reid(frames[0], people)
        for workload in args.workloads:
            pipeline.setMainTracker(tracker)
            # Isolated benchmark process only: give every fresh stream the same ID origin.
            KalmanBoxTracker.count = 0
            recorder = ResIO()
            timings, outputs, rss, rendered_sha256 = [], [], [], []
            export_ms, export_sha256 = 0., None
            with tempfile.TemporaryDirectory(prefix='pyppbox-e2e-') as directory:
                for index, frame in enumerate(frames):
                    started = time.perf_counter()
                    people, _ = pipeline.detectPeople(frame, img_is_mat=True, visual=False)
                    detected = time.perf_counter()
                    people = pipeline.trackPeople(frame, people, img_is_mat=True)
                    tracked = time.perf_counter()
                    if workload == 'refresh':
                        for person in people:
                            person.deepid = 'Unknown'
                    people, counts = reid(frame, people)
                    recognized = time.perf_counter()
                    if workload == 'render_io':
                        recorder.addPeople(index, people)
                        image = visualizePeople(frame.copy(), people, img_is_mat=True, show_reid=counts)
                        if not cv2.imwrite(str(Path(directory) / f'frame-{index:04d}.jpg'), image):
                            raise OSError('Could not write benchmark frame.')
                    finished = time.perf_counter()
                    if workload == 'render_io':
                        rendered_sha256.append(hashlib.sha256(image.tobytes()).hexdigest())
                    timings.append(dict(detection_ms=1000*(detected-started), tracking_ms=1000*(tracked-detected),
                        reid_ms=1000*(recognized-tracked), render_io_ms=1000*(finished-recognized),
                        total_ms=1000*(finished-started)))
                    outputs.append(snapshot(people, counts))
                    rss.append(process.memory_info().rss)
                if workload == 'render_io':
                    started = time.perf_counter()
                    recorder.dumpAll(directory)
                    export_ms = 1000*(time.perf_counter()-started)
                    exports = list(Path(directory).glob('*.txt'))
                    if len(exports) != 1:
                        raise AssertionError('Expected one result export.')
                    export_sha256 = sha256(exports[0])
            values = [sample['total_ms'] for sample in timings]
            records[workload] = dict(samples=timings, outputs=outputs, first_frame_ms=values[0],
                p50_ms=float(np.median(values[1:])), p95_ms=float(np.percentile(values[1:],95)),
                p99_ms=float(np.percentile(values[1:],99)), mean_ms=float(np.mean(values)),
                processing_fps=1000*len(values)/(sum(values)+export_ms), export_ms=export_ms,
                export_sha256=export_sha256, rss_samples=rss, peak_sampled_rss=max(rss),
                rendered_sha256=rendered_sha256,
                reid_counts=[sum(output['counts'][i] for output in outputs) for i in (0,1)])
            print(f'{workload}: mean {np.mean(values):.3f} ms/frame; counts {records[workload]["reid_counts"]}', flush=True)
        if hashes != [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]:
            raise AssertionError('Processing changed an input frame.')
        report = dict(schema=1, scope='CPU processing of predecoded frames; includes optional rendering/output',
            version=pyppbox.__version__, package_file=pyppbox.__file__, platform=platform.platform(),
            python=sys.version, processor=platform.processor(), architecture=platform.machine(),
            logical_cpus=os.cpu_count(), threads=args.threads, frame_shape=list(frames[0].shape),
            batch_size=args.batch_size, frame_sha256=hashes, blank_frames=[8,9],
            asset_sha256={key:assets[key]['sha256'] for key in paths},
            packages={name:metadata.version(name) for name in ('torch','torchvision','numpy','scipy','scikit-learn',
                'opencv-contrib-python','pyppbox-torchreid','pyppbox-ultralytics','psutil')},
            detector=detector, tracker=tracker, reider=reider, import_seconds=import_seconds,
            model_load_seconds=model_load_seconds, decode_seconds=decode_seconds, workloads=records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    finally:
        close = getattr(pipeline, 'close', None)
        if close is not None:
            close()


if __name__ == '__main__':
    main()
