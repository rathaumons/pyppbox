"""Compare one face backend per process on a fixed identity-folder sample.

Run backends sequentially to avoid hardware contention. Results include both
embedding-only and complete recognize() latency. The default split samples 40
files evenly from each identity's sorted directory: the earlier half trains the
SVM and calibrates a cosine threshold, the later half tests it. These are held-out
files, not independent capture sessions. This is not a population benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['FaceNet', 'ArcFace', 'AdaFace'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--facenet-data', type=Path)
    parser.add_argument('--detector', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--samples-per-identity', type=int, default=40)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples_per_identity < 4 or args.iterations < 1 or args.threads < 1:
        parser.error('Use at least four samples per identity and positive iterations/threads.')
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(args.threads)
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PYPPBOX_DISABLE_FILE_LOG'] = '1'
    os.environ['PYPPBOX_DISABLE_TERMINAL_LOG'] = '1'
    import cv2
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.metrics import roc_auc_score
    cv2.setNumThreads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if args.backend == 'FaceNet':
        if args.device != 'cpu':
            parser.error('This native Windows FaceNet comparison uses CPU TensorFlow.')
        from pyppbox.config.myconfig import RCFGFaceNet
        from pyppbox.modules.reiders.facenet import MyFaceNet, df
        root = args.facenet_data.resolve()
        cfg = RCFGFaceNet()
        cfg.set(dict(ri_name='FaceNet', gpu_mem=0.1, model_det=str(root / 'models/det'),
                     model_file=str(root / 'models/20180402-114759/20180402-114759.pb'),
                     classifier_pkl=str(root / 'classifier/gta5.pkl'), train_data=str(args.dataset),
                     batch_size=32, min_confidence=0.75,
                     yl_h_calibration=[-125, 75], yl_w_calibration=[-55, 55]))
        backend = MyFaceNet(cfg, auto_load=True)

        def aligned(image):
            rgb = backend.prepare_image(image)
            boxes, _ = df.detect_face(rgb, backend.minsize, backend.pnet, backend.rnet,
                                      backend.onet, backend.threshold, backend.factor)
            return None if not len(boxes) else backend.make_facenet_image(boxes, rgb)

        def embed(crop):
            return backend.sess.run(backend.embeddings, feed_dict={backend.images_placeholder: crop,
                                    backend.phase_train_placeholder: False})[0]
        providers = ['TensorFlow CPU']
    else:
        # Torch's installed CUDA libraries can supply ORT's CUDA/cuDNN dependencies.
        if args.device.startswith('cuda'):
            import torch
        from pyppbox.config.myconfig import RCFGArcFace, RCFGAdaFace
        from pyppbox.modules.reiders.onnxface import MyArcFace, MyAdaFace
        cls, factory = (RCFGArcFace, MyArcFace) if args.backend == 'ArcFace' else (RCFGAdaFace, MyAdaFace)
        cfg = cls()
        cfg.set(dict(ri_name=args.backend, model_det=str(args.detector.resolve()),
                     model_file=str(args.model.resolve()), device=args.device, num_threads=args.threads,
                     classifier_pkl=str((args.output / 'classifier.pkl').resolve()),
                     train_data=str(args.dataset.resolve())))
        backend = factory(cfg)
        backend._ensure_models()
        aligned, embed = backend.align_face, backend.embed_aligned
        providers = backend.session.get_providers()
    startup = time.perf_counter() - start
    print(f'{args.backend} {args.device}: initialized in {startup:.2f}s, providers={providers}', flush=True)
    records, features, labels, training, timing_images, aligned_crops = [], [], [], [], [], []
    identities = sorted(p for p in args.dataset.iterdir() if p.is_dir())
    for label, directory in enumerate(identities):
        files = sorted(p for p in directory.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))
        selected = np.linspace(0, len(files) - 1, min(args.samples_per_identity, len(files)), dtype=int)
        for position, index in enumerate(selected):
            path = files[index]
            is_train = position < len(selected) // 2
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'Cannot decode {path}')
            error = None
            try:
                crop = aligned(image)
                vector = None if crop is None else np.asarray(embed(crop), dtype=np.float32)
            except (ValueError, RuntimeError, cv2.error) as failure:
                crop = vector = None
                error = f'{type(failure).__name__}: {failure}'
            record = dict(file=str(path), identity=directory.name, train=is_train, detected=crop is not None)
            if error:
                record['extraction_error'] = error
            records.append(record)
            if crop is not None:
                features.append(vector)
                labels.append(label)
                training.append(is_train)
                record['feature_index'] = len(features) - 1
                if not is_train:
                    timing_images.append(image)
                    aligned_crops.append(crop)
        print(f'{args.backend}: processed {directory.name} ({len(selected)} images)', flush=True)
    features, labels, training = np.stack(features), np.array(labels), np.array(training)
    if len(set(labels[training])) != len(identities):
        raise ValueError('At least one identity has no detected training samples.')
    svm = SVC(C=1, kernel='rbf', probability=True, random_state=0).fit(features[training], labels[training])
    backend.model, backend.pnames = svm, [p.name.replace('_', ' ') for p in identities]
    probabilities = svm.predict_proba(features[~training])
    predictions = probabilities.argmax(axis=1)
    test_labels = labels[~training]
    confident = probabilities.max(axis=1) >= .75
    total_test = sum(not record['train'] for record in records)
    normalized = features / np.linalg.norm(features, axis=1, keepdims=True)

    def pairs(vectors, ids):
        i, j = np.triu_indices(len(vectors), k=1)
        return (vectors @ vectors.T)[i, j], ids[i] == ids[j]

    train_scores, train_same = pairs(normalized[training], labels[training])
    # Threshold is chosen on training impostor pairs, then held fixed for test pairs.
    threshold = float(np.quantile(train_scores[~train_same], .99, method='higher'))
    test_scores, test_same = pairs(normalized[~training], test_labels)

    def latency(call, inputs):
        for image in inputs[:5]:
            call(image)
        values = []
        for i in range(args.iterations):
            begin = time.perf_counter()
            call(inputs[i % len(inputs)])
            values.append(1000 * (time.perf_counter() - begin))
        return dict(iterations=len(values), mean_ms=float(np.mean(values)),
                    median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                    mean_calls_per_second=1000 / float(np.mean(values)))

    embedding_latency = latency(embed, aligned_crops)
    recognition_latency = latency(backend.recognize, timing_images)
    report = dict(backend=args.backend, device=args.device, active_providers=providers,
                  startup_seconds=startup, threads=args.threads, platform=platform.platform(),
                  dataset=str(args.dataset.resolve()), samples_per_identity=args.samples_per_identity,
                  test_images=total_test, detected_test_images=int((~training).sum()),
                  extraction_errors=sum('extraction_error' in record for record in records),
                  detection_rate=float((~training).sum() / total_test),
                  closed_set_top1_including_detection_failures=float((predictions == test_labels).sum() / total_test),
                  closed_set_top1_among_detected=float(np.mean(predictions == test_labels)),
                  accepted_correct_at_svm_075=float(np.sum(confident & (predictions == test_labels)) / total_test),
                  accepted_wrong_at_svm_075=float(np.sum(confident & (predictions != test_labels)) / total_test),
                  cosine_threshold_from_training_impostors_99pct=threshold,
                  test_pair_tar=float(np.mean(test_scores[test_same] > threshold)),
                  test_pair_far=float(np.mean(test_scores[~test_same] > threshold)),
                  test_pair_auc=float(roc_auc_score(test_same, test_scores)),
                  test_genuine_pairs=int(test_same.sum()), test_impostor_pairs=int((~test_same).sum()),
                  embedding_only=embedding_latency, detection_alignment_svm=recognition_latency,
                  limitations='Synthetic GTA V faces; held-out files from the same dataset/capture context. '
                              'Pair comparisons are dependent. No human-domain or open-set guarantee. '
                              'SVM scores are not calibrated unknown-person probabilities. '
                              'Latency samples include only successfully detected test crops.',
                  records=records)
    import importlib.metadata
    report['versions'] = {name: importlib.metadata.version(name) for name in
                          ('pyppbox', 'numpy', 'scikit-learn', 'tensorflow', 'onnxruntime-gpu', 'torch')}
    report['detector_model'] = str(args.detector.resolve()) if args.detector else str(root / 'models/det')
    report['embedding_model'] = str(args.model.resolve()) if args.model else str(root / 'models/20180402-114759/20180402-114759.pb')
    (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez(args.output / 'features.npz', embeddings=features, labels=labels, training=training)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2), flush=True)
    backend.close()


if __name__ == '__main__':
    main()
