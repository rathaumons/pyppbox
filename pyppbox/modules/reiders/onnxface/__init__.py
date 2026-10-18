# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Optional ArcFace/AdaFace ONNX inference with SCRFD five-point alignment.

Importing this module does not import InsightFace or ONNX Runtime. Models are
local files and are never downloaded implicitly. Classifiers are backend- and
checkpoint-specific; FaceNet pickles cannot be reused.
"""
import hashlib
import os
import pickle
from pathlib import Path

import cv2
import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class ONNXFace:
    """Shared single-face recognizer preserving pyppbox's (label, percent) API.

    Use ``load_classifier`` before ``recognize``. ``extract_embedding`` can be
    used without a classifier; it returns None when no face is detected.
    The largest detected face is selected from a crop, aligned to 112x112 using
    SCRFD keypoints, then embedded and L2-normalized. Confidence is the identity
    SVM's probability estimate, not the cosine score or a liveness measurement.
    Instances own their sessions and are not intended for concurrent calls.

    ``cfg.device`` selects CPU (default), ``cuda``, or ``cuda:N``. Configuration
    normalizes numeric GPU indices; the GUI offers an editable CPU/CUDA dropdown.
    ``mps`` and ``auto`` are not supported by these adapters. An unavailable or
    failed CUDA provider raises instead of silently selecting CPU at initialization.
    """
    NAME = None

    def __init__(self, cfg, auto_load=False):
        if cfg.ri_name != self.NAME:
            raise ValueError(f'{type(self).__name__} requires a {self.NAME} configuration.')
        from copy import deepcopy
        self.cfg = deepcopy(cfg)
        self.unk, self.err = cfg.unified_strings.unk_fid, cfg.unified_strings.err_fid
        self.classifier_file = cfg.classifier_pkl
        self.train_data = cfg.train_data
        self.min_confidence = 100.0 * cfg.min_confidence
        self.auto_load = auto_load
        self.session = self.detector = self.model = None
        self.pnames = []
        self._closed = False
        if auto_load:
            try:
                self.load_classifier()
            except BaseException:
                self.close()
                raise

    def _check_open(self):
        if self._closed:
            raise RuntimeError('This reider is closed; create a new instance.')

    def _signature(self):
        return dict(backend=self.NAME, model_sha256=_sha256(self.cfg.model_file),
                    detector_sha256=_sha256(self.cfg.model_det), input_size=112,
                    input_color=self.cfg.input_color, input_mean=self.cfg.input_mean,
                    input_std=self.cfg.input_std, alignment='scrfd-largest-arcface-5pt-v1',
                    det_size=self.cfg.det_size, det_thresh=self.cfg.det_thresh)

    def _ensure_models(self):
        self._check_open()
        if self.session is not None:
            return
        try:
            import onnxruntime as ort
            from insightface.model_zoo.scrfd import SCRFD
        except ImportError as error:
            raise ImportError('ArcFace/AdaFace require the optional face dependencies: '
                              'python -m pip install "pyppbox[faces]" (Python 3.10+).') from error
        self.cfg.validateRuntime()
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.cfg.num_threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        cuda = self.cfg.device.startswith('cuda')
        if cuda:
            # pyppbox already depends on torch; its matching CUDA/cuDNN DLLs can
            # satisfy ORT's provider dependencies without a machine-wide install.
            try:
                import torch  # noqa: F401
            except ImportError:
                pass  # A separately installed CUDA runtime may still be usable.
        if cuda and 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('CUDA was requested but this ONNX Runtime has no CUDA provider. '
                               'Install onnxruntime-gpu or choose device=cpu.')
        index = int(self.cfg.device.split(':')[1]) if ':' in self.cfg.device else 0
        providers = [('CUDAExecutionProvider', {'device_id': index}), 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        session = ort.InferenceSession(self.cfg.model_file, sess_options=options, providers=providers)
        det_session = ort.InferenceSession(self.cfg.model_det, sess_options=options, providers=providers)
        if cuda and any('CUDAExecutionProvider' not in s.get_providers() for s in (session, det_session)):
            raise RuntimeError('CUDA provider initialization failed; refusing an unreported CPU fallback.')
        inputs = session.get_inputs()
        shape = inputs[0].shape if len(inputs) == 1 else []
        if (len(shape) != 4 or list(shape[1:]) != [3, 112, 112]
                or isinstance(shape[0], int) and shape[0] != 1
                or inputs[0].type != 'tensor(float)'):
            raise ValueError('Embedding model must accept one NCHW float32 input with shape [N,3,112,112].')
        detector = SCRFD(model_file=self.cfg.model_det, session=det_session)
        detector.prepare(ctx_id=index if cuda else -1,
                         input_size=(self.cfg.det_size, self.cfg.det_size),
                         det_thresh=self.cfg.det_thresh)
        self.input_name = inputs[0].name
        self.session, self.detector = session, detector

    @staticmethod
    def _image(image, is_bgr=True):
        if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
                or not image.size or image.dtype != np.uint8):
            raise ValueError('Expected a nonempty uint8 HWC three-channel image.')
        return image if is_bgr else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def align_face(self, image, is_bgr=True):
        """Return the aligned BGR 112x112 face, or None when detection fails."""
        self._ensure_models()
        from insightface.utils.face_align import norm_crop
        image = self._image(image, is_bgr)
        boxes, points = self.detector.detect(image)
        if not len(boxes):
            return None
        if points is None or np.asarray(points).shape != (len(boxes), 5, 2):
            raise ValueError('Detector must provide five alignment landmarks for each face.')
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return norm_crop(image, points[int(np.argmax(areas))], image_size=112)

    def embed_aligned(self, image, is_bgr=True):
        """Embed an already aligned 112x112 crop; return a normalized vector."""
        self._ensure_models()
        image = self._image(image, is_bgr)
        if image.shape[:2] != (112, 112):
            raise ValueError('embed_aligned expects a 112x112 crop; use extract_embedding for detection.')
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0 / self.cfg.input_std,
                                    mean=(self.cfg.input_mean,) * 3,
                                    swapRB=self.cfg.input_color == 'RGB')
        output = np.asarray(self.session.run(None, {self.input_name: blob})[0])
        if output.ndim != 2 or output.shape[0] != 1 or not output.shape[1] or not np.isfinite(output).all():
            raise ValueError('Embedding model returned an invalid feature vector.')
        vector = output[0].astype(np.float32)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('Embedding model returned a zero or invalid vector.')
        return vector / norm

    def extract_embedding(self, image, is_bgr=True):
        """Detect, align, and embed one face, or return None when none is found."""
        aligned = self.align_face(image, is_bgr)
        return None if aligned is None else self.embed_aligned(aligned)

    def load_classifier(self):
        """Load a trusted classifier pickle, checking model/preprocessing identity.

        Pickle files execute code when loaded; use only classifiers you trust.
        A failed replacement leaves the preceding classifier available.
        """
        self._check_open()
        with open(self.classifier_file, 'rb') as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or payload.get('format') != 'pyppbox.onnxface.v1':
            raise ValueError('Train a new ArcFace/AdaFace classifier; legacy classifier pickles cannot be reused.')
        if payload.get('signature') != self._signature():
            raise ValueError('Classifier checkpoint/preprocessing mismatch; retrain for this configuration.')
        model, names = payload['model'], payload['class_names']
        if (not isinstance(names, list) or len(names) < 2 or not all(isinstance(n, str) and n for n in names)
                or list(model.classes_) != list(range(len(names))) or not callable(getattr(model, 'predict_proba', None))):
            raise ValueError('Classifier labels or probability interface are invalid.')
        self._ensure_models()
        self.model, self.pnames = model, list(names)

    def recognize(self, image, is_bgr=True):
        self._check_open()
        if self.model is None:
            raise RuntimeError('Call load_classifier() before recognize(), or use auto_load=True.')
        embedding = self.extract_embedding(image, is_bgr)
        if embedding is None:
            return self.err, 0.0
        probabilities = np.asarray(self.model.predict_proba(embedding[None]))
        if (probabilities.shape != (1, len(self.pnames)) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or np.any(probabilities > 1)):
            raise ValueError('Classifier returned an invalid probability vector.')
        index = int(probabilities[0].argmax())
        confidence = float(probabilities[0, index] * 100)
        return (self.unk if confidence < self.min_confidence else self.pnames[index]), confidence

    def recognize_file(self, path):
        return self.recognize(cv2.imread(os.fspath(path)))

    def train_classifier(self, C=1.0, kernel='rbf', probability=True, decision_function_shape='ovr'):
        """Train an SVM from identity subdirectories of unaligned face images.

        Each identity needs at least one detectable face; undetected images are
        skipped and the count is reported. This trains only the identity SVM.
        Network weights remain frozen. Load the new classifier before inference.
        """
        self._check_open()
        if not probability:
            raise ValueError('probability=True is required by recognize().')
        from sklearn.svm import SVC
        from pyppbox.utils.logtools import add_info_log
        directories = sorted(p for p in Path(self.train_data).iterdir() if p.is_dir())
        if len(directories) < 2:
            raise ValueError('Training requires at least two identity directories.')
        names, features, labels, skipped = [], [], [], 0
        for label, directory in enumerate(directories):
            count = 0
            for path in sorted(directory.iterdir()):
                if not path.is_file() or path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                    continue
                image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f'Cannot decode training image: {path}')
                embedding = self.extract_embedding(image)
                if embedding is None:
                    skipped += 1
                    continue
                features.append(embedding)
                labels.append(label)
                count += 1
            if not count:
                raise ValueError(f'Identity has no detectable training faces: {directory}')
            names.append(directory.name.replace('_', ' '))
        if len(set(names)) != len(names):
            raise ValueError('Identity directory names collapse to duplicate display labels.')
        model = SVC(C=C, kernel=kernel, probability=True, decision_function_shape=decision_function_shape,
                    random_state=0).fit(np.stack(features), labels)
        payload = dict(format='pyppbox.onnxface.v1', signature=self._signature(), model=model,
                       class_names=names, samples=len(features), skipped_no_face=skipped)
        with open(self.classifier_file, 'wb') as stream:
            pickle.dump(payload, stream)
        Path(self.classifier_file).with_suffix('.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
        add_info_log(f'--------RI : Trained {self.NAME} SVM on {len(features)} faces; skipped {skipped} undetected images.')

    def close(self):
        """Release only this instance's session and classifier references."""
        self._closed = True
        self.session = self.detector = self.model = None
        self.pnames = []


class MyArcFace(ONNXFace):
    NAME = 'ArcFace'


class MyAdaFace(ONNXFace):
    NAME = 'AdaFace'
