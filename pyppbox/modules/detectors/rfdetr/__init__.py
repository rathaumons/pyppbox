# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""RF-DETR detection adapter; the optional backend is imported at initialization."""
from pathlib import Path
import sys

import cv2
import numpy as np

from pyppbox.utils.commontools import to_xywh
from pyppbox.utils.persontools import Person, findRepspoint, findRepspointBB


class MyRFDETR:
    """Adapt RF-DETR boxes to pyppbox's existing detector and Person interfaces.

    The editable GUI device field accepts CPU, CUDA/indexed GPU, MPS, and
    ``auto`` settings. Integer indices are passed as ``cuda:N``; ``auto`` leaves
    selection to RF-DETR. The installed backend and hardware determine availability.

    Attributes
    ----------
    cfg : pyppbox.config.myconfig.DCFGRFDETR
        Validated detector configuration.
    model : object
        RF-DETR model, retained across frames. With ``cfg.optimize=True``, the
        upstream model is prepared in place for inference only; training/export
        require a separate RF-DETR instance.
    """

    def __init__(self, cfg):
        """Load one RF-DETR model and optionally prepare it for inference.

        Parameters
        ----------
        cfg : pyppbox.config.myconfig.DCFGRFDETR
            Model, device, confidence, and people-class configuration.

        Raises
        ------
        ImportError
            Python is older than 3.10 or the optional backend is not installed.
        FileNotFoundError
            The configured local checkpoint does not exist.

        Notes
        -----
        An empty ``model_file`` lets RF-DETR download its pretrained checkpoint
        to its cache (controlled by ``RF_HOME``). Backend/device failures propagate.
        """
        self.model = None
        try:
            if sys.version_info < (3, 10):
                raise ImportError('RF_DETR requires Python 3.10+ and pyppbox[rfdetr].')
            if cfg.model_file and not Path(cfg.model_file).is_file():
                raise FileNotFoundError(f'RF_DETR checkpoint not found: {cfg.model_file}')
            try:
                import rfdetr
            except ModuleNotFoundError as error:
                if error.name != 'rfdetr':
                    raise
                raise ImportError('RF_DETR is optional. Install it with: pip install "pyppbox[rfdetr]"') from error
            self.cfg = cfg
            model_class = getattr(rfdetr, {
                'nano': 'RFDETRNano', 'small': 'RFDETRSmall',
                'medium': 'RFDETRMedium', 'large': 'RFDETRLarge',
            }[cfg.model_size])
            options = {}
            if cfg.device != 'auto':
                options['device'] = f'cuda:{cfg.device}' if isinstance(cfg.device, int) else cfg.device
            if cfg.model_file:
                options['pretrain_weights'] = cfg.model_file
            self.model = model_class(**options)
            if cfg.optimize:
                self.model.inference(compile=False, inplace=True)
        except BaseException:
            self.close()
            raise

    def close(self):
        """Release the owned RF-DETR backend model."""
        self.model = None

    def _detections(self, img, classes, min_width_filter):
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or (
                img.ndim != 3 or img.shape[2] != 3 or not img.size):
            raise ValueError('RF_DETR expects a nonempty uint8 BGR image with shape (H, W, 3).')
        if min_width_filter < 0:
            raise ValueError('min_width_filter must be nonnegative.')
        # RF-DETR takes RGB; pyppbox's callers and drawing utilities use BGR.
        result = self.model.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                    threshold=self.cfg.conf, include_source_image=False)
        boxes = np.asarray(result.xyxy)
        if len(boxes) == 0:
            return np.empty((0, 4), dtype=int), np.empty(0, dtype=float)
        scores = np.asarray(result.confidence)
        class_ids = np.asarray(result.class_id)
        if boxes.shape != (len(boxes), 4) or scores.shape != (len(boxes),) or class_ids.shape != scores.shape:
            raise ValueError('RF_DETR returned inconsistent boxes, confidence scores, or class IDs.')
        keep = np.isin(class_ids, classes) & np.isfinite(boxes).all(axis=1)
        keep &= np.isfinite(scores) & (scores >= self.cfg.conf) & (scores <= 1)
        boxes, scores = boxes[keep].copy(), scores[keep]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img.shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img.shape[0])
        boxes = boxes.astype(int)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        keep &= boxes[:, 2] - boxes[:, 0] >= min_width_filter
        boxes, scores = boxes[keep], scores[keep]
        order = np.argsort(-scores, kind='stable')[:self.cfg.max_det]
        return boxes[order], scores[order]

    @staticmethod
    def _draw(img, box, point):
        cv2.circle(img, point, 5, (0, 0, 255), -1)
        cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), (255, 255, 0), 2)

    def detect(self, img, visual=True, classes=None, min_width_filter=15):
        """Detect configured classes in one BGR frame.

        Parameters
        ----------
        img : ``numpy.ndarray``
            Nonempty uint8 BGR frame. The same array is returned.
        visual : bool
            Defaults to True. Draw boxes/points in place when enabled.
        classes : int or list[int] or None
            Native RF-DETR class IDs. None (default) selects ``person_class_id``.
        min_width_filter : int
            Inclusive minimum box width in original-image pixels; defaults to 15.

        Returns
        -------
        tuple
            ``(img, boxes_xywh, boxes_xyxy, repspoints, confidences)``. Aligned
            lists contain integer boxes, points, and float confidences in [0, 1],
            sorted by decreasing confidence. Empty results contain empty lists.

        Notes
        -----
        Boxes are clipped to the frame. Class and width filters precede
        ``max_det``. The model supplies boxes only; pose and masks are not exposed.
        """
        classes = self.cfg.person_class_id if classes is None else classes
        boxes, scores = self._detections(img, classes, min_width_filter)
        xywh = [to_xywh(box) for box in boxes]
        points = [findRepspoint(box, self.cfg.repspoint_calibration) for box in boxes]
        if visual:
            for box, point in zip(boxes, points):
                self._draw(img, box, point)
        return img, xywh, list(boxes), points, [float(score) for score in scores]

    def detectPeople(self, img, visual=True, min_width_filter=15,
                     alt_repspoint=False, alt_repspoint_top=True):
        """Detect people using the existing box-based Person representation.

        Parameters
        ----------
        img : ``numpy.ndarray``
            Nonempty uint8 BGR frame. The same array is returned.
        visual : bool
            Defaults to True. Draw boxes/points in place when enabled.
        min_width_filter : int
            Inclusive minimum width in original-image pixels; defaults to 15.
        alt_repspoint : bool
            Defaults to False. Use the box midpoint helper instead of calibration.
        alt_repspoint_top : bool
            Defaults to True. Prefer the top box midpoint with ``alt_repspoint``.

        Returns
        -------
        list[Person]
            People sorted by confidence, with IDs starting from zero each frame.
            Trackers assign persistent IDs. Keypoints remain empty.
        ``numpy.ndarray``
            Original frame, modified only when visual is True.
        """
        boxes, scores = self._detections(img, self.cfg.person_class_id, min_width_filter)
        people = []
        for index, (box, score) in enumerate(zip(boxes, scores)):
            point = (findRepspointBB(box, prefer_top=alt_repspoint_top) if alt_repspoint
                     else findRepspoint(box, self.cfg.repspoint_calibration))
            people.append(Person(index, index, box_xywh=to_xywh(box), box_xyxy=box,
                                 repspoint=point, det_conf=float(score)))
            if visual:
                self._draw(img, box, point)
        return people, img
