# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Backend-free validation of the existing configuration fields."""
import math
import os
import re


_FIELDS = {
    'MainCFG': 'detector tracker reider',
    'DCFGYOLOCLS': 'dt_name nms conf class_file model_cfg_file model_weights model_image_size repspoint_calibration',
    'DCFGYOLOULT': 'dt_name conf iou imgsz show_boxes device max_det model_file repspoint_calibration',
    'DCFGGT': 'dt_name gt_file gt_map_file',
    'TCFGCentroid': 'tk_name max_spread',
    'TCFGSORT': 'tk_name max_age min_hits iou_threshold',
    'TCFGDeepSORT': 'tk_name nn_budget nms_max_overlap max_cosine_distance model_file',
    'RCFGFaceNet': 'ri_name gpu_mem model_det model_file classifier_pkl train_data batch_size min_confidence yl_h_calibration yl_w_calibration',
    'RCFGTorchreid': 'ri_name classifier_pkl train_data model_name model_path min_confidence device',
}
_PATHS = {'class_file', 'model_cfg_file', 'model_weights', 'model_file', 'gt_file',
          'gt_map_file', 'model_det', 'classifier_pkl', 'train_data', 'model_path'}
_MODULE_NAMES = dict(DCFGYOLOCLS='YOLO_Classic', DCFGYOLOULT='YOLO_Ultralytics', DCFGGT='GT',
                     TCFGCentroid='Centroid', TCFGSORT='SORT', TCFGDeepSORT='DeepSORT',
                     RCFGFaceNet='FaceNet', RCFGTorchreid='Torchreid')


def _number(value, field, minimum=None, maximum=None, integer=False):
    if integer and not isinstance(value, bool) and (
            isinstance(value, int) or isinstance(value, str) and re.fullmatch(r'[+-]?\d+', value.strip())):
        number = int(value)
        if (minimum is None or number >= minimum) and (maximum is None or number <= maximum):
            return number
    try:
        number = float(value)
        valid = not isinstance(value, bool) and math.isfinite(number)
        valid &= minimum is None or number >= minimum
        valid &= maximum is None or number <= maximum
        valid &= not integer or number.is_integer()
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        bounds = f' >= {minimum}' if minimum is not None else ''
        bounds += f' and <= {maximum}' if maximum is not None else ''
        raise ValueError(f'{field} must be a finite {"integer" if integer else "number"}{bounds}.')
    return int(number) if integer else value if isinstance(value, (int, float)) else number


def _device(value, torchreid):
    if not torchreid and (value is None or value == ''):
        return value
    if isinstance(value, bool):
        raise ValueError('device must be a device name or GPU index, not a boolean.')
    if not torchreid and isinstance(value, (list, tuple)):
        if not value:
            raise ValueError('device GPU index list must not be empty.')
        return [_number(v, 'device', -1, integer=True) for v in value]
    text = str(value).strip().lower()
    if not torchreid and text == 'none':
        return None
    if re.fullmatch(r'\d+', text):
        return 'cuda:' + text if torchreid else int(text)
    pattern = r'(cpu|mps|cuda|xpu|npu)(:\d+)?'
    if not torchreid:
        pattern = r'(cpu|mps|cuda|xpu|npu)(:(\d+|\d+(,\d+)+))?|(\d+|-1)(,(\d+|-1))*|tpu\w*|intel\w*|vulkan\w*'
    if not re.fullmatch(pattern, text):
        raise ValueError('device must be cpu, cuda, cuda:N, mps, or a supported GPU index/device.')
    return text


def validate_config(class_name, document, unified_strings):
    """Return normalized known fields; retain extensions without importing backends."""
    if class_name not in _FIELDS or not document:
        return document
    fields = _FIELDS[class_name].split()
    missing = [key for key in fields if key not in document]
    if missing:
        raise ValueError(f'{class_name}: missing required settings: {", ".join(missing)}.')
    result = dict(document)
    if class_name in _MODULE_NAMES:
        key = fields[0]
        if unified_strings.getUnifiedFormat(result[key]) != _MODULE_NAMES[class_name]:
            raise ValueError(f'{class_name}: {key} must select {_MODULE_NAMES[class_name]}.')
    for field in fields:
        value = result[field]
        if field in ('detector', 'tracker', 'reider', 'dt_name', 'tk_name', 'ri_name', 'model_name'):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{class_name}: {field} must be a nonempty string.')
        elif field in _PATHS:
            path = os.fspath(value) if isinstance(value, (str, os.PathLike)) else None
            if not isinstance(path, str) or '\0' in path:
                raise ValueError(f'{class_name}: {field} must be a path string.')
            result[field] = path
        elif field in ('conf', 'iou', 'nms', 'iou_threshold', 'nms_max_overlap', 'min_confidence', 'gpu_mem'):
            result[field] = _number(value, field, 0, 1)
        elif field in ('max_age', 'min_hits', 'nn_budget'):
            result[field] = None if field == 'nn_budget' and value is None else _number(value, field, 0, integer=True)
        elif field in ('imgsz', 'model_image_size', 'max_det', 'batch_size'):
            result[field] = _number(value, field, 1, integer=True)
        elif field in ('repspoint_calibration', 'max_cosine_distance', 'max_spread'):
            # Representative-point offsets may be signed; cosine distances can exceed one.
            result[field] = _number(value, field, None if field == 'repspoint_calibration' else 0)
        elif field in ('yl_h_calibration', 'yl_w_calibration'):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f'{field} must contain two finite signed offsets.')
            result[field] = [_number(v, field) for v in value]
        elif field == 'device':
            result[field] = _device(value, class_name == 'RCFGTorchreid')
        elif field == 'show_boxes':
            if isinstance(value, str) and value.strip().lower() in ('true', 'false'):
                value = value.strip().lower() == 'true'
            if not isinstance(value, bool):
                raise ValueError("show_boxes must be a boolean or the string 'True'/'False'.")
            result[field] = value
    return result


def validate_runtime(config, classifier=False):
    """Check only assets used for inference; configuration inspection needs no weights."""
    name = type(config).__name__
    fields = {
        'DCFGYOLOCLS': ('class_file', 'model_cfg_file', 'model_weights'),
        'DCFGGT': ('gt_file',), 'TCFGDeepSORT': ('model_file',),
        'RCFGTorchreid': ('model_path',),
        'RCFGFaceNet': ('model_file', 'model_det'),
    }.get(name, ())
    # FaceNet defers its models until classifier loading; training also uses these models.
    if name == 'RCFGFaceNet' and not classifier:
        fields = ()
    if classifier and name in ('RCFGFaceNet', 'RCFGTorchreid'):
        fields += ('classifier_pkl',)
    for field in fields:
        path = getattr(config, field)
        is_directory = field == 'model_det' or (name == 'RCFGFaceNet' and field == 'model_file' and os.path.isdir(path))
        exists = os.path.isdir(path) if is_directory else os.path.isfile(path)
        if not exists:
            raise ValueError(f'{name}: {field} does not exist or has the wrong type: {path}')
        if field == 'model_det':
            for filename in ('det1.npy', 'det2.npy', 'det3.npy'):
                if not os.path.isfile(os.path.join(path, filename)):
                    raise ValueError(f'{name}: model_det is missing {filename}: {path}')
    if name == 'RCFGTorchreid' and any(size <= 0 for size in config.model_wh):
        raise ValueError(f'{name}: model_path has no input dimensions in the Torchreid model catalog: {config.model_path}')
