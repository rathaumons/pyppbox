#################################################################################
# Example of retraining all supported ReID backends' GTA V identity classifiers
#################################################################################

import os
import pyppbox
from pyppbox.standalone import close, trainReIDClassifier


pyppbox_root = os.path.dirname(pyppbox.__file__)

gta5_osnet_ain_config = {
    'ri_name': 'Torchreid', 
    'classifier_pkl': f'{pyppbox_root}/data/modules/torchreid/classifier/gta5_osnet_ain_ms_d_c.pkl', 
    'train_data': f'{pyppbox_root}/data/datasets/GTA_V_DATASET/body_128x256', 
    'model_name': 'osnet_ain_x1_0', 
    'model_path': f'{pyppbox_root}/data/modules/torchreid/models/torchreid/osnet_ain_ms_d_c.pth.tar', 
    'min_confidence': 0.35, 
    'device': 'cuda'
}

gta5_osnet_config = {
    'ri_name': 'Torchreid', 
    'classifier_pkl': f'{pyppbox_root}/data/modules/torchreid/classifier/gta5_osnet_ms_d_c.pkl', 
    'train_data': f'{pyppbox_root}/data/datasets/GTA_V_DATASET/body_128x256', 
    'model_name': 'osnet_x1_0', 
    'model_path': f'{pyppbox_root}/data/modules/torchreid/models/torchreid/osnet_ms_d_c.pth.tar', 
    'min_confidence': 0.35, 
    'device': 'cuda'
}

gta5_facenet_config = {
    'ri_name': 'FaceNet', 
    'gpu_mem': 0.585, 
    'model_det': f'{pyppbox_root}/data/modules/facenet/models/det', 
    'model_file': f'{pyppbox_root}/data/modules/facenet/models/20180402-114759/20180402-114759.pb', 
    'classifier_pkl': f'{pyppbox_root}/data/modules/facenet/classifier/gta5.pkl', 
    'train_data': f'{pyppbox_root}/data/datasets/GTA_V_DATASET/face_182x182', 
    'batch_size': 1000, 
    'min_confidence': 0.75, 
    'yl_h_calibration': [-125, 75], 
    'yl_w_calibration': [-55, 55]
}

# ArcFace/AdaFace use their default preprocessing for the packaged ONNX models.
# For NVIDIA GPU inference, install requirements/faces_gpu.txt and use 'cuda'.
gta5_arcface_config = {
    'ri_name': 'ArcFace',
    'model_det': f'{pyppbox_root}/data/modules/arcface/models/det_10g.onnx',
    'model_file': f'{pyppbox_root}/data/modules/arcface/models/w600k_r50.onnx',
    'classifier_pkl': f'{pyppbox_root}/data/modules/arcface/classifier/gta5.pkl',
    'train_data': f'{pyppbox_root}/data/datasets/GTA_V_DATASET/face_182x182',
    'device': 'cpu'
}

gta5_adaface_config = {
    'ri_name': 'AdaFace',
    'model_det': f'{pyppbox_root}/data/modules/adaface/models/det_10g.onnx',
    'model_file': f'{pyppbox_root}/data/modules/adaface/models/adaface_ir18_vgg2.onnx',
    'classifier_pkl': f'{pyppbox_root}/data/modules/adaface/classifier/gta5.pkl',
    'train_data': f'{pyppbox_root}/data/datasets/GTA_V_DATASET/face_182x182',
    'device': 'cpu'
}


if __name__ == '__main__':
    # Remove entries to train only selected classifiers. Each output is overwritten;
    # pretrained neural-network weights remain unchanged.
    for reider in (gta5_osnet_ain_config, gta5_osnet_config, gta5_facenet_config,
                   gta5_arcface_config, gta5_adaface_config):
        try:
            trainReIDClassifier(reider=reider)
        finally:
            close()

