#################################################################################
# Example of training classifier of a reider -> `trainReIDClassifier()`
#################################################################################

from pyppbox.standalone import trainReIDClassifier


# Reider
myreider={
    'ri_name': 'Torchreid', 
    'classifier_pkl': 'C:/pyppbox_v3/data/modules/torchreid/classifier/gta5_osnet_ain_ms_d_c.pkl', 
    'train_data': 'C:/pyppbox_v3/data/datasets/GTA_V_DATASET/body_128x256', 
    'model_name': 'osnet_ain_x1_0', 
    'model_path': 'C:/pyppbox_v3/data/modules/torchreid/models/torchreid/osnet_ain_ms_d_c.pth.tar', 
    'min_confidence': 0.35,
    'device': 'cuda'
}

trainReIDClassifier(
    reider=myreider, 
    train_data="", # Set train_data="" means using the default 'train_data' in line 12
    classifier_pkl="data/new_gta_osnet_ain_ms_d_c.pkl" # Set classifier_pkl="" to use the default in line 11
)

# The simplest way with internal config supposing everything is set the way you want, 
# then you can simply import the `trainReIDClassifier` and call it directly; for example:
#
# >>> from pyppbox.standalone import trainReIDClassifier
# >>> trainReIDClassifier()

