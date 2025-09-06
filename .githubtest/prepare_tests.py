#################################################################################
# Prepare for CPU-Only -> Override some parameters of internal config
#################################################################################

# Hard override internal config for YOLO Ultralytics -> CPU mode
from pyppbox.config.myconfig import MyConfigurator
internal_configs = MyConfigurator()
internal_configs.setAllDCFG()
yolo_utlt_doc={
    'dt_name': 'YOLO_Ultralytics', 
    'conf': 0.4, 
    'iou': 0.7, 
    'imgsz': 1024, 
    'show_boxes': False, 
    'device': 'cpu', # <- Switch to CPU
    'max_det': 100, 
    'model_file': 'data/modules/yolo_ultralytics/yolov8n-pose.pt', # yolov8n-pose.pt
    'repspoint_calibration': 0.25
}
internal_configs.dumpAllDCFG(
    [
        internal_configs.dcfg_yolocs.getDocument(), 
        yolo_utlt_doc, 
        internal_configs.dcfg_gt.getDocument()
    ]
)

# Directories for test result
import os
test_result = "test_0"
for i in range(1, 5):
    os.makedirs(test_result + str(i))
