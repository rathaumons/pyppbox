#################################################################################
# Prepare for CPU-Only -> Overwrite some parameters of internal config
#################################################################################

# Hard overwrite internal config for YOLO Ultralytics -> CPU mode
from pyppbox.config.myconfig import MyConfigurator
internal_configs = MyConfigurator()
internal_configs.setAllDCFG()
yolo_utlt_doc={
    'dt_name': 'YOLO_Ultralytics', 
    'conf': 0.5, 
    'iou': 0.7, 
    'imgsz': 416, 
    'show_boxes': False, 
    'device': 'cpu', # <- Switch to CPU
    'max_det': 100, 
    'line_width': 500, 
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
