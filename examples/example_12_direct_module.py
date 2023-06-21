#################################################################################
# Example of using a module directly
#################################################################################

import cv2

# Import a module you need
from pyppbox.modules.detectors.yoloult import MyYOLOULT

# Import the corresponding configurator
from pyppbox.config.myconfig import DCFGYOLOULT

# Use a YAML/JSON dictionary
mydetector={
    'dt_name': 'YOLO_Ultralytics', 
    'conf': 0.5, 
    'iou': 0.7, 
    'imgsz': 416, 
    'boxes': True, 
    'device': 0, 
    'max_det': 100, 
    'line_width': 500, 
    'model_file': 'C:/pyppbox_v3/data/modules/yolo_ultralytics/yolov8l-pose.pt', 
    'repspoint_calibration': 0.25
}

yoloult_cfg = DCFGYOLOULT()
yoloult_cfg.set(mydetector) # Supports all YAML/JSON raw string, ready dictionary, and file.

detector = MyYOLOULT(yoloult_cfg)

input_video = "data/gta.mp4"
cap = cv2.VideoCapture(input_video)

while cap.isOpened():
    hasFrame, frame = cap.read()

    if hasFrame:

        # Detect people
        (visual_frame, 
         box_xywh_list, 
         box_xyxy_list, 
         respesented_point_list, 
         body_keypoint_list, 
         confidence_list) = detector.detect(frame, visual=True, classes=0)

        cv2.imshow("pyppbox: example_12_direct_module.py", visual_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
