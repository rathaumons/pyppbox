#################################################################################
# Example of using a module by setting string/dict of YAML/JSON
#################################################################################

import cv2

from pyppbox.standalone import (setMainDetector, setMainTracker, setMainReIDer, 
                                detectPeople, trackPeople, reidPeople)
from pyppbox.utils.visualizetools import visualizePeople


# Similar to example 04, using a custom YAML/JSON raw string or ready dictionary 
# allows you to set or adjust the parameters of a specific module directly in codes.

# Use a raw YAML/JSON string
"""
mydetector="[{'dt_name': 'YOLO_Ultralytics', 'conf': 0.5, 'iou': 0.7, \
    'imgsz': 416, 'boxes': True, 'device': 0, \
    'max_det': 100, 'line_width': 500, 'model_file': \
    'C:/pyppbox_v3/data/modules/yolo_ultralytics/yolov8l-pose.pt', \
    'repspoint_calibration': 0.25}]"
mytracker="[{'tk_name': 'SORT', 'max_age': 1, 'min_hits': 3, \
    'iou_threshold': 0.3}]"
myreider="[{'ri_name': 'Torchreid', 'classifier_pkl': \
    'C:/pyppbox_v3/data/modules/torchreid/classifier/gta5_osnet_ain_ms_d_c.pkl', \
    'train_data': 'C:/pyppbox_v3/data/datasets/GTA_V_DATASET/body_128x256', \
    'model_name': 'osnet_ain_x1_0', \
    'model_path': 'C:/pyppbox_v3/data/modules/torchreid/models/torchreid/\
    osnet_ain_ms_d_c.pth.tar', 'min_confidence': 0.35}]"
"""

# Use a ready YAML/JSON dictionary of a detector
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

# Use a ready YAML/JSON dictionary of a tracker
mytracker={
    'tk_name': 'SORT', 
    'max_age': 1, 
    'min_hits': 3, 
    'iou_threshold': 0.3
}

# Use a ready YAML/JSON dictionary of a reider
myreider={
    'ri_name': 'Torchreid', 
    'classifier_pkl': 'C:/pyppbox_v3/data/modules/torchreid/classifier/gta5_osnet_ain_ms_d_c.pkl', 
    'train_data': 'C:/pyppbox_v3/data/datasets/GTA_V_DATASET/body_128x256', 
    'model_name': 'osnet_ain_x1_0', 
    'model_path': 'C:/pyppbox_v3/data/modules/torchreid/models/torchreid/osnet_ain_ms_d_c.pth.tar', 
    'min_confidence': 0.35
}

setMainDetector(detector=mydetector)
setMainTracker(tracker=mytracker)
setMainReIDer(reider=myreider)

input_video = "data/gta.mp4"
cap = cv2.VideoCapture(input_video)

while cap.isOpened():
    hasFrame, frame = cap.read()

    if hasFrame:

        # Detect people without visualizing
        detected_people, _ = detectPeople(frame, img_is_mat=True, visual=False)

        # Track the detected people
        tracked_people = trackPeople(frame, detected_people, img_is_mat=True)

        # Re-identify the tracked people
        reidentified_people, reid_count = reidPeople(
            frame, 
            tracked_people, 
            img_is_mat=True
        )

        # Visualize people in video frame with reid status `show_reid=reid_count`
        visualized_mat = visualizePeople(
            frame, 
            reidentified_people, 
            show_reid=reid_count
        )
        cv2.imshow("pyppbox: example_05_module_json_str_dict.py", visualized_mat)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

