#################################################################################
# Test 04: Detector GT (CPU-Only)
#################################################################################

import cv2

from pyppbox.standalone import (setMainDetector, setMainTracker, setMainReIDer, 
                                detectPeople, trackPeople, reidPeople)
from pyppbox.utils.visualizetools import visualizePeople


mydetector={
    'dt_name': 'GT', # Here, 'dt_name' must be 'GT' -> Check `gttools` for more
    'gt_file': '../examples/data/gta.mp4.txt', # Set path of ground-truth text file
    'gt_map_file': '../examples/data/gt_map.txt' # Set path of Video:GT mapping text file
}

setMainDetector(detector=mydetector)
setMainTracker(tracker="SORT")
setMainReIDer(reider="FaceNet")

# Set tracker=None and reider=None if you want to run full GT mode including 
# the real ID as in the GT file.
"""
setTracker(tracker="None")  # FYI, "None" is not None
setReIDer(reider="None")    # FYI, "None" is not None
"""

input_video = "../examples/data/gta.mp4"
cap = cv2.VideoCapture(input_video)

frame_index = 0

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
        
        # Save the visualized frame
        cv2.imwrite("test_04/frame_" + str(frame_index) + ".jpg", visualized_mat)
        frame_index += 1

    else:
        break
cap.release()

