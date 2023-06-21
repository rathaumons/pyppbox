#################################################################################
# Example of using GT (Ground-truth) text file as a detector
#################################################################################

import cv2

from pyppbox.standalone import (setMainDetector, setMainTracker, setMainReIDer, 
                                detectPeople, trackPeople, reidPeople)
from pyppbox.utils.visualizetools import visualizePeople


mydetector={
    'dt_name': 'GT', # Here, 'dt_name' must be 'GT' -> Check `gttools` for more
    'gt_file': 'data/gta.mp4.txt', # Set path of ground-truth text file
    'gt_map_file': 'data/gt_map.txt' # Set path of Video=GT mapping text file
}

setMainDetector(detector=mydetector)
setMainTracker(tracker="SORT")
setMainReIDer(reider="Torchreid")

# Set tracker=None and reider=None if you want to run full GT mode including 
# the real ID as in the GT file.
"""
setTracker(tracker="None")  # FYI, "None" is not None
setReIDer(reider="None")    # FYI, "None" is not None
"""

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
        cv2.imshow("pyppbox: example_10_detector_gt_file.py", visualized_mat)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

