#################################################################################
# Example of using the internal config directory of pyppbox package
#################################################################################

import cv2

from pyppbox.standalone import setConfigDir, detectPeople, trackPeople, reidPeople
from pyppbox.utils.visualizetools import visualizePeople


# The config directory is a directory where consists of 4 required YAML files:
#   - main.yaml, indicates which detector/tracker/reider is used.
#   - detectors.yaml, stores all detectors' configurations.
#   - trackers.yaml, stores all trackers' configurations.
#   - reiders.yaml, stores all reiders' configurations.
#
# The internal config directory of pyppbox is '{pyppbox root}/config/cfg'.

# Use the internal config directory of pyppbox
setConfigDir(config_dir=None, load_all=True)

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
        cv2.imshow("pyppbox: example_02_internal_cfg_directory.py", visualized_mat)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

