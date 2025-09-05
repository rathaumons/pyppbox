#################################################################################
# Example of using modules' YAML files
#################################################################################

import cv2

from pyppbox.standalone import (setMainDetector, setMainTracker, setMainReIDer, 
                                detectPeople, trackPeople, reidPeople)
from pyppbox.utils.visualizetools import visualizePeople


# Using a custom YAML/JSON file allows you to set or adjust the parameters 
# of a specific module easily without changing your code.

# This YAML/JSON file contains only the configurations of a supported detector
mydetector = "single_config/mydetector.yaml"
# mydetector = "single_config/mydetector.json"

# This YAML/JSON file contains only the configurations of a supported tracker
# mytracker = "single_config/mytracker.yaml"
mytracker = "single_config/mytracker.json"

# This YAML/JSON file contains only the configurations of a supported reider
# myreider = "single_config/myreider.yaml"
myreider = "single_config/myreider.json"

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
        cv2.imshow("pyppbox: example_04_module_yaml_file.py", visualized_mat)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

