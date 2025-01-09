#################################################################################
# Example of using a supported module by setting its name
#################################################################################

import cv2

from pyppbox.standalone import (setMainDetector, setMainTracker, setMainReIDer, 
                                setMainModules, detectPeople, trackPeople, reidPeople)
from pyppbox.utils.visualizetools import visualizePeople


# As in example 04 & 05, `setMainDetector()`, `setMainTracker()`, and aslo 
# `setMainReIDer()` allow you to easily set a detector, a tracker and a reider 
# by a raw string, a ready dictionary, or a YAML/JSON file without having to
# call the `setConfigDir()` in advance. 
# 
# Different from setting a dictionary and a YAML/JSON file, setting a supported 
# module by its name still relies on the config directory. These methods load
# the configurations from the config directory set by last `setConfigDir()`, and 
# if `setConfigDir()` has not been called before, they will reference the internal 
# config directory in order to  load the corresponding configurations. If you 
# wish to change from referencing the internal configurations to your custom 
# config directory like "cfg", you can, in advance, call 
# `setConfigDir(config_dir="cfg", load_all=False)` .

"""
setMainDetector(detector="YOLO_Classic")
setMainTracker(tracker="SORT")
setMainReIDer(reider="Torchreid")
"""

# Instead of calling `setMainDetector()`, `setMainTracker()`, and aslo 
# `setMainReIDer()`, you can call `setMainModules()` to override the default 
# configurations and set and load the ones you need.
main_configurations = {'detector': 'YOLO_Classic', 
                       'tracker': 'SORT', 
                       'reider': 'Torchreid'}
setMainModules(main_yaml=main_configurations)


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

        # Visualize people in video frame, with reid status `show_reid=reid_count`
        visualized_mat = visualizePeople(
            frame, 
            reidentified_people, 
            show_reid=reid_count
        )
        cv2.imshow("pyppbox: example_06_module_name.py", visualized_mat)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

