#################################################################################
# Example of using pyppbox's restools
#################################################################################

import cv2

from pyppbox.standalone import setMainModules, detectPeople, trackPeople, reidPeople
from pyppbox.utils.visualizetools import visualizePeople
from pyppbox.utils.restools import ResIO # Import `ResIO`


# Set the main modules using the internal configurations of pyppbox
main_configurations = {'detector': 'YOLO_Classic', 
                       'tracker': 'SORT', 
                       'reider': 'Torchreid'}
setMainModules(main_yaml=main_configurations)

# Use ResIO
resIO = ResIO()
frame_index = 0 # ResIO requires precise frame index

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
        cv2.imshow("pyppbox: example_07_restools.py", visualized_mat)

        # Add `reidentified_people` to ResIO
        resIO.addPeople(frame_index, reidentified_people)
        frame_index += 1 # Increment the frame index

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

# ResIO -> Dump result
dump_dir = "" # Current dir
resIO.dumpAll(dump_dir=dump_dir, dump_mode=3)

