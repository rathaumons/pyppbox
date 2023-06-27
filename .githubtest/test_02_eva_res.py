#################################################################################
# Test 02: `MyEVA` and `ResIO` (CPU-Only)
#################################################################################


import cv2

from pyppbox.standalone import setMainModules, detectPeople, trackPeople, reidPeople
from pyppbox.utils.visualizetools import visualizePeople
from pyppbox.utils.evatools import MyEVA # Import `MyEVA`
from pyppbox.utils.restools import ResIO # Import `ResIO`


# Set the main modules using the internal configurations of pyppbox
main_configurations = {'detector': 'YOLO_Classic', 
                       'tracker': 'Centroid', 
                       'reider': 'FaceNet'}
setMainModules(main_yaml=main_configurations) 

# Use MyEVA
eva = MyEVA()

# The "../examples/gta.mp4.txt" is ground-truth for "gta.mp4"
gt_file = "../examples/data/gta.mp4.txt" 

# id_mode="faceid"
eva.setGTByKnownGTFile(gt_file=gt_file, id_mode="faceid") 

# (Optional) MyEVA requires you to help count reid from `reidPeople()`
total_reid = 0 

input_video = "../examples/data/gta.mp4"
cap = cv2.VideoCapture(input_video)

# Use ResIO
resIO = ResIO()
frame_index = 0 # ResIO requires precise frame index

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

        # Count reid
        total_reid = total_reid + reid_count[0] + reid_count[1]

        # Visualize people in video frame with reid status `show_reid=reid_count`
        visualized_mat = visualizePeople(
            frame, 
            reidentified_people, 
            show_reid=reid_count
        )

        # MyEVA -> Validate the `reidentified_people`
        eva.validate(reidentified_people)

        # Add `reidentified_people` to ResIO
        resIO.addPeople(frame_index, reidentified_people)
        frame_index += 1 # Increment the frame index

        # Save the visualized frame
        cv2.imwrite("test_02/frame_" + str(frame_index) + ".jpg", visualized_mat)
        frame_index += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

# MyEVA -> Get and print evaluation summary
eva.setReIDcount(total_reid)
wrong_id, missed_det, fault_det, reid_count, total_det, score = eva.getSummary(print_summary=True)

# ResIO -> Dump result
resIO.dumpAll(dump_dir="test_02", dump_mode=3)

