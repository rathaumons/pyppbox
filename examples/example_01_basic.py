#################################################################################
# Example of using `detectPeople()`, `reidPeople()`, and `visualizePeople()`
#################################################################################

import cv2

from pyppbox.standalone import (setMainDetector, detectPeople, 
                                setMainReIDer, reidPeople)
from pyppbox.utils.visualizetools import visualizePeople


image = "data/gta.jpg"

# Detect people and save as visualized image detection_output.jpg using Ultralytics
setMainDetector("YOLO_Ultralytics") # Set by name using internal configs
detected_people, visual_image = detectPeople(
    img=image, # Give an image to detect people
    visual=True, # Set True to visualize the detected people
    save=True, # Set True to save the visualized image
    save_file="detection_output.jpg" # Give a path to save the visualized image
)

# Re-identify the detected people using FaceNet
setMainReIDer("FaceNet") # Set by name using internal configs
reidentified_people, reid_count = reidPeople(img=image, people=detected_people)

# Visual people using pyppbox's visualizetools
visualized_mat = visualizePeople(
    img=image, # Give an image to visualize
    img_is_mat=False, # Set False as the given image is not cv Mat
    people=reidentified_people, # Set the re-identified people
    show_box=True, # Visualize bounding boxes
    show_skl=(True,True,5), # Visualize keypoints, skeletons, and skeleton size
    show_ids=(True,True,True), # Visualize cid, faceid, deepid
    show_repspoint=True, # Visualize represented point
)
cv2.imshow("pyppbox: example_01_basic.py", visualized_mat)
cv2.waitKey(5000) # Show visualized_mat for 5 seconds

# Save the visualized_mat as reidentification_output.jpg
cv2.imwrite("reidentification_output.jpg", visualized_mat)

