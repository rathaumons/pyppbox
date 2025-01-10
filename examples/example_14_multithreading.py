#################################################################################
# Example of using a pyppbox in multithreading
#################################################################################

import cv2
import threading
from pyppbox.utils.visualizetools import visualizePeople
from pyppbox.standalone import MT

def ppb_task(input, main_configs, name="Task"):
    ppbmt = MT() # Use `MT` for multithreading
    ppbmt.setMainModules(main_yaml=main_configs)
    cap = cv2.VideoCapture(input)
    while cap.isOpened():
        hasFrame, frame = cap.read()
        if hasFrame:
            detected_people, _ = ppbmt.detectPeople(frame, img_is_mat=True, visual=False)
            tracked_people = ppbmt.trackPeople(frame, detected_people, img_is_mat=True)
            reidentified_people, reid_count = ppbmt.reidPeople(
                frame,
                tracked_people,
                img_is_mat=True
            )
            visualized_mat = visualizePeople(
                frame,
                reidentified_people,
                show_reid=reid_count
            )
            cv2.imshow("pyppbox: example_13_multithreading.py (" + name + ")", visualized_mat)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()

if __name__ == '__main__':
    input_one = "data/gta.mp4"
    input_two = "data/gta.mp4"
    main_configs_one = {'detector': 'YOLO_Classic',
                        'tracker': 'SORT',
                        'reider': 'Torchreid'}
    main_configs_two = {'detector': 'YOLO_Classic',
                        'tracker': 'Centroid',
                        'reider': 'FaceNet'}
    thread_one = threading.Thread(target=ppb_task, args=(input_one, main_configs_one, "Task 1"))
    thread_two = threading.Thread(target=ppb_task, args=(input_two, main_configs_two, "Task 2"))
    thread_one.start()
    thread_two.start()
    thread_one.join()
    thread_two.join()
