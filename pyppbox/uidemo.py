"""
    pyppbox: Toolbox for people detecting, tracking, and re-identifying.
    Copyright (C) 2022 UMONS-Numediart

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


from __future__ import division, print_function, absolute_import

import os
import sys
import cv2
import imutils.video

from timeit import time
from pyppbox.ppboxmng import PManager


try:

    # pmg = PManager()
    pmg = PManager(enableEval=True)
    
    input_source = pmg.getInputFile()
    print("Input video: " + str(input_source))
    
    cap = cv2.VideoCapture(input_source)
    cap_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # screenshot a frame for offline EVA
    screenshot = []
    screenshot_at = 60

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
    frame_id = 0

    while cap.isOpened():
        t1 = time.time()
        hasFrame, frame = cap.read()
        if hasFrame:

            # resize frame if force HD
            if pmg.forceHD(): frame = cv2.resize(frame, (1280, int((1280/cap_width) * cap_height)))

            # update detector
            ppobl = pmg.detectFramePPOBL(frame, True)

            # update tracker
            pmg.updateTrackerPPOBL(ppobl)

            # update reid 
            pmg.reidNormal() # level 1: normal
            pmg.reidDupkiller() # level 2: dupkiller

            # put info
            updated_pp = pmg.getCurrentPPOBL()
            det_frame = pmg.getFrameVisual()
            for person in updated_pp:
                (x, y) = person.getRepspoint()
                cv2.putText(det_frame, str(person.getCid()), (int(x - 10), int(y - 60)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(det_frame, str(person.getDeepid()), (int(x - 85), int(y - 90)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
                cv2.putText(det_frame, str(person.getFaceid()), (int(x - 85), int(y - 125)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                # Update offline EVA
                pmg.eva_io.add_person(frame_id, person)

            # Update realtime EVA frame by frame
            pmg.selfRealtimeEval()

            # Framerate & info
            fps_imutils.update()
            fps = (fps + (1./((1.000000000001*time.time())-t1))) / 2
            cv2.putText(det_frame, str(int(fps)) + " | " + str(frame_id), (15, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("pyppbox Demo", det_frame)

            # Take screenshot
            if screenshot_at == frame_id:
                screenshot = det_frame.copy()

            frame_id += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    
    cap.release()

    # Realtime EVA
    # Note: Realtime EVA only supports faceid and deepid on GTA V dataset
    pmg.getRealtimeEvalResult()

    # Offline EVA
    res_txt, extra_info = pmg.generateExtraInfoForOfflineEva()
    docs_path = os.path.expanduser('~/Documents/pyppbox')
    if not os.path.exists(docs_path): os.makedirs(docs_path)

    res_txt = os.path.join(docs_path, res_txt)
    pmg.eva_io.dump(res_txt)

    screenshot_jpg = res_txt[:-4] + "_info.jpg"
    cv2.imwrite(screenshot_jpg, screenshot)

    info_txt = res_txt[:-4] + "_info.txt"
    with open(info_txt, "a") as info_file:
        info_file.write(extra_info)

    print("\nMore info was saved to " + str(docs_path) + "\n")


except Exception as e:
    print(e)
    sys.exit(-1)

