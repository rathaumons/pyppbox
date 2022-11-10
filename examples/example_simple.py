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


import cv2
import pyppbox
from pyppbox.ppboxmng import PManager

# set main config
main_config = {'detector': 'YOLO', 
               'tracker': 'Centroid', 
               'reider': 'DeepReID', 
               'input_video': 'C:/media/hard_sur_hd.mp4', 
               'force_hd': False}
pyppbox.setMainCFG(main_config)

# initial PManager
pmg = PManager()

# start video
cap = cv2.VideoCapture(pmg.getInputFile())
while cap.isOpened():
    hasFrame, frame = cap.read()
    if hasFrame:

        # update detecter
        ppobl = pmg.detectFramePPOBL(frame, True)

        # track
        pmg.updateTrackerPPOBL(ppobl)

        # reid normal + duplicate ID killer
        pmg.reidNormal()
        pmg.reidDupkiller()

        # display info
        updated_pp = pmg.getCurrentPPOBL()
        for person in updated_pp:
            (x, y) = person.getRepspoint()
            cv2.putText(frame, str(person.getCid()), (int(x - 10), int(y - 60)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, str(person.getDeepid()), (int(x - 85), int(y - 90)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
            cv2.putText(frame, str(person.getFaceid()), (int(x - 85), int(y - 125)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("pyppbox Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
