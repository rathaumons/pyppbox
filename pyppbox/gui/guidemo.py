# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                           #
#   pyppbox: Toolbox for people detecting, tracking, and re-identifying.    #
#   Copyright (C) 2022 UMONS-Numediart                                      #
#                                                                           #
#   This program is free software: you can redistribute it and/or modify    #
#   it under the terms of the GNU General Public License as published by    #
#   the Free Software Foundation, either version 3 of the License, or       #
#   (at your option) any later version.                                     #
#                                                                           #
#   This program is distributed in the hope that it will be useful,         #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#   GNU General Public License for more details.                            #
#                                                                           #
#   You should have received a copy of the GNU General Public License       #
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.  #
#                                                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import cv2
import imutils.video

from timeit import time
from pyppbox.gui.guihub import loadUITMP, loadInputTMP
from pyppbox.utils.evatools import MyEVA
from pyppbox.utils.restools import ResIO
from pyppbox.utils.visualizetools import visualizePeople
from pyppbox.utils.commontools import (joinFPathFull, getGlobalRootDir, 
                                       getVersionString, getFileName)
from pyppbox.standalone import (setConfigDir, detectPeople, trackPeople, 
                                reidPeople, getConfig)


# Get config_dir from ui.tmp file
config_dir = loadUITMP()

# Set CFG according to config_dir
setConfigDir(config_dir=config_dir, load_all=True)

# Load input from input.tmp
input_source, force_hd = loadInputTMP()
print("---GUIDEMO : Input video <- " + getFileName(input_source))

# Use MyEVA (Only supports faceid and deepid on GTA V dataset)
eva = MyEVA()

# Find gt_map_txt for setGTByGTMap() of MyEVA
cfg = getConfig()
cfg.setGTCFG()
gt_map_txt = cfg.dcfg_gt.gt_map_file

# Find id_mode for setGTByGTMap() of MyEVA
id_mode = "deepid"
if cfg.mcfg.reider.lower() == cfg.unified_strings.facenet:
    id_mode = "faceid"

eva.setGTByGTMap(gt_map_txt=gt_map_txt, input_video=input_source, id_mode=id_mode)

# Use ResIO
resIO = ResIO()

# Title
title = "(PYPPBOX v" + getVersionString() + ") DEMO:"
title = title + " DT=" + cfg.mcfg.detector
title = title + ", TK=" + cfg.mcfg.tracker
title = title + ", RI=" + cfg.mcfg.reider

# Count ReID
total_reid = 0

cap = cv2.VideoCapture(joinFPathFull(getGlobalRootDir(), input_source))
cap_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fps = 0.0
fps_imutils = imutils.video.FPS().start()

# Need frame_index for display
frame_index = 0

while cap.isOpened():
    t1 = time.time()
    hasFrame, frame = cap.read()
    if hasFrame:
        # Resize frame if force HD
        if force_hd: 
            frame_resolution = (1280, int((1280/cap_width) * cap_height))
            frame = cv2.resize(frame, frame_resolution)
        # Detect people
        people, _ = detectPeople(frame, img_is_mat=True)
        # Track people
        people = trackPeople(frame, people, img_is_mat=True)
        # ReID people
        people, reid_count = reidPeople(
            frame, 
            people, 
            deduplicate=True, 
            img_is_mat=True
        )
        total_reid = total_reid + reid_count[0] + reid_count[1]
        # Add people to ResIO
        resIO.addPeople(frame_index, people)
        # MyEVA -> Validate people
        eva.validate(people)
        # Visualize people
        visual_frame = visualizePeople(
            frame, people, 
            show_reid=reid_count, 
            img_is_mat=True
        )
        # Add framerate & info
        fps_imutils.update()
        fps = (fps + (1./((1.000000000001*time.time())-t1))) / 2
        cv2.putText(visual_frame, str(int(fps)) + " | " + str(frame_index), (15, 30), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Show
        cv2.imshow(title, visual_frame)

        frame_index += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

# MyEVA 
eva.setReIDcount(total_reid)
wrong_id, missed_det, fault_det, reid_count, total_det, score = eva.getSummary(print_summary=True)

# ResIO
import os
res_dir = os.path.expanduser('~/Documents/pyppbox')
if not os.path.exists(res_dir): os.makedirs(res_dir)
resIO.dumpAll(res_dir, dump_mode=3)
