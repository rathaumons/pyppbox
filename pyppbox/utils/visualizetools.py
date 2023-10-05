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
import numpy as np

from .persontools import Person
from .commontools import getCVMat
from .logtools import add_error_log, add_warning_log

# For ultralytics's skeleton
has_ultralytics = True
try:
    from ultralytics.utils.plotting import Colors
    colors = Colors()
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
                [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
except ImportError as e:
    has_ultralytics = False
    add_warning_log("visualizetools: ultralytics or pyppbox-ultralytics is not installed.")

# For cid
cid_col = (0, 0, 255)
cid_font_thickness = 2
cid_font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# For faceid
faceid_col = (0, 255, 255)
faceid_font_thickness = 2
faceid_footnote_text = "faceid"
faceid_font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# For deepid
deepid_col = (255, 0, 255)
deepid_font_thickness = 2
deepid_footnote_text = "deepid"
deepid_font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# For foot note
footnote_font_scale = 1
footnote_font_thickness = 2
footnote_font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# For reid
reid_pos = (125, 30)
reid_col = (0, 255, 255)
reid_dup_col = (0, 0, 255)
reid_status_font = cv2.FONT_HERSHEY_COMPLEX_SMALL


def __addSKL__(img, kpts, radius=5, kpt_line=True):
    # Ultralytics YOLO
    h, w, c = img.shape
    shape = (h, w)
    nkpt, ndim = kpts.shape
    is_pose = nkpt == 17 and ndim == 3
    kpt_line &= is_pose
    for i, k in enumerate(kpts):
        color_k = [int(x) for x in kpt_color[i]] if is_pose else colors(i)
        x_coord, y_coord = k[0], k[1]
        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
            if len(k) == 3:
                conf = k[2]
                if conf < 0.5: continue
            cv2.circle(img, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)
    if kpt_line:
        ndim = kpts.shape[-1]
        for i, sk in enumerate(skeleton):
            pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
            pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
            if ndim == 3:
                conf1 = kpts[(sk[0] - 1), 2]
                conf2 = kpts[(sk[1] - 1), 2]
                if conf1 < 0.5 or conf2 < 0.5: continue
            if (pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or 
                pos1[0] < 0 or pos1[1] < 0): continue
            if (pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or 
                pos2[0] < 0 or pos2[1] < 0): continue
            cv2.line(img, pos1, pos2, [int(x) for x in limb_color[i]], 
                     thickness=2, lineType=cv2.LINE_AA)
    return img

def visualizePeople(img, people, show_box=True, show_skl=(True,True,5), show_ids=(True,True,True), 
                    show_reid=(0,0), show_repspoint=True, img_is_mat=True):
    """Visualize people in the image by the given people. 

    Parameters
    ----------
    img : str or Mat
        An image file or a cv :obj:`Mat`.
    people : list[Person, ...]
        Set a list of :class:`Person` object found in the input :obj:`img`.
    show_box : bool, default=True
        Indicate whether to visualize bounding boxes.
    show_skl : tuple(bool, bool, int), default=(True,True,5)
        Set how to visualize the keypoints and skeletons. 
        Set the first element to :code:`True` to visualize the keypoints.
        Set the second element to :code:`True` to visualize the skeletons.
        The third element is the size of skeleton lines.
    show_ids : tuple(bool, bool, bool), default=(True,True,True)
        Set whether to visualize the IDs.
        Set the first element to :code:`True` to visualize the cid.
        Set the second element to :code:`True` to visualize the faceid.
        Set the third element to :code:`True` to visualize the deepid.
    show_reid : tuple(int, int), default=(0,0)
        :obj:`show_reid` is corresponding to :obj:`reid_count` in :func:`reidPeople()`
        Tuple of (ReID count, ReID deduplicate count)
    show_repspoint : bool, default=True
        Indicate whether to show the :obj:`repspoint` of :class:`Person` object.
    img_is_mat : bool, default=True
        Speed up the function by telling whether the :obj:`img` is cv :obj:`Mat`.
    
    Returns
    -------
    Mat
        A visualized cv :obj:`Mat`.
    """
    # Overwrite `img_is_mat` to False when `img` is a file.
    if img_is_mat and isinstance(img, str): img_is_mat = False
    if not img_is_mat: img = getCVMat(img)
    if isinstance(people, list):
        if len(people) > 0:
            if isinstance(people[0], Person):
                (h, w, c) = img.shape
                deepid_footnote_pos = (int(w - 90), h - 35)
                faceid_footnote_pos = (int(w - 90), h - 10)
                reid_pos = (int(w - 360), 30)
                cv2.putText(img, deepid_footnote_text, deepid_footnote_pos, footnote_font, 
                            footnote_font_scale, deepid_col, footnote_font_thickness)
                cv2.putText(img, faceid_footnote_text, faceid_footnote_pos, footnote_font, 
                            footnote_font_scale, faceid_col, footnote_font_thickness)
                for p in people:
                    (x, y) = p.repspoint
                    if show_box:
                        cv2.rectangle(img, (p.box_xyxy[0], p.box_xyxy[1]), 
                                      (p.box_xyxy[2], p.box_xyxy[3]), (255, 255, 0), 2)
                    if show_repspoint:
                        cv2.circle(img, (p.repspoint[0], p.repspoint[1]), radius=5, 
                                   color=(0, 0, 255), thickness=-1)
                    if (has_ultralytics and isinstance(show_skl, tuple) and np.asarray(show_skl).shape == (3,) 
                        and len(p.keypoints) >= 15):
                        (s, l, r) = show_skl
                        if s: img = __addSKL__(img, p.keypoints, radius=r, kpt_line=l)
                    if (isinstance(show_ids, tuple) and 
                        np.asarray(show_ids).shape == (3,)):
                        if show_ids[0]:
                            cv2.putText(img, str(p.cid), (x - 10, y - 65), cid_font, 
                                        1, cid_col, cid_font_thickness)
                        if show_ids[1]:
                            cv2.putText(img, str(p.deepid + " : " + str(int(p.deepid_conf)) + "%"), 
                                        (x - 90, y - 115), deepid_font, 1, deepid_col, deepid_font_thickness)
                        if show_ids[2]:
                            cv2.putText(img, str(p.faceid + " : " + str(int(p.faceid_conf)) + "%"), 
                                        (x - 90, y - 90), faceid_font, 1, faceid_col, faceid_font_thickness)
                    if (isinstance(show_reid, tuple) and 
                        np.asarray(show_reid).shape == (2,)):
                        if show_reid[0] > 0:
                            cv2.putText(img, "                     REIDING", reid_pos, 
                                        reid_status_font, 1, reid_col, 1, cv2.LINE_AA)
                        if show_reid[1] > 0:
                            cv2.putText(img, "                     REIDING", reid_pos, 
                                        reid_status_font, 1, reid_col, 1, cv2.LINE_AA)
                            cv2.putText(img, "DEDUPLICATING <-", reid_pos, reid_status_font, 
                                        1, reid_dup_col, 1, cv2.LINE_AA)
                    else:
                        msg = "visualizePeople() -> show_ids='" + str(show_ids) + "' is not valid."
                        add_error_log(msg)
                        raise ValueError(msg)
            else:
                msg = "visualizePeople() -> Input 'people' list has unsupported element."
                add_error_log(msg)
                raise ValueError(msg)
    return img
