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


import os
import cv2

from collections import Counter

# my objects & tools
from .utils.person import Person
from .utils.gttools import *
from .utils.mytools import *

# my configurator
from .config import MyConfigurator
from .localconfig import MyLocalConfigurator

# my detectors
from .dt_yolocv.myyolocv import MyYOLOCV
from .dt_yolopt.myyolopt import MyYOLOPT

# my trackers
from .tk_sort.mysort import MySort
from .tk_deepsort.mydeepsort import MyDeepSort
from .tk_centroid.mycentroid_advanced import MyCTTracker

# my reiders
from .ri_facenet.myfacenet import MyFacenet
from .ri_deepreid.mydeepreid import MyDeepReID



class PManager(object):

    def __loadCFG__(self):
        self.__cfg__ = MyConfigurator()
        self.__mstruct__ = self.__cfg__.mstruct
        self.__cfg__.loadMCFG()
        self.__cfg__.loadDCFG()
        self.__cfg__.loadTCFG()
        self.__cfg__.loadRCFG()

    def __loadLocalCFG__(self, local_cfg_dir):
        self.__cfg__ = MyLocalConfigurator(local_cfg_dir)
        self.__mstruct__ = self.__cfg__.mstruct
        self.__cfg__.loadMCFG()
        self.__cfg__.loadDCFG()
        self.__cfg__.loadTCFG()
        self.__cfg__.loadRCFG()

    def __setCFG__(self):
        self.__setDetector__()
        self.__setTracker__()
        self.__setREIDer__()

    def __loadAllEval__(self):
        # Auto config GT file according to known inputed video
        self.__evaIO__ = EvalIO(mode=self.evalmode)
        if self.enableEval:
            print(" - EVA : Enable")
            self.__evaIO__.loadInputGTMap(self.__cfg__.dcfg_gt.input_gt_map_file)
            gt_file = self.__evaIO__.getGTFileName(self.__cfg__.mcfg.input_video)
            if gt_file != "":
                tmp = os.path.join(self.__mstruct__.gt_dir, gt_file)
                self.__cfg__.dcfg_gt.gt_file = normalizePathFDS(self.__mstruct__.global_root_dir, tmp)
                # Last to load
                self.__eva__ = MyEval(joinFPathFull(self.__mstruct__.global_root_dir, self.__cfg__.dcfg_gt.gt_file))
            else:
                self.__eva__ = MyEvalEmpty()
        else:
            print(" - EVA : Disable")

    def __updateVPInfo__(self):
        info = self.__dtname__ + self.__mstruct__.vp_sp + self.__tkname__ + self.__mstruct__.vp_sp + self.__riname__
        cv2.putText(self.frame_visual, info, self.__mstruct__.vp, self.__mstruct__.vp_font, 1, self.__mstruct__.vp_co, 1, cv2.LINE_AA)

    def __init__(self, enableEval=False, localConfig=False):
        self.enableEval = enableEval
        self.__ppobjlist__ = []
        self.__dtname__ = ""
        self.__tkname__ = ""
        self.__riname__ = ""
        self.evalmode = ""
        if not localConfig:
            self.__loadCFG__()
            self.__setCFG__()
            self.__loadAllEval__()
    
    def setLocalConfig(self, local_cfg_dir):
        self.__loadLocalCFG__(getAbsPathFDS(local_cfg_dir))
        self.__setCFG__()
        self.__loadAllEval__()
    
    def getInputFile(self):
        return self.__cfg__.mcfg.input_video
    
    def forceHD(self):
        return self.__cfg__.mcfg.force_hd

    def getCurrentPPOBL(self):
        return self.__ppobjlist__
    
    def getFrameClean(self):
        return self.frame_clean

    def getFrameVisual(self):
        return self.frame_visual

    def setFrame(self, frame):
        self.frame_clean = frame.copy()
        self.frame_visual = frame.copy()
    
    def setPPObjList(self, current_ppobl):
        if current_ppobl:
            self.__ppobjlist__ = current_ppobl
        else:
            self.__ppobjlist__ = []



    ###########################################
    # Detector
    ###########################################

    def __setDetector__(self):
        if self.__cfg__.mcfg.detector.lower() == self.__mstruct__.str.yolo:
            self.__dt__ = MyYOLOCV(self.__cfg__.dcfg_yolocv)
            self.__dtname__ = self.__mstruct__.str.dtname_yl
        elif self.__cfg__.mcfg.detector.lower() == self.__mstruct__.str.yolo_utlt:
            self.__dt__ = MyYOLOPT(self.__cfg__.dcfg_yolopt)
            self.__dtname__ = self.__mstruct__.str.dtname_yl
        elif self.__cfg__.mcfg.detector.lower() == self.__mstruct__.str.gt:
            print("DT: In GT mode, TK and RI options will be ignored.")
            self.__cfg__.mcfg.tracker = "None"
            self.__cfg__.mcfg.reider = "None"
            self.__dtname__ = self.__mstruct__.str.dtname_gt
            self.__dt__ = GTLoader()
            self.__dt__.loadGT(self.__cfg__.dcfg_gt.gt_file)
            self.evalmode = "gt_real"
        else:
            self.__dt__ = EmptyDetecter()
            print("DT: In None mode, TK and RI options will be ignored.")
            self.__cfg__.mcfg.tracker = "None"
            self.__cfg__.mcfg.reider = "None"

    def detectFramePPOBL(self, frame, visual):
        self.frame_clean = frame.copy()
        self.frame_visual = frame.copy()
        tmp_ppobl = []
                    
        if self.__cfg__.mcfg.detector.lower() == self.__mstruct__.str.yolo:
            self.frame_visual, pboxes_xywh, pboxes_xyxy, repspoints = self.__dt__.detectFrame(frame, visual=visual, repspoint_callibration=self.__cfg__.dcfg_yolocv.repspoint_callibration)
            for i in range(0, len(pboxes_xyxy)):
                tmp_ppobl.append(
                    Person(i, i, self.__mstruct__.str.unk_fid, self.__mstruct__.str.unk_did, 
                    (repspoints[i][0], repspoints[i][1]), box_xywh=pboxes_xywh[i], box_xyxy=pboxes_xyxy[i])
                )
        elif self.__cfg__.mcfg.detector.lower() == self.__mstruct__.str.yolo_utlt:
            self.frame_visual, pboxes_xywh, pboxes_xyxy, repspoints, keypoints = self.__dt__.detectFrame(frame, visual=visual)
            for i in range(0, len(pboxes_xyxy)):
                tmp_ppobl.append(
                    Person(i, i, self.__mstruct__.str.unk_fid, self.__mstruct__.str.unk_did, 
                    (repspoints[i][0], repspoints[i][1]), box_xywh=pboxes_xywh[i], box_xyxy=pboxes_xyxy[i], keypoints=keypoints[i])
                )
        elif self.__dtname__ == self.__mstruct__.str.dtname_gt:
            tmp_ppobl = self.__dt__.nextFrame()
        
        self.__updateVPInfo__()

        return tmp_ppobl




    ###########################################
    # Tracker
    ###########################################

    def __setTracker__(self):
        if self.__cfg__.mcfg.tracker.lower() == self.__mstruct__.str.centroid:
            self.__tk__ = MyCTTracker(self.__cfg__.tcfg_centroid)
            self.__tkname__ = self.__mstruct__.str.tkname_ct
        elif self.__cfg__.mcfg.tracker.lower() == self.__mstruct__.str.sort:
            self.__tk__ = MySort(self.__cfg__.tcfg_sort)
            self.__tkname__ = self.__mstruct__.str.tkname_st
        elif self.__cfg__.mcfg.tracker.lower() == self.__mstruct__.str.deepsort:
            self.__tk__ = MyDeepSort(self.__cfg__.tcfg_deepsort)
            self.__tkname__ = self.__mstruct__.str.tkname_ds
        else:
            self.__tk__ = NothingTracker()

    def updateTrackerPPOBL(self, ppobjlist):
        if ppobjlist:
            self.__ppobjlist__ = self.__tk__.update(self.frame_clean, ppobjlist)
        else:
            self.__ppobjlist__ = []




    ###########################################
    # Re-ider
    ###########################################

    def __setREIDer__(self):
        if self.__cfg__.mcfg.reider.lower() == self.__mstruct__.str.facenet:
            self.__ri__ = MyFacenet(self.__cfg__.rcfg_facenet)
            self.__riname__ = self.__mstruct__.str.riname_fn
            self.evalmode = "faceid"
        elif self.__cfg__.mcfg.reider.lower() == self.__mstruct__.str.deepreid:
            self.__ri__ = MyDeepReID(self.__cfg__.rcfg_deepreid)
            self.__ri__.load_classifier()
            self.__riname__ = self.__mstruct__.str.riname_dr
            self.evalmode = "deepid"
        else:
            if self.__dtname__ != "" and self.__tkname__ != "":
                self.__ri__ = TackingOnlyReider(static=True)
                self.evalmode = ""
            else:
                self.__ri__ = NothingReider()
            
    def reidNormal(self):
        if self.__riname__ == self.__mstruct__.str.riname_fn:
            self.__reidFaceNormal__()
        elif self.__riname__ == self.__mstruct__.str.riname_dr:
            self.__reidDeepNormal__()
        else:
            self.__reidEmpty__()

    def reidDupkiller(self):
        if self.__riname__ == self.__mstruct__.str.riname_fn:
            self.__reidDupFacekiller__()
        elif self.__riname__ == self.__mstruct__.str.riname_dr:
            self.__reidDupDeepkiller__()
        else:
            self.__reidEmpty__()

    def __reidEmpty__(self):
        index = 0
        for person in self.__ppobjlist__:
            deepid = str(person.getDeepid())
            if self.__mstruct__.str.err_did in deepid or self.__mstruct__.str.unk_did in deepid:
                cv2.putText(self.frame_visual, "Tracking lost!", self.__mstruct__.vp_reid, self.__mstruct__.vp_font, 1, (0, 255, 255), 1, cv2.LINE_AA)
                self.__ppobjlist__[index].updateDeepid(self.__ri__.recognize(self.__mstruct__.str.unk_did))
                self.__selfEVARTcheckInReID__()
            index += 1

    def __reidDeepNormal__(self):
        index = 0
        self.__deepidlistTMP__ = []
        for person in self.__ppobjlist__:
            deepid = str(person.getDeepid())
            if self.__mstruct__.str.err_did in deepid or self.__mstruct__.str.unk_did in deepid:
                miniframe = self.frame_clean.copy()
                try:
                    [x, y, w, h] = person.getBoxXYWH()
                    miniframe = miniframe[int(y):int(y)+int(h), int(x):int(x)+int(w)]
                    # cv2.imshow("miniframe", cv2.resize(miniframe, self.__cfg__.dr_wh))
                    miniframe = cv2.cvtColor(miniframe, cv2.COLOR_BGR2RGB)
                    self.__ppobjlist__[index].updateDeepid(self.__ri__.recoginize_plus(cv2.resize(miniframe, self.__cfg__.dr_wh)))
                    cv2.putText(self.frame_visual, "REIDing...", self.__mstruct__.vp_reid, self.__mstruct__.vp_font, 1, (0, 255, 255), 1, cv2.LINE_AA)
                    self.__selfEVARTcheckInReID__()
                except Exception as e:
                    print("PMG: " + str(e))
            self.__deepidlistTMP__.append(deepid[:-3])
            index += 1

    def __reidDupDeepkiller__(self):
        if len(self.__deepidlistTMP__) != len(set(self.__deepidlistTMP__)):
            ddeepids = [k for k, v in Counter(self.__deepidlistTMP__).items() if v > 1]
            for ddeepid in ddeepids:
                index = 0
                for person in self.__ppobjlist__:
                    try:
                        if str(person.getDeepid())[:-3] == ddeepid:
                            [x, y, w, h] = person.getBoxXYWH()
                            miniframe = self.frame_clean.copy()
                            miniframe = miniframe[int(y):int(y)+int(h), int(x):int(x)+int(w)]
                            miniframe = cv2.cvtColor(miniframe, cv2.COLOR_BGR2RGB)
                            self.__ppobjlist__[index].updateDeepid(self.__ri__.recoginize_plus(cv2.resize(miniframe, self.__cfg__.dr_wh)))
                            cv2.putText(self.frame_visual, "REIDing...", self.__mstruct__.vp_reid, self.__mstruct__.vp_font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                            self.__selfEVARTcheckInReID__()
                    except Exception as e:
                        print("PMG: " + str(e))
                    index += 1

    def __reidFaceNormal__(self):
        index = 0
        self.__faceidlistTMP__ = []
        for person in self.__ppobjlist__:
            faceid = str(person.getFaceid())
            if self.__mstruct__.str.err_fid in faceid or self.__mstruct__.str.unk_fid in faceid:
                (x, y) = person.getRepspoint()
                miniframe = self.frame_clean.copy()
                try:
                    if self.__dtname__ == self.__mstruct__.str.dtname_yl:
                        miniframe = miniframe[int(y+self.__cfg__.rcfg_facenet.yl_h_callibration[0]):int(y+self.__cfg__.rcfg_facenet.yl_h_callibration[1]), 
                                                int(x+self.__cfg__.rcfg_facenet.yl_w_callibration[0]):int(x+self.__cfg__.rcfg_facenet.yl_w_callibration[1])]
                    # cv2.imshow("miniframe", miniframe)
                    miniframe = cv2.cvtColor(miniframe, cv2.COLOR_BGR2RGB)
                    self.__ppobjlist__[index].updateFaceid(self.__ri__.recognize_face(miniframe))
                    cv2.putText(self.frame_visual, "REIDing...", self.__mstruct__.vp_reid, self.__mstruct__.vp_font, 1, (0, 255, 255), 1, cv2.LINE_AA)
                    self.__selfEVARTcheckInReID__()
                except Exception as e:
                    print("PMG: " + str(e))
            self.__faceidlistTMP__.append(faceid[:-3])
            index += 1

    def __reidDupFacekiller__(self):
        if len(self.__faceidlistTMP__) != len(set(self.__faceidlistTMP__)):
            dfaceids = [k for k, v in Counter(self.__faceidlistTMP__).items() if v > 1]
            for dfaceid in dfaceids:
                index = 0
                for person in self.__ppobjlist__:
                    try:
                        if str(person.getFaceid())[:-3] == dfaceid:
                            (x, y) = person.getRepspoint()
                            miniframe = self.frame_clean.copy()
                            if self.__dtname__ == self.__mstruct__.str.dtname_yl:
                                miniframe = miniframe[int(y+self.__cfg__.rcfg_facenet.yl_h_callibration[0]):int(y+self.__cfg__.rcfg_facenet.yl_h_callibration[1]), 
                                                        int(x+self.__cfg__.rcfg_facenet.yl_w_callibration[0]):int(x+self.__cfg__.rcfg_facenet.yl_w_callibration[1])]
                            # cv2.imshow("miniframe2", miniframe)
                            miniframe = cv2.cvtColor(miniframe, cv2.COLOR_BGR2RGB)
                            self.__ppobjlist__[index].updateFaceid(self.__ri__.recognize_face(miniframe))
                            cv2.putText(self.frame_visual, "REIDing...", self.__mstruct__.vp_reid, self.__mstruct__.vp_font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                            self.__selfEVARTcheckInReID__()
                    except Exception as e:
                        print("PMG: " + str(e))
                    index += 1




    ###########################################
    # EVA_RT & EVA_IO
    ###########################################

    def __selfEVARTcheckInReID__(self):
        if self.enableEval:
            self.__eva__.checkInReID()

    def addPersonEVAIO(self, frame_id, person):
        self.__evaIO__.add_person(frame_id, person)

    def dumpResultEVAIO(self, result_txt_file):
        self.__evaIO__.dump(result_txt_file)

    def updateEVART(self):
        if self.enableEval:
            self.__eva__.checkFrame(self.__ppobjlist__, id_mode=self.evalmode)

    def getEVARTSummary(self):
        if self.enableEval:
            return self.__eva__.getSummary()

    def generateExtraInfoForEVAIO(self):
        res_txt = str(self.__cfg__.mcfg.detector).lower() + "_" + str(self.__cfg__.mcfg.tracker).lower() + "_" + str(self.__cfg__.mcfg.reider).lower()
        extra_info = "- Input video     :   " + str(self.__cfg__.mcfg.input_video) + "\n"
        extra_info = extra_info + "- GT file         :   " + str(self.__cfg__.dcfg_gt.gt_file) + "\n"
        extra_info = extra_info + "- Detector        :   " + str(self.__cfg__.mcfg.detector) + "\n"
        extra_info = extra_info + "- Tracker         :   " + str(self.__cfg__.mcfg.tracker) + "\n"
        extra_info = extra_info + "- ReIDer          :   " + str(self.__cfg__.mcfg.reider) + "\n"
        if self.__cfg__.mcfg.reider == "Facenet":
            extra_info = extra_info + "    + Classifier  :   " + str(self.__cfg__.rcfg_facenet.classifier_file) + "\n"
            res_txt = res_txt + ".txt"
        elif self.__cfg__.mcfg.reider == "DeepReID":
            extra_info = extra_info + "    + Model       :   " + str(self.__cfg__.rcfg_deepreid.model_name) + "\n"
            extra_info = extra_info + "    + Model file  :   " + str(self.__cfg__.rcfg_deepreid.model_path) + "\n"
            extra_info = extra_info + "    + Classifier  :   " + str(self.__cfg__.rcfg_deepreid.classifier_pkl) + "\n"
            res_txt = res_txt + "_" + str(self.__cfg__.rcfg_deepreid.model_name).lower() + ".txt"
        else:
            res_txt = res_txt + ".txt"
        return res_txt, extra_info

