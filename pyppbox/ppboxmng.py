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
import cv2
import numpy as np

from collections import Counter

# my objects & tools
from .utils.person import Person
from .utils.gttools import *
from .utils.mytools import *

# my configurator
from .config import MyConfigurator

# my detectors
from .dt_yolocv.myyolocv import MyYOLOCV

# my trackers
from .tk_sort.mysort import MySort
from .tk_deepsort.mydeepsort import MyDeepSort
from .tk_centroid.mycentroid_advanced import MyCTTracker

# my reiders
from .ri_facenet.myfacenet import MyFacenet
from .ri_deepreid.mydeepreid import MyDeepReID



class PManager(object):


    def __init__(self, enableEval=False):
        self.enableEval = enableEval
        self.curr_ppobjlist = []
        self.dt_name = ""
        self.tk_name = ""
        self.ri_name = ""
        self.evalmode = ""
        self.loadCFG()
        self.setCFG()
        self.loadAllEval()

    def loadCFG(self):
        self.cfg = MyConfigurator()
        self.mstruct = self.cfg.getMStruct()
        self.cfg.loadMCFG()
        self.cfg.loadDCFG()
        self.cfg.loadTCFG()
        self.cfg.loadRCFG()

    def setCFG(self):
        self.setDetector()
        self.setTracker()
        self.setREIDer()
    
    def getInputFile(self):
        return self.cfg.mcfg.input_video
    
    def forceHD(self):
        return self.cfg.mcfg.force_hd

    def getCurrentPPOBL(self):
        return self.curr_ppobjlist
    
    def getFrameClean(self):
        return self.frame_clean

    def getFrameVisual(self):
        return self.frame_visual

    def updateVPInfo(self):
        info = self.dt_name + self.mstruct.vp_sp + self.tk_name + self.mstruct.vp_sp + self.ri_name
        cv2.putText(self.frame_visual, info, self.mstruct.vp, self.mstruct.vp_font, 1, self.mstruct.vp_co, 1, cv2.LINE_AA)

    def loadAllEval(self):
        # Auto config GT file according to known inputed video
        self.eva_io = EvalIO(mode=self.evalmode)
        if self.enableEval:
            print(" - EVA : Enable")
            self.eva_io.loadInputGTMap(self.cfg.dcfg_gt.input_gt_map_file)
            gt_file = self.eva_io.getGTFileName(self.cfg.mcfg.input_video)
            if gt_file != "":
                tmp = os.path.join(self.mstruct.gt_dir, self.eva_io.getGTFileName(self.cfg.mcfg.input_video))
                self.cfg.dcfg_gt.gt_file = normalizePathFDS(self.mstruct.root_dir, tmp)
                # Last to load
                self.eva = MyEval(joinFPathFull(self.mstruct.root_dir, self.cfg.dcfg_gt.gt_file))
            else:
                self.eva = MyEvalEmpty()
        else:
            print(" - EVA : Disable")




    ###########################################
    # Detector
    ###########################################

    def setDetector(self):
        if self.cfg.mcfg.detector.lower() == self.mstruct.str.yolo:
            self.dt = MyYOLOCV(self.cfg.dcfg_yolo)
            self.dt_name = self.mstruct.str.dtname_yl
        elif self.cfg.mcfg.detector.lower() == self.mstruct.str.gt:
            print("DT: In GT mode, TK and RI options will be ignored.")
            self.cfg.mcfg.tracker = "None"
            self.cfg.mcfg.reider = "None"
            self.dt_name = self.mstruct.str.dtname_gt
            self.dt = GTLoader()
            self.dt.loadGT(self.cfg.dcfg_gt.gt_file)
            self.evalmode = "gt_real"
        else:
            self.dt = EmptyDetecter()
            print("DT: In None mode, TK and RI options will be ignored.")
            self.cfg.mcfg.tracker = "None"
            self.cfg.mcfg.reider = "None"

    def detectFramePPOBL(self, frame, visual):
        self.frame_clean = frame.copy()
        self.frame_visual = frame.copy()
        tmp_ppobl = []
                    
        if self.dt_name == self.mstruct.str.dtname_yl:
            self.frame_visual, bboxes, bboxes_tlbr, repspoints = self.dt.detectFrame(frame, visual=visual, repspoint_callibration=self.cfg.dcfg_yolo.repspoint_callibration)
            for i in range(0, len(bboxes_tlbr)):
                tmp_ppobl.append(
                    Person(i, i, self.mstruct.str.unk_fid, self.mstruct.str.unk_did, 
                    (repspoints[i][0], repspoints[i][1]), bbox=bboxes[i], bbox_tlbr=bboxes_tlbr[i])
                )
        
        elif self.dt_name == self.mstruct.str.dtname_gt:
            tmp_ppobl = self.dt.nextFrame()
        
        self.updateVPInfo()

        return tmp_ppobl




    ###########################################
    # Tracker
    ###########################################

    def setTracker(self):
        if self.cfg.mcfg.tracker.lower() == self.mstruct.str.centroid:
            self.tk = MyCTTracker(self.cfg.tcfg_centroid)
            self.tk_name = self.mstruct.str.tkname_ct
        elif self.cfg.mcfg.tracker.lower() == self.mstruct.str.sort:
            self.tk = MySort(self.cfg.tcfg_sort)
            self.tk_name = self.mstruct.str.tkname_st
        elif self.cfg.mcfg.tracker.lower() == self.mstruct.str.deepsort:
            self.tk = MyDeepSort(self.cfg.tcfg_deepsort)
            self.tk_name = self.mstruct.str.tkname_ds
        else:
            self.tk = NothingTracker()

    def updateTrackerPPOBL(self, current_ppobl):
        if current_ppobl:
            self.curr_ppobjlist = self.tk.update(self.frame_clean, current_ppobl)
        else:
            self.curr_ppobjlist = []




    ###########################################
    # Re-ider
    ###########################################

    def setREIDer(self):
        if self.cfg.mcfg.reider.lower() == self.mstruct.str.facenet:
            self.ri = MyFacenet(self.cfg.rcfg_facenet)
            self.ri_name = self.mstruct.str.riname_fn
            self.evalmode = "faceid"
        elif self.cfg.mcfg.reider.lower() == self.mstruct.str.deepreid:
            self.ri = MyDeepReID(self.cfg.rcfg_deepreid)
            self.ri.load_classifier()
            self.ri_name = self.mstruct.str.riname_dr
            self.evalmode = "deepid"
        else:
            if self.dt_name != "" and self.tk_name != "":
                self.ri = TackingOnlyReider(static=True)
                self.evalmode = ""
            else:
                self.ri = NothingReider()
            
    def reidNormal(self):
        if self.ri_name == self.mstruct.str.riname_fn:
            self.reidFaceNormal()
        elif self.ri_name == self.mstruct.str.riname_dr:
            self.reidDeepNormal()
        else:
            self.reidEmpty()

    def reidDupkiller(self):
        if self.ri_name == self.mstruct.str.riname_fn:
            self.reidDupFacekiller()
        elif self.ri_name == self.mstruct.str.riname_dr:
            self.reidDupDeepkiller()
        else:
            self.reidEmpty()

    def reidEmpty(self):
        index = 0
        for person in self.curr_ppobjlist:
            deepid = str(person.getDeepid())
            if self.mstruct.str.err_did in deepid or self.mstruct.str.unk_did in deepid:
                cv2.putText(self.frame_visual, "Tracking lost!", self.mstruct.vp_reid, self.mstruct.vp_font, 1, (0, 255, 255), 1, cv2.LINE_AA)
                self.curr_ppobjlist[index].updateDeepid(self.ri.recognize(self.mstruct.str.unk_did))
                self.selfRealtimeEvalReIDCheckIn()
            index += 1

    def reidDeepNormal(self):
        index = 0
        self.tmp_deepidlist = []
        for person in self.curr_ppobjlist:
            deepid = str(person.getDeepid())
            if self.mstruct.str.err_did in deepid or self.mstruct.str.unk_did in deepid:
                miniframe = self.frame_clean.copy()
                try:
                    [x, y, w, h] = person.getBbox()
                    miniframe = miniframe[int(y):int(y)+int(h), int(x):int(x)+int(w)]
                    # cv2.imshow("miniframe", cv2.resize(miniframe, self.cfg.dr_wh))
                    miniframe = cv2.cvtColor(miniframe, cv2.COLOR_BGR2RGB)
                    self.curr_ppobjlist[index].updateDeepid(self.ri.recoginize_plus(cv2.resize(miniframe, self.cfg.dr_wh)))
                    cv2.putText(self.frame_visual, "REIDing...", self.mstruct.vp_reid, self.mstruct.vp_font, 1, (0, 255, 255), 1, cv2.LINE_AA)
                    self.selfRealtimeEvalReIDCheckIn()
                except Exception as e:
                    print("PMG: " + str(e))
            self.tmp_deepidlist.append(deepid[:-3])
            index += 1

    def reidDupDeepkiller(self):
        if len(self.tmp_deepidlist) != len(set(self.tmp_deepidlist)):
            ddeepids = [k for k, v in Counter(self.tmp_deepidlist).items() if v > 1]
            for ddeepid in ddeepids:
                index = 0
                for person in self.curr_ppobjlist:
                    try:
                        if str(person.getDeepid())[:-3] == ddeepid:
                            [x, y, w, h] = person.getBbox()
                            miniframe = self.frame_clean.copy()
                            miniframe = miniframe[int(y):int(y)+int(h), int(x):int(x)+int(w)]
                            miniframe = cv2.cvtColor(miniframe, cv2.COLOR_BGR2RGB)
                            self.curr_ppobjlist[index].updateDeepid(self.ri.recoginize_plus(cv2.resize(miniframe, self.cfg.dr_wh)))
                            cv2.putText(self.frame_visual, "REIDing...", self.mstruct.vp_reid, self.mstruct.vp_font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                            self.selfRealtimeEvalReIDCheckIn()
                    except Exception as e:
                        print("PMG: " + str(e))
                    index += 1

    def reidFaceNormal(self):
        index = 0
        self.tmp_faceidlist = []
        for person in self.curr_ppobjlist:
            faceid = str(person.getFaceid())
            if self.mstruct.str.err_fid in faceid or self.mstruct.str.unk_fid in faceid:
                (x, y) = person.getRepspoint()
                miniframe = self.frame_clean.copy()
                try:
                    if self.dt_name == self.mstruct.str.dtname_yl:
                        miniframe = miniframe[int(y+self.cfg.rcfg_facenet.yl_h_callibration[0]):int(y+self.cfg.rcfg_facenet.yl_h_callibration[1]), 
                                                int(x+self.cfg.rcfg_facenet.yl_w_callibration[0]):int(x+self.cfg.rcfg_facenet.yl_w_callibration[1])]
                    # cv2.imshow("miniframe", miniframe)
                    miniframe = cv2.cvtColor(miniframe, cv2.COLOR_BGR2RGB)
                    self.curr_ppobjlist[index].updateFaceid(self.ri.recognize_face(miniframe))
                    cv2.putText(self.frame_visual, "REIDing...", self.mstruct.vp_reid, self.mstruct.vp_font, 1, (0, 255, 255), 1, cv2.LINE_AA)
                    self.selfRealtimeEvalReIDCheckIn()
                except Exception as e:
                    print("PMG: " + str(e))
            self.tmp_faceidlist.append(faceid[:-3])
            index += 1

    def reidDupFacekiller(self):
        if len(self.tmp_faceidlist) != len(set(self.tmp_faceidlist)):
            dfaceids = [k for k, v in Counter(self.tmp_faceidlist).items() if v > 1]
            for dfaceid in dfaceids:
                index = 0
                for person in self.curr_ppobjlist:
                    try:
                        if str(person.getFaceid())[:-3] == dfaceid:
                            (x, y) = person.getRepspoint()
                            miniframe = self.frame_clean.copy()
                            if self.dt_name == self.mstruct.str.dtname_yl:
                                miniframe = miniframe[int(y+self.cfg.rcfg_facenet.yl_h_callibration[0]):int(y+self.cfg.rcfg_facenet.yl_h_callibration[1]), 
                                                        int(x+self.cfg.rcfg_facenet.yl_w_callibration[0]):int(x+self.cfg.rcfg_facenet.yl_w_callibration[1])]
                            # cv2.imshow("miniframe2", miniframe)
                            miniframe = cv2.cvtColor(miniframe, cv2.COLOR_BGR2RGB)
                            self.curr_ppobjlist[index].updateFaceid(self.ri.recognize_face(miniframe))
                            cv2.putText(self.frame_visual, "REIDing...", self.mstruct.vp_reid, self.mstruct.vp_font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                            self.selfRealtimeEvalReIDCheckIn()
                    except Exception as e:
                        print("PMG: " + str(e))
                    index += 1




    ###########################################
    # Realtime & Offline EVA
    ###########################################


    def selfRealtimeEval(self):
        if self.enableEval:
            self.eva.checkFrame(self.curr_ppobjlist, id_mode=self.evalmode)
    
    def selfRealtimeEvalReIDCheckIn(self):
        if self.enableEval:
            self.eva.checkInReID()

    def getRealtimeEvalResult(self):
        if self.enableEval:
            return self.eva.getSummary()

    def generateExtraInfoForOfflineEva(self):
        res_txt = str(self.cfg.mcfg.detector).lower() + "_" + str(self.cfg.mcfg.tracker).lower() + "_" + str(self.cfg.mcfg.reider).lower()
        extra_info = "- Input video     :   " + str(self.cfg.mcfg.input_video) + "\n"
        extra_info = extra_info + "- GT file         :   " + str(self.cfg.dcfg_gt.gt_file) + "\n"
        extra_info = extra_info + "- Detector        :   " + str(self.cfg.mcfg.detector) + "\n"
        extra_info = extra_info + "- Tracker         :   " + str(self.cfg.mcfg.tracker) + "\n"
        extra_info = extra_info + "- ReIDer          :   " + str(self.cfg.mcfg.reider) + "\n"
        if self.cfg.mcfg.reider == "Facenet":
            extra_info = extra_info + "    + Classifier  :   " + str(self.cfg.rcfg_facenet.classifier_file) + "\n"
            res_txt = res_txt + ".txt"
        elif self.cfg.mcfg.reider == "DeepReID":
            extra_info = extra_info + "    + Model       :   " + str(self.cfg.rcfg_deepreid.model_name) + "\n"
            extra_info = extra_info + "    + Model file  :   " + str(self.cfg.rcfg_deepreid.model_path) + "\n"
            extra_info = extra_info + "    + Classifier  :   " + str(self.cfg.rcfg_deepreid.classifier_pkl) + "\n"
            res_txt = res_txt + "_" + str(self.cfg.rcfg_deepreid.model_name).lower() + ".txt"
        else:
            res_txt = res_txt + ".txt"
        return res_txt, extra_info

