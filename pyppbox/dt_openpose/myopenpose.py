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

import sys
import cv2
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/pybin')
os.environ['PATH']  = os.environ['PATH'] + ';' +  dir_path + '/pybin;'

import pyopenpose as op

class MyOpenPose(object):

    def __init__(self, cfg):
        self.params = dict()
        self.params["model_folder"] = cfg.model_folder
        self.params["model_pose"] = cfg.model_pose
        self.params["net_resolution"] = cfg.model_resolution
        self.params["output_resolution"] = cfg.output_resolution
        self.params["hand"] = cfg.hand
        self.params["face"] = False
        self.params["number_people_max"] = cfg.number_people_max
        self.params["disable_blending"] = cfg.disable_blending
        self.params["display"] = 0
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(self.params)
        self.opWrapper.start()

    def detectImageFile(self, path):
        self.datum = op.Datum()
        imageToProcess = cv2.imread(path)
        self.datum.cvInputData = imageToProcess
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
        return self.datum.cvOutputData, self.datum.poseKeypoints

    def detectFrame(self, frame):
        self.datum = op.Datum()
        self.datum.cvInputData = frame
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
        return self.datum.cvOutputData, self.datum.poseKeypoints
        # return self.datum.cvOutputData, self.datum.faceRectangles, self.datum.poseKeypoints

