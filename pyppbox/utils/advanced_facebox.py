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

from functools import cache
import numpy as np
from math import hypot


def getDist(p1, p2):
    res = 0.0
    (x1, y1) = p1
    (x2, y2) = p2
    res = hypot(x2 - x1, y2 - y1)
    return res

def tlbrToXyah(box):
    box_list = box.tolist()
    w = box_list[2] - box_list[0]
    h = box_list[3] - box_list[1]
    return np.asarray([box[0], box[1], w, h])

def eliminateZero(points):
    res = np.asarray(points)
    res = res[:, :-1]
    res = res[np.all(res != 0, axis=1)]
    return res

def generateBboxTlbr(points):
    points = eliminateZero(points)
    bot_left_x = min(point[0] for point in points)
    bot_left_y = min(point[1] for point in points)
    top_right_x = max(point[0] for point in points)
    top_right_y = max(point[1] for point in points)
    return np.asarray([bot_left_x, bot_left_y, top_right_x, top_right_y])

def isSomehowEqual(val_1, val_2, margin):
    res = True
    val_1 = int(float(val_1))
    val_2 = int(float(val_2))
    margin = int(float(margin))
    if abs(val_1 - val_2) > margin:
        res = False
    return res


class KeypointsManager(object):

    def __init__(self):
        pass

    def updateKeypoints(self, keypoints):
        self.keypoints = keypoints

    def countPeople(self):
        res = 0
        if self.keypoints is not None:
            res = len(self.keypoints)
        return res

    def isNotEmpty(self):
        res = True
        if self.keypoints is None:
            res = False
        elif len(self.keypoints) <= 0:
            res = False
        return res

    def getBBoxTLBR(self, index):
        return generateBboxTlbr(self.keypoints[index])

    def getBBox(self, index):
        return tlbrToXyah(generateBboxTlbr(self.keypoints[index]))

    def getNeck(self, index):
        return (self.keypoints[index][1][0], self.keypoints[index][1][1])

    def getBodyKeypointByNeck(self, neck):
        (neck_x, neck_y) = neck
        res_k = []
        for k in self.keypoints:
            if isSomehowEqual(neck_x, k[1][0], 5) and isSomehowEqual(neck_y, k[1][1], 5):
                res_k = k
                break
        return res_k

    def getPerfectFaceBox(self, neck, weight=3):
        tmp_bk = self.getBodyKeypointByNeck(neck)
        nose = (tmp_bk[0][0], tmp_bk[0][1])
        rshd = (tmp_bk[2][0], tmp_bk[0][1])
        lshd = (tmp_bk[5][0], tmp_bk[5][1])
        weight_NN, weight_SS = self.getWeights(weight=weight)
        dist = max((weight_SS * getDist(rshd, lshd)), (weight_NN * getDist(nose, neck)))
        x = tmp_bk[0][0] - dist
        y = tmp_bk[0][1] - dist
        size = 2 * dist
        return x, y, size

    def getWeights(self, weight=3):
        weight_NN = 0.6
        weight_SS = 0.388
        if weight == 1:
            weight_NN = 0.4
            weight_SS = 0.258
        elif weight == 2:
            weight_NN = 0.5
            weight_SS = 0.323
        elif weight == 4:
            weight_NN = 0.75
            weight_SS = 0.485
        elif weight == 5:
            weight_NN = 0.85
            weight_SS = 0.55
        return weight_NN, weight_SS
