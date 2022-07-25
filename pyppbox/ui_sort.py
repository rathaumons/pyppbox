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

from PyQt6 import QtCore, QtGui, QtWidgets
from pyppbox.config import MyConfigurator, MyCFGIO
from pyppbox.utils.mytools import replaceLine, joinFPathFull

root_dir = os.path.dirname(__file__)
cfg_dir = joinFPathFull(root_dir, 'cfg')


class Ui_SORTForm(object):

    def setupUi(self, SORTForm):
        SORTForm.setObjectName("SORTForm")
        SORTForm.resize(390, 150)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(SORTForm.sizePolicy().hasHeightForWidth())
        SORTForm.setSizePolicy(sizePolicy)
        SORTForm.setMinimumSize(QtCore.QSize(390, 150))
        SORTForm.setMaximumSize(QtCore.QSize(390, 150))
        self.save_pushButton = QtWidgets.QPushButton(SORTForm)
        self.save_pushButton.setGeometry(QtCore.QRect(150, 110, 91, 31))
        self.save_pushButton.setObjectName("save_pushButton")
        self.st_min_hits_label = QtWidgets.QLabel(SORTForm)
        self.st_min_hits_label.setGeometry(QtCore.QRect(10, 40, 91, 16))
        self.st_min_hits_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.st_min_hits_label.setObjectName("st_min_hits_label")
        self.st_iou_threshold_lineEdit = QtWidgets.QLineEdit(SORTForm)
        self.st_iou_threshold_lineEdit.setGeometry(QtCore.QRect(110, 70, 241, 21))
        self.st_iou_threshold_lineEdit.setObjectName("st_iou_threshold_lineEdit")
        self.st_min_hits_lineEdit = QtWidgets.QLineEdit(SORTForm)
        self.st_min_hits_lineEdit.setGeometry(QtCore.QRect(110, 40, 241, 21))
        self.st_min_hits_lineEdit.setObjectName("st_min_hits_lineEdit")
        self.st_max_age_label = QtWidgets.QLabel(SORTForm)
        self.st_max_age_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.st_max_age_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.st_max_age_label.setObjectName("st_max_age_label")
        self.st_iou_threshold_label = QtWidgets.QLabel(SORTForm)
        self.st_iou_threshold_label.setGeometry(QtCore.QRect(10, 70, 91, 16))
        self.st_iou_threshold_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.st_iou_threshold_label.setObjectName("st_iou_threshold_label")
        self.st_max_age_lineEdit = QtWidgets.QLineEdit(SORTForm)
        self.st_max_age_lineEdit.setGeometry(QtCore.QRect(110, 10, 241, 21))
        self.st_max_age_lineEdit.setObjectName("st_max_age_lineEdit")

        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)

        # custom
        self.loadCFG()
        self.loadST()

        self.save_pushButton.clicked.connect(lambda: self.updateCFG(SORTForm))

        self.retranslateUi(SORTForm)
        QtCore.QMetaObject.connectSlotsByName(SORTForm)


    def retranslateUi(self, SORTForm):
        _translate = QtCore.QCoreApplication.translate
        SORTForm.setWindowTitle(_translate("SORTForm", "SORT"))
        self.save_pushButton.setText(_translate("SORTForm", "Save"))
        self.st_min_hits_label.setText(_translate("SORTForm", "min_hits"))
        self.st_max_age_label.setText(_translate("SORTForm", "max_age"))
        self.st_iou_threshold_label.setText(_translate("SORTForm", "iou_threshold"))


    def loadCFG(self):
        self.mycfg = MyConfigurator()
        self.mycfg.loadTCFG()


    def loadST(self):
        self.st_iou_threshold_lineEdit.setText(str(self.mycfg.tcfg_sort.iou_threshold))
        self.st_max_age_lineEdit.setText(str(self.mycfg.tcfg_sort.max_age))
        self.st_min_hits_lineEdit.setText(str(self.mycfg.tcfg_sort.min_hits))


    def updateCFG(self, SORTForm):
        sort_doc = {"tk_name": "SORT",
                    "max_age": int(self.st_max_age_lineEdit.text()),
                    "min_hits": int(self.st_min_hits_lineEdit.text()),
                    "iou_threshold": float(self.st_iou_threshold_lineEdit.text())}
        centroid_doc = self.mycfg.tcfg_centroid.getDocument()
        deepsort_doc = self.mycfg.tcfg_deepsort.getDocument()
        cfgio = MyCFGIO()
        cfgio.dumpTrackersWithHeader([centroid_doc, sort_doc, deepsort_doc])
        SORTForm.close()

