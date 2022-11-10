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
from pyppbox.utils.mytools import joinFPathFull

root_dir = os.path.dirname(__file__)
cfg_dir = joinFPathFull(root_dir, 'cfg')


class Ui_CentroidForm(object):

    def setupUi(self, CentroidForm):
        CentroidForm.setObjectName("CentroidForm")
        CentroidForm.resize(390, 90)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(CentroidForm.sizePolicy().hasHeightForWidth())
        CentroidForm.setSizePolicy(sizePolicy)
        CentroidForm.setMinimumSize(QtCore.QSize(390, 90))
        CentroidForm.setMaximumSize(QtCore.QSize(390, 90))
        self.save_pushButton = QtWidgets.QPushButton(CentroidForm)
        self.save_pushButton.setGeometry(QtCore.QRect(150, 50, 91, 31))
        self.save_pushButton.setObjectName("save_pushButton")
        self.ct_max_distance_lineEdit = QtWidgets.QLineEdit(CentroidForm)
        self.ct_max_distance_lineEdit.setGeometry(QtCore.QRect(110, 10, 241, 21))
        self.ct_max_distance_lineEdit.setObjectName("ct_max_distance_lineEdit")
        self.ct_max_distance_label = QtWidgets.QLabel(CentroidForm)
        self.ct_max_distance_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.ct_max_distance_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ct_max_distance_label.setObjectName("ct_max_distance_label")

        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)

        # custom 
        self.loadCFG()
        self.loadCT()

        self.save_pushButton.clicked.connect(lambda: self.updateCFG(CentroidForm))

        self.retranslateUi(CentroidForm)
        QtCore.QMetaObject.connectSlotsByName(CentroidForm)


    def retranslateUi(self, CentroidForm):
        _translate = QtCore.QCoreApplication.translate
        CentroidForm.setWindowTitle(_translate("CentroidForm", "Centroid"))
        self.ct_max_distance_label.setText(_translate("CentroidForm", "max_distance"))
        self.save_pushButton.setText(_translate("CentroidForm", "Save"))


    def loadCFG(self):
        self.mycfg = MyConfigurator()
        self.mycfg.loadTCFG()


    def loadCT(self):
        self.ct_max_distance_lineEdit.setText(str(self.mycfg.tcfg_centroid.max_distance))


    def updateCFG(self, CentroidForm):
        centroid_doc = {"tk_name": "Centroid",
                        "max_distance": int(self.ct_max_distance_lineEdit.text())}
        sort_doc = self.mycfg.tcfg_sort.getDocument()
        deepsort_doc = self.mycfg.tcfg_deepsort.getDocument()
        cfgio = MyCFGIO()
        cfgio.dumpTrackersWithHeader([centroid_doc, sort_doc, deepsort_doc])
        CentroidForm.close()
        