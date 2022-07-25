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
from pyppbox.utils.mytools import replaceLine, normalizePathFDS, getAbsPathFDS, joinFPathFull

root_dir = os.path.dirname(__file__)
cfg_dir = joinFPathFull(root_dir, 'cfg')


class Ui_GTForm(object):

    def setupUi(self, GTForm):
        GTForm.setObjectName("GTForm")
        GTForm.resize(390, 90)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(GTForm.sizePolicy().hasHeightForWidth())
        GTForm.setSizePolicy(sizePolicy)
        GTForm.setMinimumSize(QtCore.QSize(390, 90))
        GTForm.setMaximumSize(QtCore.QSize(390, 90))
        self.gt_browse_pushButton = QtWidgets.QPushButton(GTForm)
        self.gt_browse_pushButton.setGeometry(QtCore.QRect(360, 10, 21, 24))
        self.gt_browse_pushButton.setObjectName("gt_browse_pushButton")
        self.save_pushButton = QtWidgets.QPushButton(GTForm)
        self.save_pushButton.setGeometry(QtCore.QRect(150, 50, 91, 31))
        self.save_pushButton.setObjectName("save_pushButton")
        self.gt_file_lineEdit = QtWidgets.QLineEdit(GTForm)
        self.gt_file_lineEdit.setGeometry(QtCore.QRect(110, 10, 241, 21))
        self.gt_file_lineEdit.setObjectName("gt_file_lineEdit")
        self.gt_file_label = QtWidgets.QLabel(GTForm)
        self.gt_file_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.gt_file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.gt_file_label.setObjectName("gt_file_label")

        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)

        # custom 
        self.loadCFG()
        self.loadGT()

        self.gt_browse_pushButton.clicked.connect(self.browseInputFile)
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(GTForm))

        self.retranslateUi(GTForm)
        QtCore.QMetaObject.connectSlotsByName(GTForm)


    def retranslateUi(self, GTForm):
        _translate = QtCore.QCoreApplication.translate
        GTForm.setWindowTitle(_translate("GTForm", "GT"))
        self.gt_file_label.setText(_translate("GTForm", "gt_file"))
        self.gt_browse_pushButton.setText(_translate("GTForm", "..."))
        self.save_pushButton.setText(_translate("GTForm", "Save"))


    def loadCFG(self):
        self.mycfg = MyConfigurator()
        self.mycfg.loadDCFG()


    def loadGT(self):
        self.gt_file_lineEdit.setText(getAbsPathFDS(self.mycfg.dcfg_gt.gt_file))


    def browseInputFile(self):
        default_path = joinFPathFull(root_dir, 'tmp/gt')
        file_filter = "TXT (*.txt)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Input GT file", default_path, file_filter)
        if source_file:
            self.gt_file_lineEdit.setText(source_file)


    def updateCFG(self, GTForm):
        gt_doc = {"dt_name": "GT",
                  "gt_file": normalizePathFDS(root_dir, self.gt_file_lineEdit.text()),
                  "input_gt_map_file": normalizePathFDS(root_dir, self.mycfg.dcfg_gt.input_gt_map_file)}
        yolo_doc = self.mycfg.dcfg_yolo.getDocument()
        openpose_doc = self.mycfg.dcfg_openpose.getDocument()
        cfgio = MyCFGIO()
        cfgio.dumpDetectorsWithHeader([yolo_doc, openpose_doc, gt_doc])
        GTForm.close()
        
