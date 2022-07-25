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
from pyppbox.utils.mytools import getBool, getAbsPathFDS, extendPathFDS, normalizePathFDS, joinFPathFull

root_dir = os.path.dirname(__file__)
cfg_dir = joinFPathFull(root_dir, 'cfg')


class Ui_OpenPoseForm(object):

    def setupUi(self, OpenPoseForm):
        OpenPoseForm.setObjectName("OpenPoseForm")
        OpenPoseForm.resize(390, 270)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(OpenPoseForm.sizePolicy().hasHeightForWidth())
        OpenPoseForm.setSizePolicy(sizePolicy)
        OpenPoseForm.setMinimumSize(QtCore.QSize(390, 270))
        OpenPoseForm.setMaximumSize(QtCore.QSize(390, 270))
        self.op_model_pose_label = QtWidgets.QLabel(OpenPoseForm)
        self.op_model_pose_label.setGeometry(QtCore.QRect(10, 40, 91, 16))
        self.op_model_pose_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.op_model_pose_label.setObjectName("op_model_pose_label")
        self.op_model_res_label = QtWidgets.QLabel(OpenPoseForm)
        self.op_model_res_label.setGeometry(QtCore.QRect(10, 100, 91, 16))
        self.op_model_res_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.op_model_res_label.setObjectName("op_model_res_label")
        self.op_output_res_label = QtWidgets.QLabel(OpenPoseForm)
        self.op_output_res_label.setGeometry(QtCore.QRect(10, 130, 91, 16))
        self.op_output_res_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.op_output_res_label.setObjectName("op_output_res_label")
        self.op_people_max_label = QtWidgets.QLabel(OpenPoseForm)
        self.op_people_max_label.setGeometry(QtCore.QRect(10, 160, 91, 16))
        self.op_people_max_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.op_people_max_label.setObjectName("op_people_max_label")
        self.op_disable_blending_comboBox = QtWidgets.QComboBox(OpenPoseForm)
        self.op_disable_blending_comboBox.setGeometry(QtCore.QRect(110, 190, 241, 22))
        self.op_disable_blending_comboBox.setObjectName("op_disable_blending_comboBox")
        self.op_disable_blending_comboBox.addItem("")
        self.op_disable_blending_comboBox.addItem("")
        self.op_people_max_lineEdit = QtWidgets.QLineEdit(OpenPoseForm)
        self.op_people_max_lineEdit.setGeometry(QtCore.QRect(110, 160, 241, 21))
        self.op_people_max_lineEdit.setObjectName("op_people_max_lineEdit")
        self.op_model_pose_lineEdit = QtWidgets.QLineEdit(OpenPoseForm)
        self.op_model_pose_lineEdit.setGeometry(QtCore.QRect(110, 40, 241, 21))
        self.op_model_pose_lineEdit.setObjectName("op_model_pose_lineEdit")
        self.op_hand_comboBox = QtWidgets.QComboBox(OpenPoseForm)
        self.op_hand_comboBox.setGeometry(QtCore.QRect(110, 10, 241, 22))
        self.op_hand_comboBox.setObjectName("op_hand_comboBox")
        self.op_hand_comboBox.addItem("")
        self.op_hand_comboBox.addItem("")
        self.op_hand_label = QtWidgets.QLabel(OpenPoseForm)
        self.op_hand_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.op_hand_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.op_hand_label.setObjectName("op_hand_label")
        self.op_output_res_lineEdit = QtWidgets.QLineEdit(OpenPoseForm)
        self.op_output_res_lineEdit.setGeometry(QtCore.QRect(110, 130, 241, 21))
        self.op_output_res_lineEdit.setObjectName("op_output_res_lineEdit")
        self.op_model_folder_pushButton = QtWidgets.QPushButton(OpenPoseForm)
        self.op_model_folder_pushButton.setGeometry(QtCore.QRect(360, 70, 21, 24))
        self.op_model_folder_pushButton.setObjectName("op_model_folder_pushButton")
        self.op_model_folder_lineEdit = QtWidgets.QLineEdit(OpenPoseForm)
        self.op_model_folder_lineEdit.setGeometry(QtCore.QRect(110, 70, 241, 21))
        self.op_model_folder_lineEdit.setReadOnly(True)
        self.op_model_folder_lineEdit.setObjectName("op_model_folder_lineEdit")
        self.op_model_folder_label = QtWidgets.QLabel(OpenPoseForm)
        self.op_model_folder_label.setGeometry(QtCore.QRect(10, 70, 91, 16))
        self.op_model_folder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.op_model_folder_label.setObjectName("op_model_folder_label")
        self.op_model_res_lineEdit = QtWidgets.QLineEdit(OpenPoseForm)
        self.op_model_res_lineEdit.setGeometry(QtCore.QRect(110, 100, 241, 21))
        self.op_model_res_lineEdit.setObjectName("op_model_res_lineEdit")
        self.op_disable_blending_label = QtWidgets.QLabel(OpenPoseForm)
        self.op_disable_blending_label.setGeometry(QtCore.QRect(10, 190, 91, 16))
        self.op_disable_blending_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.op_disable_blending_label.setObjectName("op_disable_blending_label")
        self.save_pushButton = QtWidgets.QPushButton(OpenPoseForm)
        self.save_pushButton.setGeometry(QtCore.QRect(150, 230, 91, 31))

        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)
        self.save_pushButton.setObjectName("save_pushButton")

        # custom 
        self.loadCFG()
        self.loadOP()

        self.op_model_folder_pushButton.clicked.connect(self.browseModelFolder)
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(OpenPoseForm))

        self.retranslateUi(OpenPoseForm)
        QtCore.QMetaObject.connectSlotsByName(OpenPoseForm)


    def retranslateUi(self, OpenPoseForm):
        _translate = QtCore.QCoreApplication.translate
        OpenPoseForm.setWindowTitle(_translate("OpenPoseForm", "OpenPose"))
        self.op_model_pose_label.setText(_translate("OpenPoseForm", "model_pose"))
        self.op_model_res_label.setText(_translate("OpenPoseForm", "model_resolution"))
        self.op_output_res_label.setText(_translate("OpenPoseForm", "output_res"))
        self.op_people_max_label.setText(_translate("OpenPoseForm", "people_max"))
        self.op_disable_blending_comboBox.setItemText(0, _translate("OpenPoseForm", "True"))
        self.op_disable_blending_comboBox.setItemText(1, _translate("OpenPoseForm", "False"))
        self.op_hand_comboBox.setItemText(0, _translate("OpenPoseForm", "True"))
        self.op_hand_comboBox.setItemText(1, _translate("OpenPoseForm", "False"))
        self.op_hand_label.setText(_translate("OpenPoseForm", "hand"))
        self.op_model_folder_pushButton.setText(_translate("OpenPoseForm", "..."))
        self.op_model_folder_label.setText(_translate("OpenPoseForm", "model_folder"))
        self.op_disable_blending_label.setText(_translate("OpenPoseForm", "disable_blending"))
        self.save_pushButton.setText(_translate("OpenPoseForm", "Save"))


    def loadCFG(self):
        self.mycfg = MyConfigurator()
        self.mycfg.loadDCFG()


    def loadOP(self):

        if self.mycfg.dcfg_openpose.disable_blending is True:
            self.op_disable_blending_comboBox.setCurrentIndex(0)
        else:
            self.op_disable_blending_comboBox.setCurrentIndex(1)

        if self.mycfg.dcfg_openpose.hand is True:
            self.op_hand_comboBox.setCurrentIndex(0)
        else:
            self.op_hand_comboBox.setCurrentIndex(1)

        self.op_model_folder_lineEdit.setText(getAbsPathFDS(self.mycfg.dcfg_openpose.model_folder))
        self.op_model_pose_lineEdit.setText(self.mycfg.dcfg_openpose.model_pose)
        self.op_model_res_lineEdit.setText(self.mycfg.dcfg_openpose.model_resolution)
        self.op_output_res_lineEdit.setText(self.mycfg.dcfg_openpose.output_resolution)
        self.op_people_max_lineEdit.setText(str(self.mycfg.dcfg_openpose.number_people_max))


    def updateCFG(self, OpenPoseForm):
        openpose_doc = {"dt_name": "OpenPose",
                        "hand": getBool(self.op_hand_comboBox.currentText()),
                        "model_pose": self.op_model_pose_lineEdit.text(),
                        "model_folder": normalizePathFDS(root_dir, self.op_model_folder_lineEdit.text()),
                        "model_resolution": self.op_model_res_lineEdit.text(),
                        "output_resolution": self.op_output_res_lineEdit.text(),
                        "number_people_max": self.op_people_max_lineEdit.text(),
                        "disable_blending": getBool(self.op_disable_blending_comboBox.currentText())}
        yolo_doc = self.mycfg.dcfg_yolo.getDocument()
        gt_doc = self.mycfg.dcfg_gt.getDocument()
        cfgio = MyCFGIO()
        cfgio.dumpDetectorsWithHeader([yolo_doc, openpose_doc, gt_doc])
        OpenPoseForm.close()
        

    def browseModelFolder(self):
        default_path = extendPathFDS(root_dir, 'dt_openpose/models')
        model_path = QtWidgets.QFileDialog.getExistingDirectory(None, "Model folder (face, hand, pose)", default_path)
        if model_path:
            self.op_model_folder_lineEdit.setText(model_path)

