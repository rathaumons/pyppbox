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
from pyppbox.utils.mytools import getAbsPathFDS, extendPathFDS, normalizePathFDS, joinFPathFull

root_dir = os.path.dirname(__file__)
cfg_dir = joinFPathFull(root_dir, 'cfg')


class Ui_YOLOForm(object):

    def setupUi(self, YOLOForm):
        YOLOForm.setObjectName("YOLOForm")
        YOLOForm.resize(390, 300)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(YOLOForm.sizePolicy().hasHeightForWidth())
        YOLOForm.setSizePolicy(sizePolicy)
        YOLOForm.setMinimumSize(QtCore.QSize(390, 300))
        YOLOForm.setMaximumSize(QtCore.QSize(390, 300))
        self.save_pushButton = QtWidgets.QPushButton(YOLOForm)
        self.save_pushButton.setGeometry(QtCore.QRect(150, 260, 91, 31))
        self.save_pushButton.setObjectName("save_pushButton")
        self.yl_model_res_h_lineEdit = QtWidgets.QLineEdit(YOLOForm)
        self.yl_model_res_h_lineEdit.setGeometry(QtCore.QRect(110, 190, 241, 21))
        self.yl_model_res_h_lineEdit.setObjectName("yl_model_res_h_lineEdit")
        self.yl_model_res_w_label = QtWidgets.QLabel(YOLOForm)
        self.yl_model_res_w_label.setGeometry(QtCore.QRect(10, 160, 91, 16))
        self.yl_model_res_w_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.yl_model_res_w_label.setObjectName("yl_model_res_w_label")
        self.yl_nms_threshold_lineEdit = QtWidgets.QLineEdit(YOLOForm)
        self.yl_nms_threshold_lineEdit.setGeometry(QtCore.QRect(110, 10, 241, 21))
        self.yl_nms_threshold_lineEdit.setObjectName("yl_nms_threshold_lineEdit")
        self.yl_model_cfg_file_pushButton = QtWidgets.QPushButton(YOLOForm)
        self.yl_model_cfg_file_pushButton.setGeometry(QtCore.QRect(360, 100, 21, 24))
        self.yl_model_cfg_file_pushButton.setObjectName("yl_model_cfg_file_pushButton")
        self.yl_model_weights_lineEdit = QtWidgets.QLineEdit(YOLOForm)
        self.yl_model_weights_lineEdit.setGeometry(QtCore.QRect(110, 130, 241, 21))
        self.yl_model_weights_lineEdit.setReadOnly(True)
        self.yl_model_weights_lineEdit.setObjectName("yl_model_weights_lineEdit")
        self.yl_class_file_lineEdit = QtWidgets.QLineEdit(YOLOForm)
        self.yl_class_file_lineEdit.setGeometry(QtCore.QRect(110, 70, 241, 21))
        self.yl_class_file_lineEdit.setReadOnly(True)
        self.yl_class_file_lineEdit.setObjectName("yl_class_file_lineEdit")
        self.yl_conf_threshold_lineEdit = QtWidgets.QLineEdit(YOLOForm)
        self.yl_conf_threshold_lineEdit.setGeometry(QtCore.QRect(110, 40, 241, 21))
        self.yl_conf_threshold_lineEdit.setObjectName("yl_conf_threshold_lineEdit")
        self.yl_repspint_callib_lineEdit = QtWidgets.QLineEdit(YOLOForm)
        self.yl_repspint_callib_lineEdit.setGeometry(QtCore.QRect(110, 220, 241, 21))
        self.yl_repspint_callib_lineEdit.setObjectName("yl_repspint_callib_lineEdit")
        self.yl_class_file_pushButton = QtWidgets.QPushButton(YOLOForm)
        self.yl_class_file_pushButton.setGeometry(QtCore.QRect(360, 70, 21, 24))
        self.yl_class_file_pushButton.setObjectName("yl_class_file_pushButton")
        self.yl_nms_threshold_label = QtWidgets.QLabel(YOLOForm)
        self.yl_nms_threshold_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.yl_nms_threshold_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.yl_nms_threshold_label.setObjectName("yl_nms_threshold_label")
        self.yl_model_weights_pushButton = QtWidgets.QPushButton(YOLOForm)
        self.yl_model_weights_pushButton.setGeometry(QtCore.QRect(360, 130, 21, 24))
        self.yl_model_weights_pushButton.setObjectName("yl_model_weights_pushButton")
        self.yl_model_cfg_file_label = QtWidgets.QLabel(YOLOForm)
        self.yl_model_cfg_file_label.setGeometry(QtCore.QRect(10, 100, 91, 16))
        self.yl_model_cfg_file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.yl_model_cfg_file_label.setObjectName("yl_model_cfg_file_label")
        self.yl_conf_threshold_label = QtWidgets.QLabel(YOLOForm)
        self.yl_conf_threshold_label.setGeometry(QtCore.QRect(10, 40, 91, 16))
        self.yl_conf_threshold_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.yl_conf_threshold_label.setObjectName("yl_conf_threshold_label")
        self.yl_class_file_label = QtWidgets.QLabel(YOLOForm)
        self.yl_class_file_label.setGeometry(QtCore.QRect(10, 70, 91, 16))
        self.yl_class_file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.yl_class_file_label.setObjectName("yl_class_file_label")
        self.yl_model_res_h_label = QtWidgets.QLabel(YOLOForm)
        self.yl_model_res_h_label.setGeometry(QtCore.QRect(10, 190, 91, 16))
        self.yl_model_res_h_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.yl_model_res_h_label.setObjectName("yl_model_res_h_label")
        self.yl_repspint_callib_label = QtWidgets.QLabel(YOLOForm)
        self.yl_repspint_callib_label.setGeometry(QtCore.QRect(10, 220, 91, 16))
        self.yl_repspint_callib_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.yl_repspint_callib_label.setObjectName("yl_repspint_callib_label")
        self.yl_model_cfg_file_lineEdit = QtWidgets.QLineEdit(YOLOForm)
        self.yl_model_cfg_file_lineEdit.setGeometry(QtCore.QRect(110, 100, 241, 21))
        self.yl_model_cfg_file_lineEdit.setReadOnly(True)
        self.yl_model_cfg_file_lineEdit.setObjectName("yl_model_cfg_file_lineEdit")
        self.yl_model_weights_label = QtWidgets.QLabel(YOLOForm)
        self.yl_model_weights_label.setGeometry(QtCore.QRect(10, 130, 91, 16))
        self.yl_model_weights_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.yl_model_weights_label.setObjectName("yl_model_weights_label")
        self.yl_model_res_w_lineEdit = QtWidgets.QLineEdit(YOLOForm)
        self.yl_model_res_w_lineEdit.setGeometry(QtCore.QRect(110, 160, 241, 21))
        self.yl_model_res_w_lineEdit.setObjectName("yl_model_res_w_lineEdit")

        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)

        # custom 
        self.loadCFG()
        self.loadYL()

        self.yl_class_file_pushButton.clicked.connect(self.browseClassFile)
        self.yl_model_cfg_file_pushButton.clicked.connect(self.browseModelCFG)
        self.yl_model_weights_pushButton.clicked.connect(self.browseModelWeights)
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(YOLOForm))

        self.retranslateUi(YOLOForm)
        QtCore.QMetaObject.connectSlotsByName(YOLOForm)


    def retranslateUi(self, YOLOForm):
        _translate = QtCore.QCoreApplication.translate
        YOLOForm.setWindowTitle(_translate("YOLOForm", "YOLO"))
        self.yl_model_res_w_label.setText(_translate("YOLOForm", "model_res_w"))
        self.yl_model_cfg_file_pushButton.setText(_translate("YOLOForm", "..."))
        self.yl_class_file_pushButton.setText(_translate("YOLOForm", "..."))
        self.yl_nms_threshold_label.setText(_translate("YOLOForm", "nms_threshold"))
        self.yl_model_weights_pushButton.setText(_translate("YOLOForm", "..."))
        self.yl_model_cfg_file_label.setText(_translate("YOLOForm", "model_cfg_file"))
        self.yl_conf_threshold_label.setText(_translate("YOLOForm", "conf_threshold"))
        self.yl_class_file_label.setText(_translate("YOLOForm", "class_file_file"))
        self.yl_model_res_h_label.setText(_translate("YOLOForm", "model_res_h"))
        self.yl_repspint_callib_label.setText(_translate("YOLOForm", "repspoint_callib"))
        self.yl_model_weights_label.setText(_translate("YOLOForm", "model_weights"))
        self.save_pushButton.setText(_translate("YOLOForm", "Save"))


    def loadCFG(self):
        self.mycfg = MyConfigurator()
        self.mycfg.loadDCFG()


    def loadYL(self):
        self.yl_nms_threshold_lineEdit.setText(str(self.mycfg.dcfg_yolo.nms_threshold))
        self.yl_conf_threshold_lineEdit.setText(str(self.mycfg.dcfg_yolo.conf_threshold))
        self.yl_model_cfg_file_lineEdit.setText(getAbsPathFDS(self.mycfg.dcfg_yolo.model_cfg_file))
        self.yl_class_file_lineEdit.setText(getAbsPathFDS(self.mycfg.dcfg_yolo.class_file))
        self.yl_model_weights_lineEdit.setText(getAbsPathFDS(self.mycfg.dcfg_yolo.model_weights))
        (res_w, res_h) = self.mycfg.dcfg_yolo.model_resolution
        self.yl_model_res_h_lineEdit.setText(str(res_h))
        self.yl_model_res_w_lineEdit.setText(str(res_w))
        self.yl_repspint_callib_lineEdit.setText(str(self.mycfg.dcfg_yolo.repspoint_callibration))


    def updateCFG(self, YOLOForm):
        yolo_doc = {"dt_name": "YOLO",
                    "nms_threshold": float(self.yl_nms_threshold_lineEdit.text()),
                    "conf_threshold": float(self.yl_conf_threshold_lineEdit.text()),
                    "class_file": normalizePathFDS(root_dir, self.yl_class_file_lineEdit.text()),
                    "model_cfg_file": normalizePathFDS(root_dir, self.yl_model_cfg_file_lineEdit.text()),
                    "model_weights": normalizePathFDS(root_dir, self.yl_model_weights_lineEdit.text()),
                    "model_resolution_width": int(self.yl_model_res_w_lineEdit.text()),
                    "model_resolution_height": int(self.yl_model_res_h_lineEdit.text()),
                    "repspoint_callibration": float(self.yl_repspint_callib_lineEdit.text())}
        openpose_doc = self.mycfg.dcfg_openpose.getDocument()
        gt_doc = self.mycfg.dcfg_gt.getDocument()
        cfgio = MyCFGIO()
        cfgio.dumpDetectorsWithHeader([yolo_doc, openpose_doc, gt_doc])
        YOLOForm.close()


    def browseClassFile(self):
        default_path = extendPathFDS(root_dir, 'dt_yolocv')
        file_filter = "COCO (*.names)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "COCO class file", default_path, file_filter)
        if source_file:
            self.yl_class_file_lineEdit.setText(source_file)


    def browseModelCFG(self):
        default_path = extendPathFDS(root_dir, 'dt_yolocv')
        cfg_filter = "Configurator (*.cfg)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Model configurator file", default_path, cfg_filter)
        if source_file:
            self.yl_model_cfg_file_lineEdit.setText(source_file)


    def browseModelWeights(self):
        default_path = extendPathFDS(root_dir, 'dt_yolocv')
        weight_filter = "Weights (*.weights)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Model weights file", default_path, weight_filter)
        if source_file:
            self.yl_model_weights_lineEdit.setText(source_file)
