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

from PyQt6 import QtCore, QtGui, QtWidgets
from pyppbox.config import MyConfigurator as MyGlobalCFG
from pyppbox.config import MyCFGIO as GlobalCFGIO
from pyppbox.localconfig import MyLocalConfigurator as MyLocalCFG
from pyppbox.localconfig import MyCFGIO as LocalCFGIO
from pyppbox.utils.mytools import getAbsPathFDS, normalizePathFDS, getAncestorDir

root_dir = os.path.dirname(__file__)


class Ui_YOLOUTLTForm(object):

    def __init__(self, cfg_mode, cfg_dir):
        self.cfg_dir = cfg_dir
        if cfg_mode == 0:
            self.mycfg = MyGlobalCFG()
            self.cfgIO = GlobalCFGIO()
        else:
            self.mycfg = MyLocalCFG(self.cfg_dir)
            self.cfgIO = LocalCFGIO(self.cfg_dir)
    
    def setupUi(self, YOLOUTLTForm):
        YOLOUTLTForm.setObjectName("YOLOUTLTForm")
        YOLOUTLTForm.resize(390, 450)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(YOLOUTLTForm.sizePolicy().hasHeightForWidth())
        YOLOUTLTForm.setSizePolicy(sizePolicy)
        YOLOUTLTForm.setMinimumSize(QtCore.QSize(390, 450))
        YOLOUTLTForm.setMaximumSize(QtCore.QSize(390, 450))
        self.max_det_lineEdit = QtWidgets.QLineEdit(parent=YOLOUTLTForm)
        self.max_det_lineEdit.setGeometry(QtCore.QRect(110, 190, 241, 21))
        self.max_det_lineEdit.setObjectName("max_det_lineEdit")
        self.devices_label = QtWidgets.QLabel(parent=YOLOUTLTForm)
        self.devices_label.setGeometry(QtCore.QRect(10, 160, 91, 16))
        self.devices_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.devices_label.setObjectName("devices_label")
        self.conf_lineEdit = QtWidgets.QLineEdit(parent=YOLOUTLTForm)
        self.conf_lineEdit.setGeometry(QtCore.QRect(110, 10, 241, 21))
        self.conf_lineEdit.setObjectName("conf_lineEdit")
        self.imgsz_lineEdit = QtWidgets.QLineEdit(parent=YOLOUTLTForm)
        self.imgsz_lineEdit.setGeometry(QtCore.QRect(110, 70, 241, 21))
        self.imgsz_lineEdit.setText("")
        self.imgsz_lineEdit.setObjectName("imgsz_lineEdit")
        self.iou_lineEdit = QtWidgets.QLineEdit(parent=YOLOUTLTForm)
        self.iou_lineEdit.setGeometry(QtCore.QRect(110, 40, 241, 21))
        self.iou_lineEdit.setObjectName("iou_lineEdit")
        self.conf_label = QtWidgets.QLabel(parent=YOLOUTLTForm)
        self.conf_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.conf_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.conf_label.setObjectName("conf_label")
        self.model_file_pushButton = QtWidgets.QPushButton(parent=YOLOUTLTForm)
        self.model_file_pushButton.setGeometry(QtCore.QRect(360, 340, 21, 24))
        self.model_file_pushButton.setObjectName("model_file_pushButton")
        self.classes_label = QtWidgets.QLabel(parent=YOLOUTLTForm)
        self.classes_label.setGeometry(QtCore.QRect(10, 100, 91, 16))
        self.classes_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.classes_label.setObjectName("classes_label")
        self.iou_label = QtWidgets.QLabel(parent=YOLOUTLTForm)
        self.iou_label.setGeometry(QtCore.QRect(10, 40, 91, 16))
        self.iou_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.iou_label.setObjectName("iou_label")
        self.imgsz_label = QtWidgets.QLabel(parent=YOLOUTLTForm)
        self.imgsz_label.setGeometry(QtCore.QRect(10, 70, 91, 16))
        self.imgsz_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.imgsz_label.setObjectName("imgsz_label")
        self.max_det_label = QtWidgets.QLabel(parent=YOLOUTLTForm)
        self.max_det_label.setGeometry(QtCore.QRect(10, 190, 91, 16))
        self.max_det_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.max_det_label.setObjectName("max_det_label")
        self.hide_labels_label = QtWidgets.QLabel(parent=YOLOUTLTForm)
        self.hide_labels_label.setGeometry(QtCore.QRect(10, 220, 91, 16))
        self.hide_labels_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.hide_labels_label.setObjectName("hide_labels_label")
        self.classes_lineEdit = QtWidgets.QLineEdit(parent=YOLOUTLTForm)
        self.classes_lineEdit.setGeometry(QtCore.QRect(110, 100, 241, 21))
        self.classes_lineEdit.setObjectName("classes_lineEdit")
        self.boxes_label = QtWidgets.QLabel(parent=YOLOUTLTForm)
        self.boxes_label.setGeometry(QtCore.QRect(10, 130, 91, 16))
        self.boxes_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.boxes_label.setObjectName("boxes_label")
        self.device_lineEdit = QtWidgets.QLineEdit(parent=YOLOUTLTForm)
        self.device_lineEdit.setGeometry(QtCore.QRect(110, 160, 241, 21))
        self.device_lineEdit.setObjectName("device_lineEdit")
        self.save_pushButton = QtWidgets.QPushButton(parent=YOLOUTLTForm)
        self.save_pushButton.setGeometry(QtCore.QRect(150, 410, 91, 31))
        self.save_pushButton.setObjectName("save_pushButton")
        self.visualize_label = QtWidgets.QLabel(parent=YOLOUTLTForm)
        self.visualize_label.setGeometry(QtCore.QRect(10, 310, 91, 16))
        self.visualize_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.visualize_label.setObjectName("visualize_label")
        self.hide_conf_label = QtWidgets.QLabel(parent=YOLOUTLTForm)
        self.hide_conf_label.setGeometry(QtCore.QRect(10, 250, 91, 16))
        self.hide_conf_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.hide_conf_label.setObjectName("hide_conf_label")
        self.model_file_label = QtWidgets.QLabel(parent=YOLOUTLTForm)
        self.model_file_label.setGeometry(QtCore.QRect(10, 340, 91, 16))
        self.model_file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.model_file_label.setObjectName("model_file_label")
        self.model_file_lineEdit = QtWidgets.QLineEdit(parent=YOLOUTLTForm)
        self.model_file_lineEdit.setGeometry(QtCore.QRect(110, 340, 241, 21))
        self.model_file_lineEdit.setObjectName("model_file_lineEdit")
        self.model_file_lineEdit.setReadOnly(True)
        self.line_width_lineEdit = QtWidgets.QLineEdit(parent=YOLOUTLTForm)
        self.line_width_lineEdit.setGeometry(QtCore.QRect(110, 280, 241, 21))
        self.line_width_lineEdit.setObjectName("line_width_lineEdit")
        self.line_width_label = QtWidgets.QLabel(parent=YOLOUTLTForm)
        self.line_width_label.setGeometry(QtCore.QRect(10, 280, 91, 16))
        self.line_width_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.line_width_label.setObjectName("line_width_label")
        self.repspint_callib_label = QtWidgets.QLabel(parent=YOLOUTLTForm)
        self.repspint_callib_label.setGeometry(QtCore.QRect(10, 370, 91, 16))
        self.repspint_callib_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.repspint_callib_label.setObjectName("repspint_callib_label")
        self.repspint_callib_lineEdit = QtWidgets.QLineEdit(parent=YOLOUTLTForm)
        self.repspint_callib_lineEdit.setGeometry(QtCore.QRect(110, 370, 241, 21))
        self.repspint_callib_lineEdit.setObjectName("repspint_callib_lineEdit")
        self.visualize_comboBox = QtWidgets.QComboBox(parent=YOLOUTLTForm)
        self.visualize_comboBox.setGeometry(QtCore.QRect(110, 310, 241, 21))
        self.visualize_comboBox.setObjectName("visualize_comboBox")
        self.visualize_comboBox.addItem("")
        self.visualize_comboBox.addItem("")
        self.hide_conf_comboBox = QtWidgets.QComboBox(parent=YOLOUTLTForm)
        self.hide_conf_comboBox.setGeometry(QtCore.QRect(110, 250, 241, 21))
        self.hide_conf_comboBox.setObjectName("hide_conf_comboBox")
        self.hide_conf_comboBox.addItem("")
        self.hide_conf_comboBox.addItem("")
        self.hide_labels_comboBox = QtWidgets.QComboBox(parent=YOLOUTLTForm)
        self.hide_labels_comboBox.setGeometry(QtCore.QRect(110, 220, 241, 21))
        self.hide_labels_comboBox.setObjectName("hide_labels_comboBox")
        self.hide_labels_comboBox.addItem("")
        self.hide_labels_comboBox.addItem("")
        self.boxes_comboBox = QtWidgets.QComboBox(parent=YOLOUTLTForm)
        self.boxes_comboBox.setGeometry(QtCore.QRect(110, 130, 241, 21))
        self.boxes_comboBox.setObjectName("boxes_comboBox")
        self.boxes_comboBox.addItem("")
        self.boxes_comboBox.addItem("")
        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)
        self.save_pushButton.setFocus()
        # custom 
        self.loadYLUTLT()
        self.model_file_pushButton.clicked.connect(self.browseModelFile)
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(YOLOUTLTForm))
        # others
        self.retranslateUi(YOLOUTLTForm)
        QtCore.QMetaObject.connectSlotsByName(YOLOUTLTForm)

    def retranslateUi(self, YOLOUTLTForm):
        _translate = QtCore.QCoreApplication.translate
        YOLOUTLTForm.setWindowTitle(_translate("YOLOUTLTForm", "YOLO Ultralytics"))
        self.max_det_lineEdit.setPlaceholderText(_translate("YOLOUTLTForm", "100"))
        self.devices_label.setText(_translate("YOLOUTLTForm", "device"))
        self.conf_lineEdit.setPlaceholderText(_translate("YOLOUTLTForm", "0.5"))
        self.imgsz_lineEdit.setPlaceholderText(_translate("YOLOUTLTForm", "416"))
        self.iou_lineEdit.setPlaceholderText(_translate("YOLOUTLTForm", "0.7"))
        self.conf_label.setText(_translate("YOLOUTLTForm", "conf"))
        self.model_file_pushButton.setText(_translate("YOLOUTLTForm", "..."))
        self.classes_label.setText(_translate("YOLOUTLTForm", "classes"))
        self.iou_label.setText(_translate("YOLOUTLTForm", "iou"))
        self.imgsz_label.setText(_translate("YOLOUTLTForm", "imgsz"))
        self.max_det_label.setText(_translate("YOLOUTLTForm", "max_det"))
        self.hide_labels_label.setText(_translate("YOLOUTLTForm", "hide_labels"))
        self.classes_lineEdit.setPlaceholderText(_translate("YOLOUTLTForm", "0"))
        self.boxes_label.setText(_translate("YOLOUTLTForm", "boxes"))
        self.device_lineEdit.setPlaceholderText(_translate("YOLOUTLTForm", "0"))
        self.save_pushButton.setText(_translate("YOLOUTLTForm", "Save"))
        self.visualize_label.setText(_translate("YOLOUTLTForm", "visualize"))
        self.hide_conf_label.setText(_translate("YOLOUTLTForm", "hide_conf"))
        self.model_file_label.setText(_translate("YOLOUTLTForm", "model_file"))
        self.line_width_lineEdit.setPlaceholderText(_translate("YOLOUTLTForm", "500"))
        self.line_width_label.setText(_translate("YOLOUTLTForm", "line_width"))
        self.repspint_callib_label.setText(_translate("YOLOUTLTForm", "repspoint_callib"))
        self.repspint_callib_lineEdit.setPlaceholderText(_translate("YOLOUTLTForm", "0.25"))
        self.visualize_comboBox.setItemText(0, _translate("YOLOUTLTForm", "True"))
        self.visualize_comboBox.setItemText(1, _translate("YOLOUTLTForm", "False"))
        self.hide_conf_comboBox.setItemText(0, _translate("YOLOUTLTForm", "True"))
        self.hide_conf_comboBox.setItemText(1, _translate("YOLOUTLTForm", "False"))
        self.hide_labels_comboBox.setItemText(0, _translate("YOLOUTLTForm", "True"))
        self.hide_labels_comboBox.setItemText(1, _translate("YOLOUTLTForm", "False"))
        self.boxes_comboBox.setItemText(0, _translate("YOLOUTLTForm", "True"))
        self.boxes_comboBox.setItemText(1, _translate("YOLOUTLTForm", "False"))

    def loadYLUTLT(self):
        self.mycfg.loadDCFG()
        self.conf_lineEdit.setText(str(self.mycfg.dcfg_yolopt.conf))
        self.iou_lineEdit.setText(str(self.mycfg.dcfg_yolopt.iou))
        self.imgsz_lineEdit.setText(str(self.mycfg.dcfg_yolopt.imgsz))
        self.classes_lineEdit.setText(str(self.mycfg.dcfg_yolopt.classes))
        self.device_lineEdit.setText(str(self.mycfg.dcfg_yolopt.device))
        self.max_det_lineEdit.setText(str(self.mycfg.dcfg_yolopt.max_det))
        self.line_width_lineEdit.setText(str(self.mycfg.dcfg_yolopt.line_width))
        self.model_file_lineEdit.setText(getAbsPathFDS(self.mycfg.dcfg_yolopt.model_file))
        self.repspint_callib_lineEdit.setText(str(self.mycfg.dcfg_yolopt.repspoint_callibration))
        self.loadComboBoxes()

    def loadComboBoxes(self):
        if self.mycfg.dcfg_yolopt.boxes is True:
            self.boxes_comboBox.setCurrentIndex(0)
        else:
            self.boxes_comboBox.setCurrentIndex(1)
        if self.mycfg.dcfg_yolopt.hide_labels is True:
            self.hide_labels_comboBox.setCurrentIndex(0)
        else:
            self.hide_labels_comboBox.setCurrentIndex(1)
        if self.mycfg.dcfg_yolopt.hide_conf is True:
            self.hide_conf_comboBox.setCurrentIndex(0)
        else:
            self.hide_conf_comboBox.setCurrentIndex(1)
        if self.mycfg.dcfg_yolopt.visualize is True:
            self.visualize_comboBox.setCurrentIndex(0)
        else:
            self.visualize_comboBox.setCurrentIndex(1)

    def updateCFG(self, YOLOUTLTForm):
        yolo_utlt_doc = {
            "dt_name": "YOLO_Ultralytics",
            "conf": float(self.conf_lineEdit.text()),
            "iou": float(self.iou_lineEdit.text()),
            "imgsz": int(self.imgsz_lineEdit.text()),
            "classes": int(self.classes_lineEdit.text()),
            "boxes": self.boxes_comboBox.currentText(),
            "device": int(self.device_lineEdit.text()),
            "max_det": int(self.max_det_lineEdit.text()),
            "hide_labels": self.hide_labels_comboBox.currentText(),
            "hide_conf": self.hide_conf_comboBox.currentText(),
            "line_width": int(self.line_width_lineEdit.text()),
            "visualize": self.visualize_comboBox.currentText(),
            "model_file": normalizePathFDS(root_dir, self.model_file_lineEdit.text()),
            "repspoint_callibration": float(self.repspint_callib_lineEdit.text())
        }
        yolo_doc = self.mycfg.dcfg_yolocv.getDocument()
        gt_doc = self.mycfg.dcfg_gt.getDocument()
        self.cfgIO.dumpDetectorsWithHeader([yolo_doc, yolo_utlt_doc, gt_doc])
        YOLOUTLTForm.close()

    def browseModelFile(self):
        default_path = getAncestorDir(self.mycfg.dcfg_yolopt.model_file)
        model_filter = "Model (*.pt)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Model file", default_path, model_filter)
        if source_file:
            self.model_file_lineEdit.setText(source_file)

