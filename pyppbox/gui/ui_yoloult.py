# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                           #
#   pyppbox: Toolbox for people detecting, tracking, and re-identifying.    #
#   Copyright (C) 2022 UMONS-Numediart                                      #
#                                                                           #
#   This program is free software: you can redistribute it and/or modify    #
#   it under the terms of the GNU General Public License as published by    #
#   the Free Software Foundation, either version 3 of the License, or       #
#   (at your option) any later version.                                     #
#                                                                           #
#   This program is distributed in the hope that it will be useful,         #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#   GNU General Public License for more details.                            #
#                                                                           #
#   You should have received a copy of the GNU General Public License       #
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.  #
#                                                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


from PyQt6 import QtCore, QtGui, QtWidgets
from pyppbox.config.unifiedstrings import UnifiedStrings
from pyppbox.config.myconfig import MyConfigurator as MyCFG
from pyppbox.utils.commontools import (getAbsPathFDS, normalizePathFDS, 
                                       getGlobalRootDir, getAncestorDir, 
                                       getFloat, getInt)

unified_strings = UnifiedStrings()
root_dir = getGlobalRootDir()

class Ui_YOLOULT(object):

    def __init__(self, cfg_mode, cfg_dir):
        self.cfg_dir = cfg_dir
        if cfg_mode == 0:
            self.mycfg = MyCFG()
        else:
            self.mycfg = MyCFG(self.cfg_dir)
    
    def setupUi(self, yoloult_ui):
        yoloult_ui.setObjectName("yoloult_ui")
        yoloult_ui.resize(390, 330)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, 
                                           QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(yoloult_ui.sizePolicy().hasHeightForWidth())
        yoloult_ui.setSizePolicy(sizePolicy)
        yoloult_ui.setMinimumSize(QtCore.QSize(390, 330))
        yoloult_ui.setMaximumSize(QtCore.QSize(390, 330))
        self.max_det_lineEdit = QtWidgets.QLineEdit(parent=yoloult_ui)
        self.max_det_lineEdit.setGeometry(QtCore.QRect(110, 160, 241, 21))
        self.max_det_lineEdit.setObjectName("max_det_lineEdit")
        self.devices_label = QtWidgets.QLabel(parent=yoloult_ui)
        self.devices_label.setGeometry(QtCore.QRect(10, 130, 91, 16))
        self.devices_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                        QtCore.Qt.AlignmentFlag.AlignTrailing|
                                        QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.devices_label.setObjectName("devices_label")
        self.conf_lineEdit = QtWidgets.QLineEdit(parent=yoloult_ui)
        self.conf_lineEdit.setGeometry(QtCore.QRect(110, 10, 241, 21))
        self.conf_lineEdit.setObjectName("conf_lineEdit")
        self.imgsz_lineEdit = QtWidgets.QLineEdit(parent=yoloult_ui)
        self.imgsz_lineEdit.setGeometry(QtCore.QRect(110, 70, 241, 21))
        self.imgsz_lineEdit.setText("")
        self.imgsz_lineEdit.setObjectName("imgsz_lineEdit")
        self.iou_lineEdit = QtWidgets.QLineEdit(parent=yoloult_ui)
        self.iou_lineEdit.setGeometry(QtCore.QRect(110, 40, 241, 21))
        self.iou_lineEdit.setObjectName("iou_lineEdit")
        self.conf_label = QtWidgets.QLabel(parent=yoloult_ui)
        self.conf_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.conf_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                     QtCore.Qt.AlignmentFlag.AlignTrailing|
                                     QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.conf_label.setObjectName("conf_label")
        self.model_file_pushButton = QtWidgets.QPushButton(parent=yoloult_ui)
        self.model_file_pushButton.setGeometry(QtCore.QRect(360, 220, 21, 24))
        self.model_file_pushButton.setObjectName("model_file_pushButton")
        self.iou_label = QtWidgets.QLabel(parent=yoloult_ui)
        self.iou_label.setGeometry(QtCore.QRect(10, 40, 91, 16))
        self.iou_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                    QtCore.Qt.AlignmentFlag.AlignTrailing|
                                    QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.iou_label.setObjectName("iou_label")
        self.imgsz_label = QtWidgets.QLabel(parent=yoloult_ui)
        self.imgsz_label.setGeometry(QtCore.QRect(10, 70, 91, 16))
        self.imgsz_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                      QtCore.Qt.AlignmentFlag.AlignTrailing|
                                      QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.imgsz_label.setObjectName("imgsz_label")
        self.max_det_label = QtWidgets.QLabel(parent=yoloult_ui)
        self.max_det_label.setGeometry(QtCore.QRect(10, 160, 91, 16))
        self.max_det_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                        QtCore.Qt.AlignmentFlag.AlignTrailing|
                                        QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.max_det_label.setObjectName("max_det_label")
        self.boxes_label = QtWidgets.QLabel(parent=yoloult_ui)
        self.boxes_label.setGeometry(QtCore.QRect(10, 100, 91, 16))
        self.boxes_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                      QtCore.Qt.AlignmentFlag.AlignTrailing|
                                      QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.boxes_label.setObjectName("boxes_label")
        self.device_lineEdit = QtWidgets.QLineEdit(parent=yoloult_ui)
        self.device_lineEdit.setGeometry(QtCore.QRect(110, 130, 241, 21))
        self.device_lineEdit.setObjectName("device_lineEdit")
        self.save_pushButton = QtWidgets.QPushButton(parent=yoloult_ui)
        self.save_pushButton.setGeometry(QtCore.QRect(150, 290, 91, 31))
        self.save_pushButton.setObjectName("save_pushButton")
        self.model_file_label = QtWidgets.QLabel(parent=yoloult_ui)
        self.model_file_label.setGeometry(QtCore.QRect(10, 220, 91, 16))
        self.model_file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                           QtCore.Qt.AlignmentFlag.AlignTrailing|
                                           QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.model_file_label.setObjectName("model_file_label")
        self.model_file_lineEdit = QtWidgets.QLineEdit(parent=yoloult_ui)
        self.model_file_lineEdit.setGeometry(QtCore.QRect(110, 220, 241, 21))
        self.model_file_lineEdit.setObjectName("model_file_lineEdit")
        self.model_file_lineEdit.setReadOnly(True)
        self.line_width_lineEdit = QtWidgets.QLineEdit(parent=yoloult_ui)
        self.line_width_lineEdit.setGeometry(QtCore.QRect(110, 190, 241, 21))
        self.line_width_lineEdit.setObjectName("line_width_lineEdit")
        self.line_width_label = QtWidgets.QLabel(parent=yoloult_ui)
        self.line_width_label.setGeometry(QtCore.QRect(10, 190, 91, 16))
        self.line_width_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                           QtCore.Qt.AlignmentFlag.AlignTrailing|
                                           QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.line_width_label.setObjectName("line_width_label")
        self.repspoint_calib_label = QtWidgets.QLabel(parent=yoloult_ui)
        self.repspoint_calib_label.setGeometry(QtCore.QRect(10, 250, 91, 16))
        self.repspoint_calib_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                                QtCore.Qt.AlignmentFlag.AlignTrailing|
                                                QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.repspoint_calib_label.setObjectName("repspoint_calib_label")
        self.repspoint_calib_lineEdit = QtWidgets.QLineEdit(parent=yoloult_ui)
        self.repspoint_calib_lineEdit.setGeometry(QtCore.QRect(110, 250, 241, 21))
        self.repspoint_calib_lineEdit.setObjectName("repspoint_calib_lineEdit")
        self.boxes_comboBox = QtWidgets.QComboBox(parent=yoloult_ui)
        self.boxes_comboBox.setGeometry(QtCore.QRect(110, 100, 241, 21))
        self.boxes_comboBox.setObjectName("boxes_comboBox")
        self.boxes_comboBox.addItem("")
        self.boxes_comboBox.addItem("")
        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)
        self.save_pushButton.setFocus()
        self.retranslateUi(yoloult_ui)
        # custom 
        self.loadYLUTLT()
        self.model_file_pushButton.clicked.connect(self.browseModelFile)
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(yoloult_ui))
        QtCore.QMetaObject.connectSlotsByName(yoloult_ui)

    def retranslateUi(self, yoloult_ui):
        _translate = QtCore.QCoreApplication.translate
        yoloult_ui.setWindowTitle(_translate("yoloult_ui", "YOLO Ultralytics"))
        self.max_det_lineEdit.setPlaceholderText(_translate("yoloult_ui", "100"))
        self.devices_label.setText(_translate("yoloult_ui", "device"))
        self.conf_lineEdit.setPlaceholderText(_translate("yoloult_ui", "0.5"))
        self.imgsz_lineEdit.setPlaceholderText(_translate("yoloult_ui", "416"))
        self.iou_lineEdit.setPlaceholderText(_translate("yoloult_ui", "0.7"))
        self.conf_label.setText(_translate("yoloult_ui", "conf"))
        self.model_file_pushButton.setText(_translate("yoloult_ui", "..."))
        self.iou_label.setText(_translate("yoloult_ui", "iou"))
        self.imgsz_label.setText(_translate("yoloult_ui", "imgsz"))
        self.max_det_label.setText(_translate("yoloult_ui", "max_det"))
        self.boxes_label.setText(_translate("yoloult_ui", "show_boxes"))
        self.device_lineEdit.setPlaceholderText(_translate("yoloult_ui", "0"))
        self.save_pushButton.setText(_translate("yoloult_ui", "Save"))
        self.model_file_label.setText(_translate("yoloult_ui", "model_file"))
        self.line_width_lineEdit.setPlaceholderText(_translate("yoloult_ui", "500"))
        self.line_width_label.setText(_translate("yoloult_ui", "line_width"))
        self.repspoint_calib_label.setText(_translate("yoloult_ui", "repspoint_calib"))
        self.repspoint_calib_lineEdit.setPlaceholderText(_translate("yoloult_ui", "0.25"))
        self.boxes_comboBox.setItemText(0, _translate("yoloult_ui", "True"))
        self.boxes_comboBox.setItemText(1, _translate("yoloult_ui", "False"))

    def loadYLUTLT(self):
        self.mycfg.setAllDCFG()
        self.conf_lineEdit.setText(str(self.mycfg.dcfg_yolout.conf))
        self.iou_lineEdit.setText(str(self.mycfg.dcfg_yolout.iou))
        self.imgsz_lineEdit.setText(str(self.mycfg.dcfg_yolout.imgsz))
        self.device_lineEdit.setText(str(self.mycfg.dcfg_yolout.device))
        self.max_det_lineEdit.setText(str(self.mycfg.dcfg_yolout.max_det))
        self.line_width_lineEdit.setText(str(self.mycfg.dcfg_yolout.line_width))
        self.model_file_lineEdit.setText(getAbsPathFDS(self.mycfg.dcfg_yolout.model_file))
        self.repspoint_calib_lineEdit.setText(str(self.mycfg.dcfg_yolout.repspoint_calibration))
        self.loadComboBoxes()

    def loadComboBoxes(self):
        if self.mycfg.dcfg_yolout.show_boxes is True:
            self.boxes_comboBox.setCurrentIndex(0)
        else:
            self.boxes_comboBox.setCurrentIndex(1)

    def updateCFG(self, yoloult_ui):
        device = 0
        if 'cpu' in self.device_lineEdit.text().lower():
            device = 'cpu'
        elif 'cuda' in self.device_lineEdit.text().lower():
            device = 'cuda'
        else:
            device = getInt(self.device_lineEdit.text())
        yolo_utlt_doc = {
            "dt_name": unified_strings.getUnifiedFormat("YOLO_Ultralytics"),
            "conf": getFloat(self.conf_lineEdit.text(), default_val=0.5),
            "iou": getFloat(self.iou_lineEdit.text(), default_val=0.7),
            "imgsz": getInt(self.imgsz_lineEdit.text(), default_val=416),
            "show_boxes": self.boxes_comboBox.currentText(),
            "device": device,
            "max_det": getInt(self.max_det_lineEdit.text(), default_val=100),
            "line_width": getInt(self.line_width_lineEdit.text(), default_val=500),
            "model_file": normalizePathFDS(root_dir, self.model_file_lineEdit.text()),
            "repspoint_calibration": getFloat(self.repspoint_calib_lineEdit.text(), default_val=0.25)
        }
        yolocs_doc = self.mycfg.dcfg_yolocs.getDocument()
        gt_doc = self.mycfg.dcfg_gt.getDocument()
        self.mycfg.dumpAllDCFG([yolocs_doc, yolo_utlt_doc, gt_doc])
        yoloult_ui.close()

    def browseModelFile(self):
        default_path = getAncestorDir(self.mycfg.dcfg_yolout.model_file)
        model_filter = "Model (*.pt)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Model file", 
                                                               default_path, model_filter)
        if source_file:
            self.model_file_lineEdit.setText(source_file)
