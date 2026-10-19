# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


from PyQt6 import QtCore, QtWidgets
from pyppbox.gui.configwidgets import config_form, device_dropdown
from pyppbox.gui.numericwidgets import FloatSpinBox, IntSpinBox
from pyppbox.config.unifiedstrings import UnifiedStrings
from pyppbox.config.myconfig import MyConfigurator as MyCFG
from pyppbox.utils.commontools import (getAbsPathFDS, normalizePathFDS, getGlobalRootDir, getAncestorDir, getInt, getBool)

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
        self.max_det_spinBox = IntSpinBox(parent=yoloult_ui)
        self.max_det_spinBox.setObjectName("max_det_spinBox")
        self.devices_label = QtWidgets.QLabel(parent=yoloult_ui)
        self.devices_label.setObjectName("devices_label")
        self.conf_spinBox = FloatSpinBox(parent=yoloult_ui)
        self.conf_spinBox.setObjectName("conf_spinBox")
        self.imgsz_spinBox = IntSpinBox(parent=yoloult_ui)
        self.imgsz_spinBox.setMinimum(1)
        self.imgsz_spinBox.setSingleStep(32)
        self.imgsz_spinBox.setObjectName("imgsz_spinBox")
        self.iou_spinBox = FloatSpinBox(parent=yoloult_ui)
        self.iou_spinBox.setObjectName("iou_spinBox")
        self.conf_label = QtWidgets.QLabel(parent=yoloult_ui)
        self.conf_label.setObjectName("conf_label")
        self.model_file_pushButton = QtWidgets.QPushButton(parent=yoloult_ui)
        self.model_file_pushButton.setObjectName("model_file_pushButton")
        self.iou_label = QtWidgets.QLabel(parent=yoloult_ui)
        self.iou_label.setObjectName("iou_label")
        self.imgsz_label = QtWidgets.QLabel(parent=yoloult_ui)
        self.imgsz_label.setObjectName("imgsz_label")
        self.max_det_label = QtWidgets.QLabel(parent=yoloult_ui)
        self.max_det_label.setObjectName("max_det_label")
        self.boxes_label = QtWidgets.QLabel(parent=yoloult_ui)
        self.boxes_label.setObjectName("boxes_label")
        self.device_comboBox = device_dropdown(yoloult_ui, ['0', 'cpu', 'cuda:0', 'mps'])
        self.device_comboBox.setObjectName('device_comboBox')
        self.device_comboBox.setToolTip('Leave empty for automatic selection, or choose/type a device.')
        self.device_lineEdit = self.device_comboBox.lineEdit()
        self.device_lineEdit.setObjectName("device_lineEdit")
        self.save_pushButton = QtWidgets.QPushButton(parent=yoloult_ui)
        self.save_pushButton.setObjectName("save_pushButton")
        self.model_file_label = QtWidgets.QLabel(parent=yoloult_ui)
        self.model_file_label.setObjectName("model_file_label")
        self.model_file_lineEdit = QtWidgets.QLineEdit(parent=yoloult_ui)
        self.model_file_lineEdit.setObjectName("model_file_lineEdit")
        self.repspoint_calib_label = QtWidgets.QLabel(parent=yoloult_ui)
        self.repspoint_calib_label.setObjectName("repspoint_calib_label")
        self.repspoint_calib_spinBox = FloatSpinBox(parent=yoloult_ui)
        self.repspoint_calib_spinBox.setObjectName("repspoint_calib_spinBox")
        self.boxes_comboBox = QtWidgets.QComboBox(parent=yoloult_ui)
        self.boxes_comboBox.setObjectName("boxes_comboBox")
        self.boxes_comboBox.addItem("")
        self.boxes_comboBox.addItem("")
        self.save_pushButton.setFocus()
        self.retranslateUi(yoloult_ui)
        # custom 
        self.loadYLUTLT()
        config_form(yoloult_ui, [
            (self.conf_label, self.conf_spinBox),
            (self.iou_label, self.iou_spinBox),
            (self.imgsz_label, self.imgsz_spinBox),
            (self.boxes_label, self.boxes_comboBox),
            (self.devices_label, self.device_comboBox),
            (self.max_det_label, self.max_det_spinBox),
            (self.model_file_label, self.model_file_lineEdit, self.model_file_pushButton),
            (self.repspoint_calib_label, self.repspoint_calib_spinBox),
        ], self.save_pushButton)
        self.model_file_pushButton.clicked.connect(self.browseModelFile)
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(yoloult_ui))
        QtCore.QMetaObject.connectSlotsByName(yoloult_ui)

    def retranslateUi(self, yoloult_ui):
        _translate = QtCore.QCoreApplication.translate
        yoloult_ui.setWindowTitle(_translate("yoloult_ui", "YOLO Ultralytics"))
        self.devices_label.setText(_translate("yoloult_ui", "device"))
        self.conf_label.setText(_translate("yoloult_ui", "conf"))
        self.model_file_pushButton.setText(_translate("yoloult_ui", "..."))
        self.iou_label.setText(_translate("yoloult_ui", "iou"))
        self.imgsz_label.setText(_translate("yoloult_ui", "imgsz"))
        self.max_det_label.setText(_translate("yoloult_ui", "max_det"))
        self.boxes_label.setText(_translate("yoloult_ui", "show_boxes"))
        self.device_lineEdit.setPlaceholderText(_translate("yoloult_ui", "0"))
        self.save_pushButton.setText(_translate("yoloult_ui", "Save"))
        self.model_file_label.setText(_translate("yoloult_ui", "model_file"))
        self.repspoint_calib_label.setText(_translate("yoloult_ui", "repspoint_calib"))
        self.boxes_comboBox.setItemText(0, _translate("yoloult_ui", "True"))
        self.boxes_comboBox.setItemText(1, _translate("yoloult_ui", "False"))

    def loadYLUTLT(self):
        self.mycfg.setAllDCFG()
        self.conf_spinBox.setValue(self.mycfg.dcfg_yolout.conf)
        self.iou_spinBox.setValue(self.mycfg.dcfg_yolout.iou)
        self.imgsz_spinBox.setValue(self.mycfg.dcfg_yolout.imgsz)
        self.device_comboBox.setCurrentText(str(self.mycfg.dcfg_yolout.device))
        self.max_det_spinBox.setValue(self.mycfg.dcfg_yolout.max_det)
        self.model_file_lineEdit.setText(getAbsPathFDS(self.mycfg.dcfg_yolout.model_file))
        self.repspoint_calib_spinBox.setValue(self.mycfg.dcfg_yolout.repspoint_calibration)
        self.loadComboBoxes()

    def loadComboBoxes(self):
        if self.mycfg.dcfg_yolout.show_boxes is True:
            self.boxes_comboBox.setCurrentIndex(0)
        else:
            self.boxes_comboBox.setCurrentIndex(1)

    def updateCFG(self, yoloult_ui):
        device = self.device_lineEdit.text().strip()
        yolo_utlt_doc = {
            "dt_name": unified_strings.getUnifiedFormat("YOLO_Ultralytics"),
            "conf": self.conf_spinBox.value(),
            "iou": self.iou_spinBox.value(),
            "imgsz": self.imgsz_spinBox.value(),
            "show_boxes": getBool(self.boxes_comboBox.currentText()),
            "device": device,
            "max_det": self.max_det_spinBox.value(),
            "model_file": normalizePathFDS(root_dir, self.model_file_lineEdit.text()),
            "repspoint_calibration": self.repspoint_calib_spinBox.value()
        }
        try:
            self.mycfg.dumpDetectorCFG(yolo_utlt_doc)
        except (ValueError, OSError) as error:
            QtWidgets.QMessageBox.warning(yoloult_ui, 'Cannot save settings', str(error))
            return
        yoloult_ui.close()

    def browseModelFile(self):
        default_path = getAncestorDir(self.mycfg.dcfg_yolout.model_file)
        model_filter = "Model (*.pt)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Model file", 
                                                               default_path, model_filter)
        if source_file:
            self.model_file_lineEdit.setText(source_file)
