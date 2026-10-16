# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


from PyQt6 import QtCore, QtGui, QtWidgets
from pyppbox.gui.numericwidgets import FloatSpinBox, IntSpinBox
from pyppbox.config.unifiedstrings import UnifiedStrings
from pyppbox.config.myconfig import MyConfigurator as MyCFG
from pyppbox.utils.commontools import (getAbsPathFDS, normalizePathFDS, getGlobalRootDir, getAncestorDir)


unified_strings = UnifiedStrings()
root_dir = getGlobalRootDir()

class Ui_YOLOCLS(object):

    def __init__(self, cfg_mode, cfg_dir):
        self.cfg_dir = cfg_dir
        if cfg_mode == 0:
            self.mycfg = MyCFG()
        else:
            self.mycfg = MyCFG(self.cfg_dir)

    def setupUi(self, yolocls_ui):
        yolocls_ui.setObjectName("yolocls_ui")
        yolocls_ui.resize(390, 270)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, 
                                           QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(yolocls_ui.sizePolicy().hasHeightForWidth())
        yolocls_ui.setSizePolicy(sizePolicy)
        yolocls_ui.setMinimumSize(QtCore.QSize(390, 270))
        yolocls_ui.setMaximumSize(QtCore.QSize(390, 270))
        self.save_pushButton = QtWidgets.QPushButton(yolocls_ui)
        self.save_pushButton.setGeometry(QtCore.QRect(150, 230, 91, 31))
        self.save_pushButton.setObjectName("save_pushButton")
        self.yl_model_img_size_label = QtWidgets.QLabel(yolocls_ui)
        self.yl_model_img_size_label.setGeometry(QtCore.QRect(10, 160, 91, 16))
        self.yl_model_img_size_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                                  QtCore.Qt.AlignmentFlag.AlignTrailing|
                                                  QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.yl_model_img_size_label.setObjectName("yl_model_img_size_label")
        self.yl_nms_spinBox = FloatSpinBox(yolocls_ui)
        self.yl_nms_spinBox.setGeometry(QtCore.QRect(110, 10, 241, 21))
        self.yl_nms_spinBox.setObjectName("yl_nms_spinBox")
        self.yl_model_cfg_file_pushButton = QtWidgets.QPushButton(yolocls_ui)
        self.yl_model_cfg_file_pushButton.setGeometry(QtCore.QRect(360, 100, 21, 24))
        self.yl_model_cfg_file_pushButton.setObjectName("yl_model_cfg_file_pushButton")
        self.yl_model_weights_lineEdit = QtWidgets.QLineEdit(yolocls_ui)
        self.yl_model_weights_lineEdit.setGeometry(QtCore.QRect(110, 130, 241, 21))
        self.yl_model_weights_lineEdit.setReadOnly(True)
        self.yl_model_weights_lineEdit.setObjectName("yl_model_weights_lineEdit")
        self.yl_class_file_lineEdit = QtWidgets.QLineEdit(yolocls_ui)
        self.yl_class_file_lineEdit.setGeometry(QtCore.QRect(110, 70, 241, 21))
        self.yl_class_file_lineEdit.setReadOnly(True)
        self.yl_class_file_lineEdit.setObjectName("yl_class_file_lineEdit")
        self.yl_conf_spinBox = FloatSpinBox(yolocls_ui)
        self.yl_conf_spinBox.setGeometry(QtCore.QRect(110, 40, 241, 21))
        self.yl_conf_spinBox.setObjectName("yl_conf_spinBox")
        self.yl_repspoint_calib_spinBox = FloatSpinBox(yolocls_ui)
        self.yl_repspoint_calib_spinBox.setGeometry(QtCore.QRect(110, 190, 241, 21))
        self.yl_repspoint_calib_spinBox.setObjectName("yl_repspoint_calib_spinBox")
        self.yl_class_file_pushButton = QtWidgets.QPushButton(yolocls_ui)
        self.yl_class_file_pushButton.setGeometry(QtCore.QRect(360, 70, 21, 24))
        self.yl_class_file_pushButton.setObjectName("yl_class_file_pushButton")
        self.yl_nms_label = QtWidgets.QLabel(yolocls_ui)
        self.yl_nms_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.yl_nms_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                       QtCore.Qt.AlignmentFlag.AlignTrailing|
                                       QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.yl_nms_label.setObjectName("yl_nms_label")
        self.yl_model_weights_pushButton = QtWidgets.QPushButton(yolocls_ui)
        self.yl_model_weights_pushButton.setGeometry(QtCore.QRect(360, 130, 21, 24))
        self.yl_model_weights_pushButton.setObjectName("yl_model_weights_pushButton")
        self.yl_model_cfg_file_label = QtWidgets.QLabel(yolocls_ui)
        self.yl_model_cfg_file_label.setGeometry(QtCore.QRect(10, 100, 91, 16))
        self.yl_model_cfg_file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                                  QtCore.Qt.AlignmentFlag.AlignTrailing|
                                                  QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.yl_model_cfg_file_label.setObjectName("yl_model_cfg_file_label")
        self.yl_conf_label = QtWidgets.QLabel(yolocls_ui)
        self.yl_conf_label.setGeometry(QtCore.QRect(10, 40, 91, 16))
        self.yl_conf_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                        QtCore.Qt.AlignmentFlag.AlignTrailing|
                                        QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.yl_conf_label.setObjectName("yl_conf_label")
        self.yl_class_file_label = QtWidgets.QLabel(yolocls_ui)
        self.yl_class_file_label.setGeometry(QtCore.QRect(10, 70, 91, 16))
        self.yl_class_file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                              QtCore.Qt.AlignmentFlag.AlignTrailing|
                                              QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.yl_class_file_label.setObjectName("yl_class_file_label")
        self.yl_repspint_calib_label = QtWidgets.QLabel(yolocls_ui)
        self.yl_repspint_calib_label.setGeometry(QtCore.QRect(10, 190, 91, 16))
        self.yl_repspint_calib_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                                  QtCore.Qt.AlignmentFlag.AlignTrailing|
                                                  QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.yl_repspint_calib_label.setObjectName("yl_repspint_calib_label")
        self.yl_model_cfg_file_lineEdit = QtWidgets.QLineEdit(yolocls_ui)
        self.yl_model_cfg_file_lineEdit.setGeometry(QtCore.QRect(110, 100, 241, 21))
        self.yl_model_cfg_file_lineEdit.setReadOnly(True)
        self.yl_model_cfg_file_lineEdit.setObjectName("yl_model_cfg_file_lineEdit")
        self.yl_model_weights_label = QtWidgets.QLabel(yolocls_ui)
        self.yl_model_weights_label.setGeometry(QtCore.QRect(10, 130, 91, 16))
        self.yl_model_weights_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                                 QtCore.Qt.AlignmentFlag.AlignTrailing|
                                                 QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.yl_model_weights_label.setObjectName("yl_model_weights_label")
        self.yl_model_imgsize_spinBox = IntSpinBox(yolocls_ui)
        self.yl_model_imgsize_spinBox.setMinimum(1)
        self.yl_model_imgsize_spinBox.setSingleStep(32)
        self.yl_model_imgsize_spinBox.setGeometry(QtCore.QRect(110, 160, 241, 21))
        self.yl_model_imgsize_spinBox.setObjectName("yl_model_imgsize_spinBox")
        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)
        self.retranslateUi(yolocls_ui)
        # custom 
        self.loadYL()
        self.yl_class_file_pushButton.clicked.connect(self.browseClassFile)
        self.yl_model_cfg_file_pushButton.clicked.connect(self.browseModelCFG)
        self.yl_model_weights_pushButton.clicked.connect(self.browseModelWeights)
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(yolocls_ui))
        QtCore.QMetaObject.connectSlotsByName(yolocls_ui)

    def retranslateUi(self, yolocls_ui):
        _translate = QtCore.QCoreApplication.translate
        yolocls_ui.setWindowTitle(_translate("yolocls_ui", "YOLO Classic"))
        self.yl_model_img_size_label.setText(_translate("yolocls_ui", "model_img_size"))
        self.yl_model_cfg_file_pushButton.setText(_translate("yolocls_ui", "..."))
        self.yl_class_file_pushButton.setText(_translate("yolocls_ui", "..."))
        self.yl_nms_label.setText(_translate("yolocls_ui", "nms"))
        self.yl_model_weights_pushButton.setText(_translate("yolocls_ui", "..."))
        self.yl_model_cfg_file_label.setText(_translate("yolocls_ui", "model_cfg_file"))
        self.yl_conf_label.setText(_translate("yolocls_ui", "conf"))
        self.yl_class_file_label.setText(_translate("yolocls_ui", "class_file_file"))
        self.yl_repspint_calib_label.setText(_translate("yolocls_ui", "repspoint_calib"))
        self.yl_model_weights_label.setText(_translate("yolocls_ui", "model_weights"))
        self.save_pushButton.setText(_translate("yolocls_ui", "Save"))

    def loadYL(self):
        self.mycfg.setAllDCFG()
        self.yl_nms_spinBox.setValue(self.mycfg.dcfg_yolocs.nms)
        self.yl_conf_spinBox.setValue(self.mycfg.dcfg_yolocs.conf)
        self.yl_model_cfg_file_lineEdit.setText(getAbsPathFDS(self.mycfg.dcfg_yolocs.model_cfg_file))
        self.yl_class_file_lineEdit.setText(getAbsPathFDS(self.mycfg.dcfg_yolocs.class_file))
        self.yl_model_weights_lineEdit.setText(getAbsPathFDS(self.mycfg.dcfg_yolocs.model_weights))
        self.yl_model_imgsize_spinBox.setValue(self.mycfg.dcfg_yolocs.model_image_size)
        self.yl_repspoint_calib_spinBox.setValue(self.mycfg.dcfg_yolocs.repspoint_calibration)

    def updateCFG(self, yolocls_ui):
        yolocs_doc = {
            "dt_name": unified_strings.getUnifiedFormat("YOLO_Classic"),
            "nms": self.yl_nms_spinBox.value(),
            "conf": self.yl_conf_spinBox.value(),
            "class_file": normalizePathFDS(root_dir, self.yl_class_file_lineEdit.text()),
            "model_cfg_file": normalizePathFDS(root_dir, self.yl_model_cfg_file_lineEdit.text()),
            "model_weights": normalizePathFDS(root_dir, self.yl_model_weights_lineEdit.text()),
            "model_image_size": self.yl_model_imgsize_spinBox.value(),
            "repspoint_calibration": self.yl_repspoint_calib_spinBox.value()
        }
        self.mycfg.dumpDetectorCFG(yolocs_doc)
        yolocls_ui.close()

    def browseClassFile(self):
        default_path = getAncestorDir(self.mycfg.dcfg_yolocs.class_file)
        file_filter = "COCO (*.names)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "COCO class file", 
                                                               default_path, file_filter)
        if source_file:
            self.yl_class_file_lineEdit.setText(source_file)

    def browseModelCFG(self):
        default_path = getAncestorDir(self.mycfg.dcfg_yolocs.model_cfg_file)
        cfg_filter = "Configurator (*.cfg)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Model configurator file", 
                                                               default_path, cfg_filter)
        if source_file:
            self.yl_model_cfg_file_lineEdit.setText(source_file)

    def browseModelWeights(self):
        default_path = getAncestorDir(self.mycfg.dcfg_yolocs.model_weights)
        weight_filter = "Weights (*.weights)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Model weights file", 
                                                               default_path, weight_filter)
        if source_file:
            self.yl_model_weights_lineEdit.setText(source_file)
