# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


from PyQt6 import QtCore, QtGui, QtWidgets
from pyppbox.gui.configwidgets import config_form
from pyppbox.gui.numericwidgets import FloatSpinBox, OptionalIntSpinBox
from pyppbox.config.unifiedstrings import UnifiedStrings
from pyppbox.config.myconfig import MyConfigurator as MyCFG
from pyppbox.utils.commontools import (getAbsPathFDS, normalizePathFDS, getGlobalRootDir, getAncestorDir)


unified_strings = UnifiedStrings()
root_dir = getGlobalRootDir()

class Ui_DeepSORT(object):

    def __init__(self, cfg_mode, cfg_dir):
        self.cfg_dir = cfg_dir
        if cfg_mode == 0:
            self.mycfg = MyCFG()
        else:
            self.mycfg = MyCFG(self.cfg_dir)

    def setupUi(self, deepsort_ui):
        deepsort_ui.setObjectName("deepsort_ui")
        deepsort_ui.resize(390, 180)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, 
                                           QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(deepsort_ui.sizePolicy().hasHeightForWidth())
        deepsort_ui.setSizePolicy(sizePolicy)
        deepsort_ui.setMinimumSize(QtCore.QSize(390, 180))
        deepsort_ui.setMaximumSize(QtCore.QSize(390, 180))
        self.save_pushButton = QtWidgets.QPushButton(deepsort_ui)
        self.save_pushButton.setGeometry(QtCore.QRect(150, 140, 91, 31))
        self.save_pushButton.setObjectName("save_pushButton")
        self.ds_max_overlap_spinBox = FloatSpinBox(deepsort_ui)
        self.ds_max_overlap_spinBox.setGeometry(QtCore.QRect(110, 40, 241, 21))
        self.ds_max_overlap_spinBox.setObjectName("ds_max_overlap_spinBox")
        self.ds_model_file_pushButton = QtWidgets.QPushButton(deepsort_ui)
        self.ds_model_file_pushButton.setGeometry(QtCore.QRect(360, 100, 21, 24))
        self.ds_model_file_pushButton.setObjectName("ds_model_file_pushButton")
        self.ds_cosine_distance_label = QtWidgets.QLabel(deepsort_ui)
        self.ds_cosine_distance_label.setGeometry(QtCore.QRect(10, 70, 91, 16))
        self.ds_cosine_distance_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                                   QtCore.Qt.AlignmentFlag.AlignTrailing|
                                                   QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ds_cosine_distance_label.setObjectName("ds_cosine_distance_label")
        self.ds_model_file_lineEdit = QtWidgets.QLineEdit(deepsort_ui)
        self.ds_model_file_lineEdit.setGeometry(QtCore.QRect(110, 100, 241, 21))
        self.ds_model_file_lineEdit.setObjectName("ds_model_file_lineEdit")
        self.ds_nn_budget_spinBox = OptionalIntSpinBox(deepsort_ui)
        self.ds_nn_budget_spinBox.setGeometry(QtCore.QRect(110, 10, 241, 21))
        self.ds_nn_budget_spinBox.setObjectName("ds_nn_budget_spinBox")
        self.ds_model_file_label = QtWidgets.QLabel(deepsort_ui)
        self.ds_model_file_label.setGeometry(QtCore.QRect(10, 100, 91, 16))
        self.ds_model_file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                              QtCore.Qt.AlignmentFlag.AlignTrailing|
                                              QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ds_model_file_label.setObjectName("ds_model_file_label")
        self.ds_cosine_distance_spinBox = FloatSpinBox(deepsort_ui)
        self.ds_cosine_distance_spinBox.setMaximum(2.0)
        self.ds_cosine_distance_spinBox.setGeometry(QtCore.QRect(110, 70, 241, 21))
        self.ds_cosine_distance_spinBox.setObjectName("ds_cosine_distance_spinBox")
        self.ds_nn_budget_label = QtWidgets.QLabel(deepsort_ui)
        self.ds_nn_budget_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.ds_nn_budget_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                             QtCore.Qt.AlignmentFlag.AlignTrailing|
                                             QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ds_nn_budget_label.setObjectName("ds_nn_budget_label")
        self.ds_max_overlap_label = QtWidgets.QLabel(deepsort_ui)
        self.ds_max_overlap_label.setGeometry(QtCore.QRect(10, 40, 91, 16))
        self.ds_max_overlap_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                               QtCore.Qt.AlignmentFlag.AlignTrailing|
                                               QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ds_max_overlap_label.setObjectName("ds_max_overlap_label")
        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)
        self.retranslateUi(deepsort_ui)
        # custom 
        self.loadDS()
        config_form(deepsort_ui, [
            (self.ds_nn_budget_label, self.ds_nn_budget_spinBox),
            (self.ds_max_overlap_label, self.ds_max_overlap_spinBox),
            (self.ds_cosine_distance_label, self.ds_cosine_distance_spinBox),
            (self.ds_model_file_label, self.ds_model_file_lineEdit, self.ds_model_file_pushButton),
        ], self.save_pushButton, automatic_device='TensorFlow')
        self.ds_model_file_pushButton.clicked.connect(self.browseModelFile)
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(deepsort_ui))
        QtCore.QMetaObject.connectSlotsByName(deepsort_ui)

    def retranslateUi(self, deepsort_ui):
        _translate = QtCore.QCoreApplication.translate
        deepsort_ui.setWindowTitle(_translate("deepsort_ui", "DeepSORT"))
        self.save_pushButton.setText(_translate("deepsort_ui", "Save"))
        self.ds_model_file_pushButton.setText(_translate("deepsort_ui", "..."))
        self.ds_cosine_distance_label.setText(_translate("deepsort_ui", "cosine_distance"))
        self.ds_model_file_label.setText(_translate("deepsort_ui", "model_file"))
        self.ds_nn_budget_label.setText(_translate("deepsort_ui", "nn_budget"))
        self.ds_max_overlap_label.setText(_translate("deepsort_ui", "max_overlap"))

    def loadDS(self):
        self.mycfg.setAllTCFG()
        self.ds_cosine_distance_spinBox.setValue(self.mycfg.tcfg_deepsort.max_cosine_distance)
        self.ds_max_overlap_spinBox.setValue(self.mycfg.tcfg_deepsort.nms_max_overlap)
        self.ds_nn_budget_spinBox.setValue(self.mycfg.tcfg_deepsort.nn_budget)
        self.ds_model_file_lineEdit.setText(getAbsPathFDS(self.mycfg.tcfg_deepsort.model_file))

    def updateCFG(self, YOLOForm):
        deepsort_doc = {
            "tk_name": unified_strings.getUnifiedFormat("DeepSORT"),
            "nn_budget": self.ds_nn_budget_spinBox.value(),
            "nms_max_overlap": self.ds_max_overlap_spinBox.value(),
            "max_cosine_distance": self.ds_cosine_distance_spinBox.value(),
            "model_file": normalizePathFDS(root_dir, self.ds_model_file_lineEdit.text())
        }
        try:
            self.mycfg.dumpTrackerCFG(deepsort_doc)
        except (ValueError, OSError) as error:
            QtWidgets.QMessageBox.warning(YOLOForm, 'Cannot save settings', str(error))
            return
        YOLOForm.close()

    def browseModelFile(self):
        default_path = getAncestorDir(self.mycfg.tcfg_deepsort.model_file)
        model_filter = "Protobuf (*.pb)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Model file", 
                                                               default_path, model_filter)
        if source_file:
            self.ds_model_file_lineEdit.setText(source_file)
