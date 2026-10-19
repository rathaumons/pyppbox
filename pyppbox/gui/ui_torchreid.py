# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


from PyQt6 import QtCore, QtWidgets
from pyppbox.gui.configwidgets import config_form, device_dropdown
from pyppbox.gui.numericwidgets import FloatSpinBox
from pyppbox.config.unifiedstrings import UnifiedStrings
from pyppbox.config.myconfig import MyConfigurator as MyCFG
from pyppbox.config.torchreidmodels import TorchreidModelDict
from pyppbox.utils.commontools import (getAbsPathFDS, normalizePathFDS, getFileName, getGlobalRootDir, getAncestorDir, joinFPathFull, getInt)


unified_strings = UnifiedStrings()
root_dir = getGlobalRootDir()
global_cfg_root = joinFPathFull(root_dir, 'cfg')
internal_cfg_dir = joinFPathFull(global_cfg_root, 'configurations')

class Ui_Torchreid(object):

    def __init__(self, cfg_mode, cfg_dir):
        self.cfg_dir = cfg_dir
        if cfg_mode == 0:
            self.mycfg = MyCFG()
        else:
            self.mycfg = MyCFG(self.cfg_dir)

    def setupUi(self, torchreid_ui):
        torchreid_ui.setObjectName("torchreid_ui")
        self.dr_model_path_label = QtWidgets.QLabel(torchreid_ui)
        self.dr_model_path_label.setObjectName("dr_model_path_label")
        self.dr_min_confidence_spinBox = FloatSpinBox(torchreid_ui)
        self.dr_min_confidence_spinBox.setObjectName("dr_min_confidence_spinBox")
        self.save_pushButton = QtWidgets.QPushButton(torchreid_ui)
        self.save_pushButton.setObjectName("save_pushButton")
        self.dr_min_confidence_label = QtWidgets.QLabel(torchreid_ui)
        self.dr_min_confidence_label.setObjectName("dr_min_confidence_label")
        self.dr_classifier_pkl_label = QtWidgets.QLabel(torchreid_ui)
        self.dr_classifier_pkl_label.setObjectName("dr_classifier_pkl_label")
        self.dr_model_path_lineEdit = QtWidgets.QLineEdit(torchreid_ui)
        self.dr_model_path_lineEdit.setObjectName("dr_model_path_lineEdit")
        self.dr_train_data_pushButton = QtWidgets.QPushButton(torchreid_ui)
        self.dr_train_data_pushButton.setObjectName("dr_train_data_pushButton")
        self.dr_model_name_lineEdit = QtWidgets.QLineEdit(torchreid_ui)
        self.dr_model_name_lineEdit.setObjectName("dr_model_name_lineEdit")
        self.dr_model_path_pushButton = QtWidgets.QPushButton(torchreid_ui)
        self.dr_model_path_pushButton.setObjectName("dr_model_path_pushButton")
        self.dr_model_name_label = QtWidgets.QLabel(torchreid_ui)
        self.dr_model_name_label.setObjectName("dr_model_name_label")
        self.dr_train_data_label = QtWidgets.QLabel(torchreid_ui)
        self.dr_train_data_label.setObjectName("dr_train_data_label")
        self.dr_train_data_lineEdit = QtWidgets.QLineEdit(torchreid_ui)
        self.dr_train_data_lineEdit.setObjectName("dr_train_data_lineEdit")
        self.dr_classifier_pkl_pushButton = QtWidgets.QPushButton(torchreid_ui)
        self.dr_classifier_pkl_pushButton.setObjectName("dr_classifier_pkl_pushButton")
        self.dr_classifier_pkl_lineEdit = QtWidgets.QLineEdit(torchreid_ui)
        self.dr_classifier_pkl_lineEdit.setObjectName("dr_classifier_pkl_lineEdit")
        self.dr_devices_label = QtWidgets.QLabel(torchreid_ui)
        self.dr_devices_label.setObjectName("dr_devices_label")
        self.dr_device_comboBox = device_dropdown(torchreid_ui, ['cpu', 'cuda', 'cuda:0', 'mps'])
        self.dr_device_comboBox.setObjectName('dr_device_comboBox')
        self.dr_device_lineEdit = self.dr_device_comboBox.lineEdit()
        self.dr_device_lineEdit.setObjectName("dr_device_lineEdit")
        self.retranslateUi(torchreid_ui)
        # custom
        self.md = TorchreidModelDict()
        self.loadDR()
        config_form(torchreid_ui, [
            (self.dr_classifier_pkl_label, self.dr_classifier_pkl_lineEdit, self.dr_classifier_pkl_pushButton),
            (self.dr_train_data_label, self.dr_train_data_lineEdit, self.dr_train_data_pushButton),
            (self.dr_model_name_label, self.dr_model_name_lineEdit),
            (self.dr_model_path_label, self.dr_model_path_lineEdit, self.dr_model_path_pushButton),
            (self.dr_min_confidence_label, self.dr_min_confidence_spinBox),
            (self.dr_devices_label, self.dr_device_comboBox),
        ], self.save_pushButton)
        self.dr_classifier_pkl_pushButton.clicked.connect(self.browseClassifierPKL)
        self.dr_train_data_pushButton.clicked.connect(self.browseTrainData)
        self.dr_model_path_pushButton.clicked.connect(self.browseModelPath)
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(torchreid_ui))
        QtCore.QMetaObject.connectSlotsByName(torchreid_ui)

    def retranslateUi(self, torchreid_ui):
        _translate = QtCore.QCoreApplication.translate
        torchreid_ui.setWindowTitle(_translate("torchreid_ui", "Torchreid"))
        self.dr_model_path_label.setText(_translate("torchreid_ui", "model_path"))
        self.save_pushButton.setText(_translate("torchreid_ui", "Save"))
        self.dr_min_confidence_label.setText(_translate("torchreid_ui", "min_confidence"))
        self.dr_devices_label.setText(_translate("torchreid_ui", "device"))
        self.dr_device_lineEdit.setPlaceholderText(_translate("torchreid_ui", "cuda"))
        self.dr_classifier_pkl_label.setText(_translate("torchreid_ui", "classifier_pkl"))
        self.dr_train_data_pushButton.setText(_translate("torchreid_ui", "..."))
        self.dr_model_path_pushButton.setText(_translate("torchreid_ui", "..."))
        self.dr_model_name_label.setText(_translate("torchreid_ui", "model_name"))
        self.dr_train_data_label.setText(_translate("torchreid_ui", "train_data"))
        self.dr_classifier_pkl_pushButton.setText(_translate("torchreid_ui", "..."))

    def loadDR(self):
        self.mycfg.setAllRCFG()
        self.dr_classifier_pkl_lineEdit.setText(getAbsPathFDS(self.mycfg.rcfg_torchreid.classifier_pkl))
        self.dr_train_data_lineEdit.setText(getAbsPathFDS(self.mycfg.rcfg_torchreid.train_data))
        self.dr_model_name_lineEdit.setText(str(self.mycfg.rcfg_torchreid.model_name))
        self.dr_model_path_lineEdit.setText(getAbsPathFDS(self.mycfg.rcfg_torchreid.model_path))
        self.dr_min_confidence_spinBox.setValue(self.mycfg.rcfg_torchreid.min_confidence)
        self.dr_device_comboBox.setCurrentText(str(self.mycfg.rcfg_torchreid.device))

    def updateCFG(self, torchreid_ui):
        device = self.dr_device_lineEdit.text().strip()
        Torchreid_doc = {
            "ri_name": unified_strings.getUnifiedFormat("Torchreid"),
            "classifier_pkl": normalizePathFDS(root_dir, self.dr_classifier_pkl_lineEdit.text()),
            "train_data": normalizePathFDS(root_dir, self.dr_train_data_lineEdit.text()),
            "model_name": self.dr_model_name_lineEdit.text(),
            "model_path": normalizePathFDS(root_dir, self.dr_model_path_lineEdit.text()),
            "min_confidence": self.dr_min_confidence_spinBox.value(),
            "device": device
        }
        try:
            self.mycfg.dumpReiderCFG(Torchreid_doc)
        except (ValueError, OSError) as error:
            QtWidgets.QMessageBox.warning(torchreid_ui, 'Cannot save settings', str(error))
            return
        torchreid_ui.close()

    def browseTrainData(self):
        default_path = getAncestorDir(self.mycfg.rcfg_torchreid.train_data)
        train_data = QtWidgets.QFileDialog.getExistingDirectory(None, "Train data folder", default_path)
        if train_data:
            self.dr_train_data_lineEdit.setText(train_data)

    def browseModelPath(self):
        default_path = getAncestorDir(self.mycfg.rcfg_torchreid.model_path)
        model_filter = "PyTorch model (*.pth *.pyth *.tar *.*)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Model file", 
                                                               default_path, model_filter)
        if source_file:
            self.dr_model_path_lineEdit.setText(source_file)
            self.autoSetModelName(source_file)

    def browseClassifierPKL(self):
        default_path = getAncestorDir(self.mycfg.rcfg_torchreid.classifier_pkl)
        pkl_filter = "Pickle (*.pkl)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Classifier file", 
                                                               default_path, pkl_filter)
        if source_file:
            self.dr_classifier_pkl_lineEdit.setText(source_file)

    def autoSetModelName(self, model_path, enable=True):
        if enable:
            self.dr_model_name_lineEdit.setText(self.md.findModelArch(getFileName(model_path)))
