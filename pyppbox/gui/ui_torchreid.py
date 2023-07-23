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
from pyppbox.modules.reiders.torchreid.model_dict import TorchreidModelDict
from pyppbox.utils.commontools import (getAbsPathFDS, normalizePathFDS, getFileName, 
                                       getGlobalRootDir, getAncestorDir, joinFPathFull,
                                       getFloat, getInt)


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
        torchreid_ui.resize(390, 240)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, 
                                           QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(torchreid_ui.sizePolicy().hasHeightForWidth())
        torchreid_ui.setSizePolicy(sizePolicy)
        torchreid_ui.setMinimumSize(QtCore.QSize(390, 240))
        torchreid_ui.setMaximumSize(QtCore.QSize(390, 240))
        self.dr_model_path_label = QtWidgets.QLabel(torchreid_ui)
        self.dr_model_path_label.setGeometry(QtCore.QRect(10, 100, 91, 16))
        self.dr_model_path_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                              QtCore.Qt.AlignmentFlag.AlignTrailing|
                                              QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.dr_model_path_label.setObjectName("dr_model_path_label")
        self.dr_min_confidence_lineEdit = QtWidgets.QLineEdit(torchreid_ui)
        self.dr_min_confidence_lineEdit.setGeometry(QtCore.QRect(110, 130, 241, 21))
        self.dr_min_confidence_lineEdit.setReadOnly(False)
        self.dr_min_confidence_lineEdit.setObjectName("dr_min_confidence_lineEdit")
        self.save_pushButton = QtWidgets.QPushButton(torchreid_ui)
        self.save_pushButton.setGeometry(QtCore.QRect(150, 200, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)
        self.save_pushButton.setObjectName("save_pushButton")
        self.dr_min_confidence_label = QtWidgets.QLabel(torchreid_ui)
        self.dr_min_confidence_label.setGeometry(QtCore.QRect(10, 130, 91, 16))
        self.dr_min_confidence_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                                  QtCore.Qt.AlignmentFlag.AlignTrailing|
                                                  QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.dr_min_confidence_label.setObjectName("dr_min_confidence_label")
        self.dr_classifier_pkl_label = QtWidgets.QLabel(torchreid_ui)
        self.dr_classifier_pkl_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.dr_classifier_pkl_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                                  QtCore.Qt.AlignmentFlag.AlignTrailing|
                                                  QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.dr_classifier_pkl_label.setObjectName("dr_classifier_pkl_label")
        self.dr_model_path_lineEdit = QtWidgets.QLineEdit(torchreid_ui)
        self.dr_model_path_lineEdit.setGeometry(QtCore.QRect(110, 100, 241, 21))
        self.dr_model_path_lineEdit.setReadOnly(True)
        self.dr_model_path_lineEdit.setObjectName("dr_model_path_lineEdit")
        self.dr_train_data_pushButton = QtWidgets.QPushButton(torchreid_ui)
        self.dr_train_data_pushButton.setGeometry(QtCore.QRect(360, 40, 21, 24))
        self.dr_train_data_pushButton.setObjectName("dr_train_data_pushButton")
        self.dr_model_name_lineEdit = QtWidgets.QLineEdit(torchreid_ui)
        self.dr_model_name_lineEdit.setGeometry(QtCore.QRect(110, 70, 241, 21))
        self.dr_model_name_lineEdit.setObjectName("dr_model_name_lineEdit")
        self.dr_model_path_pushButton = QtWidgets.QPushButton(torchreid_ui)
        self.dr_model_path_pushButton.setGeometry(QtCore.QRect(360, 100, 21, 24))
        self.dr_model_path_pushButton.setObjectName("dr_model_path_pushButton")
        self.dr_model_name_label = QtWidgets.QLabel(torchreid_ui)
        self.dr_model_name_label.setGeometry(QtCore.QRect(10, 70, 91, 16))
        self.dr_model_name_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                              QtCore.Qt.AlignmentFlag.AlignTrailing|
                                              QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.dr_model_name_label.setObjectName("dr_model_name_label")
        self.dr_train_data_label = QtWidgets.QLabel(torchreid_ui)
        self.dr_train_data_label.setGeometry(QtCore.QRect(10, 40, 91, 16))
        self.dr_train_data_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                              QtCore.Qt.AlignmentFlag.AlignTrailing|
                                              QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.dr_train_data_label.setObjectName("dr_train_data_label")
        self.dr_train_data_lineEdit = QtWidgets.QLineEdit(torchreid_ui)
        self.dr_train_data_lineEdit.setGeometry(QtCore.QRect(110, 40, 241, 21))
        self.dr_train_data_lineEdit.setReadOnly(True)
        self.dr_train_data_lineEdit.setObjectName("dr_train_data_lineEdit")
        self.dr_classifier_pkl_pushButton = QtWidgets.QPushButton(torchreid_ui)
        self.dr_classifier_pkl_pushButton.setGeometry(QtCore.QRect(360, 10, 21, 24))
        self.dr_classifier_pkl_pushButton.setObjectName("dr_classifier_pkl_pushButton")
        self.dr_classifier_pkl_lineEdit = QtWidgets.QLineEdit(torchreid_ui)
        self.dr_classifier_pkl_lineEdit.setGeometry(QtCore.QRect(110, 10, 241, 21))
        self.dr_classifier_pkl_lineEdit.setReadOnly(True)
        self.dr_classifier_pkl_lineEdit.setObjectName("dr_classifier_pkl_lineEdit")
        self.dr_devices_label = QtWidgets.QLabel(torchreid_ui)
        self.dr_devices_label.setGeometry(QtCore.QRect(10, 160, 91, 16))
        self.dr_devices_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                        QtCore.Qt.AlignmentFlag.AlignTrailing|
                                        QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.dr_devices_label.setObjectName("dr_devices_label")
        self.dr_device_lineEdit = QtWidgets.QLineEdit(torchreid_ui)
        self.dr_device_lineEdit.setGeometry(QtCore.QRect(110, 160, 241, 21))
        self.dr_device_lineEdit.setObjectName("dr_device_lineEdit")
        self.retranslateUi(torchreid_ui)
        # custom
        self.md = TorchreidModelDict()
        self.loadDR()
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
        self.dr_min_confidence_lineEdit.setText(str(self.mycfg.rcfg_torchreid.min_confidence))
        self.dr_device_lineEdit.setText(str(self.mycfg.rcfg_torchreid.device))

    def updateCFG(self, torchreid_ui):
        device = 'cuda'
        if 'cpu' in self.dr_device_lineEdit.text().lower():
            device = 'cpu'
        elif 'cuda' in self.dr_device_lineEdit.text().lower():
            device = 'cuda'
        else:
            device = getInt(self.dr_device_lineEdit.text())
        Torchreid_doc = {
            "ri_name": unified_strings.getUnifiedFormat("Torchreid"),
            "classifier_pkl": normalizePathFDS(root_dir, self.dr_classifier_pkl_lineEdit.text()),
            "train_data": normalizePathFDS(root_dir, self.dr_train_data_lineEdit.text()),
            "model_name": self.dr_model_name_lineEdit.text(),
            "model_path": normalizePathFDS(root_dir, self.dr_model_path_lineEdit.text()),
            "min_confidence": getFloat(self.dr_min_confidence_lineEdit.text(), default_val=0.35),
            "device": device
        }
        facenet_doc = self.mycfg.rcfg_facenet.getDocument()
        self.mycfg.dumpAllRCFG([facenet_doc, Torchreid_doc])
        torchreid_ui.close()

    def browseTrainData(self):
        default_path = getAncestorDir(self.mycfg.rcfg_torchreid.train_data)
        train_data = QtWidgets.QFileDialog.getExistingDirectory(None, "Train data folder", default_path)
        if train_data:
            self.dr_train_data_lineEdit.setText(train_data)

    def browseModelPath(self):
        default_path = getAncestorDir(self.mycfg.rcfg_torchreid.model_path)
        model_filter = "PyTorch model (*.pth;*.pyth;*.tar)"
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
