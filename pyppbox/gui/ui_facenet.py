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
                                       get2Dlist, getFloat, getInt)


unified_strings = UnifiedStrings()
root_dir = getGlobalRootDir()

class Ui_FaceNet(object):

    def __init__(self, cfg_mode, cfg_dir):
        self.cfg_dir = cfg_dir
        if cfg_mode == 0:
            self.mycfg = MyCFG()
        else:
            self.mycfg = MyCFG(self.cfg_dir)

    def setupUi(self, facenet_ui):
        facenet_ui.setObjectName("facenet_ui")
        facenet_ui.resize(390, 330)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, 
                                           QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(facenet_ui.sizePolicy().hasHeightForWidth())
        facenet_ui.setSizePolicy(sizePolicy)
        facenet_ui.setMinimumSize(QtCore.QSize(390, 330))
        facenet_ui.setMaximumSize(QtCore.QSize(390, 330))
        self.save_pushButton = QtWidgets.QPushButton(facenet_ui)
        self.save_pushButton.setGeometry(QtCore.QRect(150, 290, 91, 31))
        self.save_pushButton.setObjectName("save_pushButton")
        self.fn_min_confidence_label = QtWidgets.QLabel(facenet_ui)
        self.fn_min_confidence_label.setGeometry(QtCore.QRect(10, 190, 91, 16))
        self.fn_min_confidence_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                                  QtCore.Qt.AlignmentFlag.AlignTrailing|
                                                  QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_min_confidence_label.setObjectName("fn_min_confidence_label")
        self.fn_model_file_lineEdit = QtWidgets.QLineEdit(facenet_ui)
        self.fn_model_file_lineEdit.setGeometry(QtCore.QRect(110, 70, 241, 21))
        self.fn_model_file_lineEdit.setReadOnly(True)
        self.fn_model_file_lineEdit.setObjectName("fn_model_file_lineEdit")
        self.fn_classifier_pkl_label = QtWidgets.QLabel(facenet_ui)
        self.fn_classifier_pkl_label.setGeometry(QtCore.QRect(10, 100, 91, 16))
        self.fn_classifier_pkl_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                                  QtCore.Qt.AlignmentFlag.AlignTrailing|
                                                  QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_classifier_pkl_label.setObjectName("fn_classifier_pkl_label")
        self.fn_train_data_label = QtWidgets.QLabel(facenet_ui)
        self.fn_train_data_label.setGeometry(QtCore.QRect(10, 130, 91, 16))
        self.fn_train_data_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                              QtCore.Qt.AlignmentFlag.AlignTrailing|
                                              QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_train_data_label.setObjectName("dr_train_data_label")
        self.fn_yl_h_calib_label = QtWidgets.QLabel(facenet_ui)
        self.fn_yl_h_calib_label.setGeometry(QtCore.QRect(10, 220, 91, 16))
        self.fn_yl_h_calib_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                              QtCore.Qt.AlignmentFlag.AlignTrailing|
                                              QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_yl_h_calib_label.setObjectName("fn_yl_h_calib_label")
        self.fn_gpu_mem_label = QtWidgets.QLabel(facenet_ui)
        self.fn_gpu_mem_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.fn_gpu_mem_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                           QtCore.Qt.AlignmentFlag.AlignTrailing|
                                           QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_gpu_mem_label.setObjectName("fn_gpu_mem_label")
        self.fn_model_file_pushButton = QtWidgets.QPushButton(facenet_ui)
        self.fn_model_file_pushButton.setGeometry(QtCore.QRect(360, 70, 21, 24))
        self.fn_model_file_pushButton.setObjectName("fn_model_file_pushButton")
        self.fn_yl_w_calib_label = QtWidgets.QLabel(facenet_ui)
        self.fn_yl_w_calib_label.setGeometry(QtCore.QRect(10, 250, 91, 16))
        self.fn_yl_w_calib_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                              QtCore.Qt.AlignmentFlag.AlignTrailing|
                                              QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_yl_w_calib_label.setObjectName("fn_yl_w_calib_label")
        self.fn_model_det_pushButton = QtWidgets.QPushButton(facenet_ui)
        self.fn_model_det_pushButton.setGeometry(QtCore.QRect(360, 40, 21, 24))
        self.fn_model_det_pushButton.setObjectName("fn_model_det_pushButton")
        self.fn_model_det_label = QtWidgets.QLabel(facenet_ui)
        self.fn_model_det_label.setGeometry(QtCore.QRect(10, 40, 91, 16))
        self.fn_model_det_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                             QtCore.Qt.AlignmentFlag.AlignTrailing|
                                             QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_model_det_label.setObjectName("fn_model_det_label")
        self.fn_yl_w_calib_lineEdit = QtWidgets.QLineEdit(facenet_ui)
        self.fn_yl_w_calib_lineEdit.setGeometry(QtCore.QRect(110, 250, 241, 21))
        self.fn_yl_w_calib_lineEdit.setReadOnly(False)
        self.fn_yl_w_calib_lineEdit.setObjectName("fn_yl_w_calib_lineEdit")
        self.fn_batch_size_lineEdit = QtWidgets.QLineEdit(facenet_ui)
        self.fn_batch_size_lineEdit.setGeometry(QtCore.QRect(110, 160, 241, 21))
        self.fn_batch_size_lineEdit.setObjectName("fn_batch_size_lineEdit")
        self.fn_batch_size_label = QtWidgets.QLabel(facenet_ui)
        self.fn_batch_size_label.setGeometry(QtCore.QRect(10, 160, 91, 16))
        self.fn_batch_size_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                              QtCore.Qt.AlignmentFlag.AlignTrailing|
                                              QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_batch_size_label.setObjectName("fn_batch_size_label")
        self.fn_classifier_pkl_lineEdit = QtWidgets.QLineEdit(facenet_ui)
        self.fn_classifier_pkl_lineEdit.setGeometry(QtCore.QRect(110, 100, 241, 21))
        self.fn_classifier_pkl_lineEdit.setReadOnly(True)
        self.fn_classifier_pkl_lineEdit.setObjectName("fn_classifier_pkl_lineEdit")
        self.fn_train_data_lineEdit = QtWidgets.QLineEdit(facenet_ui)
        self.fn_train_data_lineEdit.setGeometry(QtCore.QRect(110, 130, 241, 21))
        self.fn_train_data_lineEdit.setReadOnly(True)
        self.fn_train_data_lineEdit.setObjectName("dr_train_data_lineEdit")
        self.fn_model_file_label = QtWidgets.QLabel(facenet_ui)
        self.fn_model_file_label.setGeometry(QtCore.QRect(10, 70, 91, 16))
        self.fn_model_file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                              QtCore.Qt.AlignmentFlag.AlignTrailing|
                                              QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_model_file_label.setObjectName("fn_model_file_label")
        self.fn_gpu_mem_lineEdit = QtWidgets.QLineEdit(facenet_ui)
        self.fn_gpu_mem_lineEdit.setGeometry(QtCore.QRect(110, 10, 241, 21))
        self.fn_gpu_mem_lineEdit.setObjectName("fn_gpu_mem_lineEdit")
        self.fn_min_confidence_lineEdit = QtWidgets.QLineEdit(facenet_ui)
        self.fn_min_confidence_lineEdit.setGeometry(QtCore.QRect(110, 190, 241, 21))
        self.fn_min_confidence_lineEdit.setReadOnly(False)
        self.fn_min_confidence_lineEdit.setObjectName("fn_min_confidence_lineEdit")
        self.fn_classifier_pkl_pushButton = QtWidgets.QPushButton(facenet_ui)
        self.fn_classifier_pkl_pushButton.setGeometry(QtCore.QRect(360, 100, 21, 24))
        self.fn_classifier_pkl_pushButton.setObjectName("fn_classifier_pkl_pushButton")
        self.fn_train_data_pushButton = QtWidgets.QPushButton(facenet_ui)
        self.fn_train_data_pushButton.setGeometry(QtCore.QRect(360, 130, 21, 24))
        self.fn_train_data_pushButton.setObjectName("dr_train_data_pushButton")
        self.fn_yl_h_calib_lineEdit = QtWidgets.QLineEdit(facenet_ui)
        self.fn_yl_h_calib_lineEdit.setGeometry(QtCore.QRect(110, 220, 241, 21))
        self.fn_yl_h_calib_lineEdit.setObjectName("fn_yl_h_calib_lineEdit")
        self.fn_model_det_lineEdit = QtWidgets.QLineEdit(facenet_ui)
        self.fn_model_det_lineEdit.setGeometry(QtCore.QRect(110, 40, 241, 21))
        self.fn_model_det_lineEdit.setReadOnly(True)
        self.fn_model_det_lineEdit.setObjectName("fn_model_det_lineEdit")
        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)
        self.retranslateUi(facenet_ui)
        # custom
        self.loadFN()
        self.fn_model_det_pushButton.clicked.connect(self.browseModelDet)
        self.fn_model_file_pushButton.clicked.connect(self.browseModelFile)
        self.fn_classifier_pkl_pushButton.clicked.connect(self.browseClassifierFile)
        self.fn_train_data_pushButton.clicked.connect(self.browseTrainData)
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(facenet_ui))
        QtCore.QMetaObject.connectSlotsByName(facenet_ui)

    def retranslateUi(self, facenet_ui):
        _translate = QtCore.QCoreApplication.translate
        facenet_ui.setWindowTitle(_translate("facenet_ui", "FaceNet"))
        self.save_pushButton.setText(_translate("facenet_ui", "Save"))
        self.fn_min_confidence_label.setText(_translate("facenet_ui", "min_confidence"))
        self.fn_classifier_pkl_label.setText(_translate("facenet_ui", "classifier_pkl"))
        self.fn_train_data_label.setText(_translate("facenet_ui", "train_data"))
        self.fn_yl_h_calib_label.setText(_translate("facenet_ui", "yl_h_calib"))
        self.fn_gpu_mem_label.setText(_translate("facenet_ui", "gpu_mem"))
        self.fn_model_file_pushButton.setText(_translate("facenet_ui", "..."))
        self.fn_yl_w_calib_label.setText(_translate("facenet_ui", "yl_w_calib"))
        self.fn_model_det_pushButton.setText(_translate("facenet_ui", "..."))
        self.fn_model_det_label.setText(_translate("facenet_ui", "model_det"))
        self.fn_batch_size_label.setText(_translate("facenet_ui", "batch_size"))
        self.fn_model_file_label.setText(_translate("facenet_ui", "model_file"))
        self.fn_classifier_pkl_pushButton.setText(_translate("facenet_ui", "..."))
        self.fn_train_data_pushButton.setText(_translate("facenet_ui", "..."))

    def loadFN(self):
        self.mycfg.setAllRCFG()
        self.fn_batch_size_lineEdit.setText(str(self.mycfg.rcfg_facenet.batch_size))
        self.fn_gpu_mem_lineEdit.setText(str(self.mycfg.rcfg_facenet.gpu_mem))
        self.fn_min_confidence_lineEdit.setText(str(self.mycfg.rcfg_facenet.min_confidence))
        self.fn_classifier_pkl_lineEdit.setText(getAbsPathFDS(self.mycfg.rcfg_facenet.classifier_pkl))
        self.fn_train_data_lineEdit.setText(getAbsPathFDS(self.mycfg.rcfg_facenet.train_data))
        self.fn_model_file_lineEdit.setText(getAbsPathFDS(self.mycfg.rcfg_facenet.model_file))
        self.fn_model_det_lineEdit.setText(getAbsPathFDS(self.mycfg.rcfg_facenet.model_det))
        self.fn_yl_h_calib_lineEdit.setText(str(self.mycfg.rcfg_facenet.yl_h_calibration))
        self.fn_yl_w_calib_lineEdit.setText(str(self.mycfg.rcfg_facenet.yl_w_calibration))

    def updateCFG(self, facenet_ui):
        facenet_doc = {
            "ri_name": unified_strings.getUnifiedFormat("FaceNet"),
            "gpu_mem": getFloat(self.fn_gpu_mem_lineEdit.text(), default_val=0.585),
            "model_det": normalizePathFDS(root_dir, self.fn_model_det_lineEdit.text()), 
            "model_file": normalizePathFDS(root_dir, self.fn_model_file_lineEdit.text()),
            "classifier_pkl": normalizePathFDS(root_dir, self.fn_classifier_pkl_lineEdit.text()),
            "train_data": normalizePathFDS(root_dir, self.fn_train_data_lineEdit.text()),
            "batch_size": getInt(self.fn_batch_size_lineEdit.text(), default_val=0.5),
            "min_confidence": getFloat(self.fn_min_confidence_lineEdit.text(), default_val=0.75),
            "yl_h_calibration": get2Dlist(self.fn_yl_h_calib_lineEdit.text()),
            "yl_w_calibration": get2Dlist(self.fn_yl_w_calib_lineEdit.text())
        }
        deepreid_doc = self.mycfg.rcfg_torchreid.getDocument()
        self.mycfg.dumpAllRCFG([facenet_doc, deepreid_doc])
        facenet_ui.close()

    def browseModelDet(self):
        default_path = getAncestorDir(self.mycfg.rcfg_facenet.model_det)
        model_det = QtWidgets.QFileDialog.getExistingDirectory(
            None, 
            "Det folder (det1.npy, det2.npy, det3.npy)", 
            default_path
        )
        if model_det:
            self.fn_model_det_lineEdit.setText(model_det)

    def browseModelFile(self):
        default_path = getAncestorDir(self.mycfg.rcfg_facenet.model_file)
        model_filter = "Protobuf (*.pb)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Model file", 
                                                               default_path, model_filter)
        if source_file:
            self.fn_model_file_lineEdit.setText(source_file)

    def browseClassifierFile(self):
        default_path = getAncestorDir(self.mycfg.rcfg_facenet.classifier_pkl)
        pkl_filter = "Pickle (*.pkl)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Classifier file", 
                                                               default_path, pkl_filter)
        if source_file:
            self.fn_classifier_pkl_lineEdit.setText(source_file)

    def browseTrainData(self):
        default_path = getAncestorDir(self.mycfg.rcfg_facenet.train_data)
        train_data = QtWidgets.QFileDialog.getExistingDirectory(None, "Train data folder", default_path)
        if train_data:
            self.fn_train_data_lineEdit.setText(train_data)
