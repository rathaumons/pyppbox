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
from pyppbox.utils.mytools import getAbsPathFDS, extendPathFDS, normalizePathFDS, getFileName, joinFPathFull
from pyppbox.utils.deepreid_model_dict import ModelDictionary

root_dir = os.path.dirname(__file__)
cfg_dir = joinFPathFull(root_dir, 'cfg')


class Ui_DeepReIDForm(object):


    def setupUi(self, DeepReIDForm):
        DeepReIDForm.setObjectName("DeepReIDForm")
        DeepReIDForm.resize(390, 240)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(DeepReIDForm.sizePolicy().hasHeightForWidth())
        DeepReIDForm.setSizePolicy(sizePolicy)
        DeepReIDForm.setMinimumSize(QtCore.QSize(390, 240))
        DeepReIDForm.setMaximumSize(QtCore.QSize(390, 240))
        self.dr_model_path_label = QtWidgets.QLabel(DeepReIDForm)
        self.dr_model_path_label.setGeometry(QtCore.QRect(10, 130, 91, 16))
        self.dr_model_path_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.dr_model_path_label.setObjectName("dr_model_path_label")
        self.dr_min_confidence_lineEdit = QtWidgets.QLineEdit(DeepReIDForm)
        self.dr_min_confidence_lineEdit.setGeometry(QtCore.QRect(110, 160, 241, 21))
        self.dr_min_confidence_lineEdit.setReadOnly(False)
        self.dr_min_confidence_lineEdit.setObjectName("dr_min_confidence_lineEdit")
        self.save_pushButton = QtWidgets.QPushButton(DeepReIDForm)
        self.save_pushButton.setGeometry(QtCore.QRect(150, 200, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)
        self.save_pushButton.setObjectName("save_pushButton")
        self.dr_min_confidence_label = QtWidgets.QLabel(DeepReIDForm)
        self.dr_min_confidence_label.setGeometry(QtCore.QRect(10, 160, 91, 16))
        self.dr_min_confidence_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.dr_min_confidence_label.setObjectName("dr_min_confidence_label")
        self.dr_classes_txt_pushButton = QtWidgets.QPushButton(DeepReIDForm)
        self.dr_classes_txt_pushButton.setGeometry(QtCore.QRect(360, 10, 21, 24))
        self.dr_classes_txt_pushButton.setObjectName("dr_classes_txt_pushButton")
        self.dr_classifier_pkl_label = QtWidgets.QLabel(DeepReIDForm)
        self.dr_classifier_pkl_label.setGeometry(QtCore.QRect(10, 40, 91, 16))
        self.dr_classifier_pkl_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.dr_classifier_pkl_label.setObjectName("dr_classifier_pkl_label")
        self.dr_model_path_lineEdit = QtWidgets.QLineEdit(DeepReIDForm)
        self.dr_model_path_lineEdit.setGeometry(QtCore.QRect(110, 130, 241, 21))
        self.dr_model_path_lineEdit.setReadOnly(True)
        self.dr_model_path_lineEdit.setObjectName("dr_model_path_lineEdit")
        self.dr_train_data_pushButton = QtWidgets.QPushButton(DeepReIDForm)
        self.dr_train_data_pushButton.setGeometry(QtCore.QRect(360, 70, 21, 24))
        self.dr_train_data_pushButton.setObjectName("dr_train_data_pushButton")
        self.dr_model_name_lineEdit = QtWidgets.QLineEdit(DeepReIDForm)
        self.dr_model_name_lineEdit.setGeometry(QtCore.QRect(110, 100, 241, 21))
        self.dr_model_name_lineEdit.setObjectName("dr_model_name_lineEdit")
        self.dr_model_path_pushButton = QtWidgets.QPushButton(DeepReIDForm)
        self.dr_model_path_pushButton.setGeometry(QtCore.QRect(360, 130, 21, 24))
        self.dr_model_path_pushButton.setObjectName("dr_model_path_pushButton")
        self.dr_model_name_label = QtWidgets.QLabel(DeepReIDForm)
        self.dr_model_name_label.setGeometry(QtCore.QRect(10, 100, 91, 16))
        self.dr_model_name_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.dr_model_name_label.setObjectName("dr_model_name_label")
        self.dr_train_data_label = QtWidgets.QLabel(DeepReIDForm)
        self.dr_train_data_label.setGeometry(QtCore.QRect(10, 70, 91, 16))
        self.dr_train_data_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.dr_train_data_label.setObjectName("dr_train_data_label")
        self.dr_classes_txt_lineEdit = QtWidgets.QLineEdit(DeepReIDForm)
        self.dr_classes_txt_lineEdit.setGeometry(QtCore.QRect(110, 10, 241, 21))
        self.dr_classes_txt_lineEdit.setReadOnly(True)
        self.dr_classes_txt_lineEdit.setObjectName("dr_classes_txt_lineEdit")
        self.dr_train_data_lineEdit = QtWidgets.QLineEdit(DeepReIDForm)
        self.dr_train_data_lineEdit.setGeometry(QtCore.QRect(110, 70, 241, 21))
        self.dr_train_data_lineEdit.setReadOnly(True)
        self.dr_train_data_lineEdit.setObjectName("dr_train_data_lineEdit")
        self.dr_classifier_pkl_pushButton = QtWidgets.QPushButton(DeepReIDForm)
        self.dr_classifier_pkl_pushButton.setGeometry(QtCore.QRect(360, 40, 21, 24))
        self.dr_classifier_pkl_pushButton.setObjectName("dr_classifier_pkl_pushButton")
        self.dr_classes_txt_label = QtWidgets.QLabel(DeepReIDForm)
        self.dr_classes_txt_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.dr_classes_txt_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.dr_classes_txt_label.setObjectName("dr_classes_txt_label")
        self.dr_classifier_pkl_lineEdit = QtWidgets.QLineEdit(DeepReIDForm)
        self.dr_classifier_pkl_lineEdit.setGeometry(QtCore.QRect(110, 40, 241, 21))
        self.dr_classifier_pkl_lineEdit.setReadOnly(True)
        self.dr_classifier_pkl_lineEdit.setObjectName("dr_classifier_pkl_lineEdit")

        # custom

        self.md = ModelDictionary()
        self.md.loadCFG(joinFPathFull(cfg_dir, "deepreid_model_dict.yaml"))
        
        self.loadCFG()
        self.loadDR()

        self.dr_classes_txt_pushButton.clicked.connect(self.browseClassesTXT)
        self.dr_classifier_pkl_pushButton.clicked.connect(self.browseClassifierPKL)
        self.dr_train_data_pushButton.clicked.connect(self.browseTrainData)
        self.dr_model_path_pushButton.clicked.connect(self.browseModelPath)
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(DeepReIDForm))

        self.retranslateUi(DeepReIDForm)
        QtCore.QMetaObject.connectSlotsByName(DeepReIDForm)


    def retranslateUi(self, DeepReIDForm):
        _translate = QtCore.QCoreApplication.translate
        DeepReIDForm.setWindowTitle(_translate("DeepReIDForm", "DeepReID"))
        self.dr_model_path_label.setText(_translate("DeepReIDForm", "model_path"))
        self.save_pushButton.setText(_translate("DeepReIDForm", "Save"))
        self.dr_min_confidence_label.setText(_translate("DeepReIDForm", "min_confidence"))
        self.dr_classes_txt_pushButton.setText(_translate("DeepReIDForm", "..."))
        self.dr_classifier_pkl_label.setText(_translate("DeepReIDForm", "classifier_pkl"))
        self.dr_train_data_pushButton.setText(_translate("DeepReIDForm", "..."))
        self.dr_model_path_pushButton.setText(_translate("DeepReIDForm", "..."))
        self.dr_model_name_label.setText(_translate("DeepReIDForm", "model_name"))
        self.dr_train_data_label.setText(_translate("DeepReIDForm", "train_data"))
        self.dr_classifier_pkl_pushButton.setText(_translate("DeepReIDForm", "..."))
        self.dr_classes_txt_label.setText(_translate("DeepReIDForm", "classes_txt"))


    def loadCFG(self):
        self.mycfg = MyConfigurator()
        self.mycfg.loadRCFG()


    def loadDR(self):
        self.dr_classes_txt_lineEdit.setText(getAbsPathFDS(self.mycfg.rcfg_deepreid.classes_txt))
        self.dr_classifier_pkl_lineEdit.setText(getAbsPathFDS(self.mycfg.rcfg_deepreid.classifier_pkl))
        self.dr_train_data_lineEdit.setText(getAbsPathFDS(self.mycfg.rcfg_deepreid.train_data))
        self.dr_model_name_lineEdit.setText(str(self.mycfg.rcfg_deepreid.model_name))
        self.dr_model_path_lineEdit.setText(getAbsPathFDS(self.mycfg.rcfg_deepreid.model_path))
        self.dr_min_confidence_lineEdit.setText(str(self.mycfg.rcfg_deepreid.min_confidence))


    def updateCFG(self, DeepReIDForm):
        deepreid_doc = {"ri_name": "DeepReID",
                       "classes_txt": normalizePathFDS(root_dir, self.dr_classes_txt_lineEdit.text()),
                       "classifier_pkl": normalizePathFDS(root_dir, self.dr_classifier_pkl_lineEdit.text()),
                       "train_data": normalizePathFDS(root_dir, self.dr_train_data_lineEdit.text()),
                       "model_name": self.dr_model_name_lineEdit.text(),
                       "model_path": normalizePathFDS(root_dir, self.dr_model_path_lineEdit.text()),
                       "min_confidence": float(self.dr_min_confidence_lineEdit.text())}
        facenet_doc = self.mycfg.rcfg_facenet.getDocument()
        cfgio = MyCFGIO()
        cfgio.dumpReidersWithHeader([facenet_doc, deepreid_doc])
        DeepReIDForm.close()


    def browseTrainData(self):
        default_path = extendPathFDS(root_dir, 'ri_deepreid/data/train')
        train_data = QtWidgets.QFileDialog.getExistingDirectory(None, "Train data folder", default_path)
        if train_data:
            self.dr_train_data_lineEdit.setText(train_data)


    def browseModelPath(self):
        default_path = extendPathFDS(root_dir, 'ri_deepreid/pretrained')
        model_filter = "PyTorch model (*.pth;*.pyth;*.tar)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Model file", default_path, model_filter)
        if source_file:
            self.dr_model_path_lineEdit.setText(source_file)
            self.autoSetModelName(source_file)


    def browseClassifierPKL(self):
        default_path = extendPathFDS(root_dir, 'ri_deepreid/classifier')
        pkl_filter = "Pickle (*.pkl)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Classifier file", default_path, pkl_filter)
        if source_file:
            self.dr_classifier_pkl_lineEdit.setText(source_file)


    def browseClassesTXT(self):
        default_path = joinFPathFull(root_dir, 'ri_deepreid/classifier')
        txt_filter = "Text (*.txt)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Classes file", default_path, txt_filter)
        if source_file:
            self.dr_classes_txt_lineEdit.setText(source_file)


    def autoSetModelName(self, model_path, enable=True):
        if enable:
            self.dr_model_name_lineEdit.setText(self.md.findModelArch(getFileName(model_path)))
        
