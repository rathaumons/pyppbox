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
from pyppbox.utils.mytools import getAbsPathFDS, normalizePathFDS, joinFPathFull, get2Dlist

root_dir = os.path.dirname(__file__)
cfg_dir = joinFPathFull(root_dir, 'cfg')


class Ui_FacenetForm(object):
    
    def setupUi(self, FacenetForm):
        FacenetForm.setObjectName("FacenetForm")
        FacenetForm.resize(390, 360)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(FacenetForm.sizePolicy().hasHeightForWidth())
        FacenetForm.setSizePolicy(sizePolicy)
        FacenetForm.setMinimumSize(QtCore.QSize(390, 360))
        FacenetForm.setMaximumSize(QtCore.QSize(390, 360))
        self.save_pushButton = QtWidgets.QPushButton(FacenetForm)
        self.save_pushButton.setGeometry(QtCore.QRect(150, 320, 91, 31))
        self.save_pushButton.setObjectName("save_pushButton")
        self.fn_op_h_callib_label = QtWidgets.QLabel(FacenetForm)
        self.fn_op_h_callib_label.setGeometry(QtCore.QRect(10, 250, 91, 16))
        self.fn_op_h_callib_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_op_h_callib_label.setObjectName("fn_op_h_callib_label")
        self.fn_min_confidence_label = QtWidgets.QLabel(FacenetForm)
        self.fn_min_confidence_label.setGeometry(QtCore.QRect(10, 160, 91, 16))
        self.fn_min_confidence_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_min_confidence_label.setObjectName("fn_min_confidence_label")
        self.fn_model_file_lineEdit = QtWidgets.QLineEdit(FacenetForm)
        self.fn_model_file_lineEdit.setGeometry(QtCore.QRect(110, 70, 241, 21))
        self.fn_model_file_lineEdit.setReadOnly(True)
        self.fn_model_file_lineEdit.setObjectName("fn_model_file_lineEdit")
        self.fn_classifier_file_label = QtWidgets.QLabel(FacenetForm)
        self.fn_classifier_file_label.setGeometry(QtCore.QRect(10, 100, 91, 16))
        self.fn_classifier_file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_classifier_file_label.setObjectName("fn_classifier_file_label")
        self.fn_yl_h_callib_label = QtWidgets.QLabel(FacenetForm)
        self.fn_yl_h_callib_label.setGeometry(QtCore.QRect(10, 190, 91, 16))
        self.fn_yl_h_callib_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_yl_h_callib_label.setObjectName("fn_yl_h_callib_label")
        self.fn_gpu_mem_label = QtWidgets.QLabel(FacenetForm)
        self.fn_gpu_mem_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.fn_gpu_mem_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_gpu_mem_label.setObjectName("fn_gpu_mem_label")
        self.fn_model_file_pushButton = QtWidgets.QPushButton(FacenetForm)
        self.fn_model_file_pushButton.setGeometry(QtCore.QRect(360, 70, 21, 24))
        self.fn_model_file_pushButton.setObjectName("fn_model_file_pushButton")
        self.fn_op_h_callib_lineEdit = QtWidgets.QLineEdit(FacenetForm)
        self.fn_op_h_callib_lineEdit.setGeometry(QtCore.QRect(110, 250, 241, 21))
        self.fn_op_h_callib_lineEdit.setObjectName("fn_op_h_callib_lineEdit")
        self.fn_yl_w_callib_label = QtWidgets.QLabel(FacenetForm)
        self.fn_yl_w_callib_label.setGeometry(QtCore.QRect(10, 220, 91, 16))
        self.fn_yl_w_callib_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_yl_w_callib_label.setObjectName("fn_yl_w_callib_label")
        self.fn_model_det_pushButton = QtWidgets.QPushButton(FacenetForm)
        self.fn_model_det_pushButton.setGeometry(QtCore.QRect(360, 40, 21, 24))
        self.fn_model_det_pushButton.setObjectName("fn_model_det_pushButton")
        self.fn_model_det_label = QtWidgets.QLabel(FacenetForm)
        self.fn_model_det_label.setGeometry(QtCore.QRect(10, 40, 91, 16))
        self.fn_model_det_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_model_det_label.setObjectName("fn_model_det_label")
        self.fn_yl_w_callib_lineEdit = QtWidgets.QLineEdit(FacenetForm)
        self.fn_yl_w_callib_lineEdit.setGeometry(QtCore.QRect(110, 220, 241, 21))
        self.fn_yl_w_callib_lineEdit.setReadOnly(False)
        self.fn_yl_w_callib_lineEdit.setObjectName("fn_yl_w_callib_lineEdit")
        self.fn_batch_size_lineEdit = QtWidgets.QLineEdit(FacenetForm)
        self.fn_batch_size_lineEdit.setGeometry(QtCore.QRect(110, 130, 241, 21))
        self.fn_batch_size_lineEdit.setObjectName("fn_batch_size_lineEdit")
        self.fn_batch_size_label = QtWidgets.QLabel(FacenetForm)
        self.fn_batch_size_label.setGeometry(QtCore.QRect(10, 130, 91, 16))
        self.fn_batch_size_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_batch_size_label.setObjectName("fn_batch_size_label")
        self.fn_classifier_file_lineEdit = QtWidgets.QLineEdit(FacenetForm)
        self.fn_classifier_file_lineEdit.setGeometry(QtCore.QRect(110, 100, 241, 21))
        self.fn_classifier_file_lineEdit.setReadOnly(True)
        self.fn_classifier_file_lineEdit.setObjectName("fn_classifier_file_lineEdit")
        self.fn_model_file_label = QtWidgets.QLabel(FacenetForm)
        self.fn_model_file_label.setGeometry(QtCore.QRect(10, 70, 91, 16))
        self.fn_model_file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_model_file_label.setObjectName("fn_model_file_label")
        self.fn_gpu_mem_lineEdit = QtWidgets.QLineEdit(FacenetForm)
        self.fn_gpu_mem_lineEdit.setGeometry(QtCore.QRect(110, 10, 241, 21))
        self.fn_gpu_mem_lineEdit.setObjectName("fn_gpu_mem_lineEdit")
        self.fn_min_confidence_lineEdit = QtWidgets.QLineEdit(FacenetForm)
        self.fn_min_confidence_lineEdit.setGeometry(QtCore.QRect(110, 160, 241, 21))
        self.fn_min_confidence_lineEdit.setReadOnly(False)
        self.fn_min_confidence_lineEdit.setObjectName("fn_min_confidence_lineEdit")
        self.fn_classifier_file_pushButton = QtWidgets.QPushButton(FacenetForm)
        self.fn_classifier_file_pushButton.setGeometry(QtCore.QRect(360, 100, 21, 24))
        self.fn_classifier_file_pushButton.setObjectName("fn_classifier_file_pushButton")
        self.fn_op_w_callib_lineEdit = QtWidgets.QLineEdit(FacenetForm)
        self.fn_op_w_callib_lineEdit.setGeometry(QtCore.QRect(110, 280, 241, 21))
        self.fn_op_w_callib_lineEdit.setReadOnly(False)
        self.fn_op_w_callib_lineEdit.setObjectName("fn_op_w_callib_lineEdit")
        self.fn_op_w_callib_label = QtWidgets.QLabel(FacenetForm)
        self.fn_op_w_callib_label.setGeometry(QtCore.QRect(10, 280, 91, 16))
        self.fn_op_w_callib_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.fn_op_w_callib_label.setObjectName("fn_op_w_callib_label")
        self.fn_yl_h_callib_lineEdit = QtWidgets.QLineEdit(FacenetForm)
        self.fn_yl_h_callib_lineEdit.setGeometry(QtCore.QRect(110, 190, 241, 21))
        self.fn_yl_h_callib_lineEdit.setObjectName("fn_yl_h_callib_lineEdit")
        self.fn_model_det_lineEdit = QtWidgets.QLineEdit(FacenetForm)
        self.fn_model_det_lineEdit.setGeometry(QtCore.QRect(110, 40, 241, 21))
        self.fn_model_det_lineEdit.setReadOnly(True)
        self.fn_model_det_lineEdit.setObjectName("fn_model_det_lineEdit")

        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)

        # custom

        self.loadCFG()
        self.loadFN()

        self.fn_model_det_pushButton.clicked.connect(self.browseModelDet)
        self.fn_model_file_pushButton.clicked.connect(self.browseModelFile)
        self.fn_classifier_file_pushButton.clicked.connect(self.browseClassifierFile)
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(FacenetForm))

        self.retranslateUi(FacenetForm)
        QtCore.QMetaObject.connectSlotsByName(FacenetForm)


    def retranslateUi(self, FacenetForm):
        _translate = QtCore.QCoreApplication.translate
        FacenetForm.setWindowTitle(_translate("FacenetForm", "Facenet"))
        self.save_pushButton.setText(_translate("FacenetForm", "Save"))
        self.fn_op_h_callib_label.setText(_translate("FacenetForm", "op_h_callib"))
        self.fn_min_confidence_label.setText(_translate("FacenetForm", "min_confidence"))
        self.fn_classifier_file_label.setText(_translate("FacenetForm", "classifier_file"))
        self.fn_yl_h_callib_label.setText(_translate("FacenetForm", "yl_h_callib"))
        self.fn_gpu_mem_label.setText(_translate("FacenetForm", "gpu_mem"))
        self.fn_model_file_pushButton.setText(_translate("FacenetForm", "..."))
        self.fn_yl_w_callib_label.setText(_translate("FacenetForm", "yl_w_callib"))
        self.fn_model_det_pushButton.setText(_translate("FacenetForm", "..."))
        self.fn_model_det_label.setText(_translate("FacenetForm", "model_det"))
        self.fn_batch_size_label.setText(_translate("FacenetForm", "batch_size"))
        self.fn_model_file_label.setText(_translate("FacenetForm", "model_file"))
        self.fn_classifier_file_pushButton.setText(_translate("FacenetForm", "..."))
        self.fn_op_w_callib_label.setText(_translate("FacenetForm", "op_w_callib"))


    def loadCFG(self):
        self.mycfg = MyConfigurator()
        self.mycfg.loadRCFG()


    def loadFN(self):
        self.fn_batch_size_lineEdit.setText(str(self.mycfg.rcfg_facenet.batch_size))
        self.fn_gpu_mem_lineEdit.setText(str(self.mycfg.rcfg_facenet.gpu_mem))
        self.fn_min_confidence_lineEdit.setText(str(self.mycfg.rcfg_facenet.min_confidence))
        self.fn_classifier_file_lineEdit.setText(getAbsPathFDS(self.mycfg.rcfg_facenet.classifier_file))
        self.fn_model_file_lineEdit.setText(getAbsPathFDS(self.mycfg.rcfg_facenet.model_file))
        self.fn_model_det_lineEdit.setText(getAbsPathFDS(self.mycfg.rcfg_facenet.model_det))
        self.fn_yl_h_callib_lineEdit.setText(str(self.mycfg.rcfg_facenet.yl_h_callibration))
        self.fn_yl_w_callib_lineEdit.setText(str(self.mycfg.rcfg_facenet.yl_w_callibration))
        self.fn_op_h_callib_lineEdit.setText(str(self.mycfg.rcfg_facenet.op_h_callibration))
        self.fn_op_w_callib_lineEdit.setText(str(self.mycfg.rcfg_facenet.op_w_callibration))


    def updateCFG(self, FacenetForm):
        facenet_doc = {"ri_name": "Facenet",
                        "gpu_mem": float(self.fn_gpu_mem_lineEdit.text()),
                        "model_det": normalizePathFDS(root_dir, self.fn_model_det_lineEdit.text()), 
                        "model_file": normalizePathFDS(root_dir, self.fn_model_file_lineEdit.text()),
                        "classifier_file": normalizePathFDS(root_dir, self.fn_classifier_file_lineEdit.text()),
                        "batch_size": int(self.fn_batch_size_lineEdit.text()),
                        "min_confidence": float(self.fn_min_confidence_lineEdit.text()),
                        "yl_h_callibration": get2Dlist(self.fn_yl_h_callib_lineEdit.text()),
                        "yl_w_callibration": get2Dlist(self.fn_yl_w_callib_lineEdit.text()),
                        "op_h_callibration": get2Dlist(self.fn_op_h_callib_lineEdit.text()),
                        "op_w_callibration": get2Dlist(self.fn_op_w_callib_lineEdit.text())}
        deepreid_doc = self.mycfg.rcfg_deepreid.getDocument()
        cfgio = MyCFGIO()
        cfgio.dumpReidersWithHeader([facenet_doc, deepreid_doc])
        FacenetForm.close()


    def browseModelDet(self):
        default_path = joinFPathFull(root_dir, 'ri_facenet/models/det')
        model_det = QtWidgets.QFileDialog.getExistingDirectory(None, "Det folder (det1.npy, det2.npy, det3.npy)", default_path)
        if model_det:
            self.fn_model_det_lineEdit.setText(model_det)


    def browseModelFile(self):
        default_path = joinFPathFull(root_dir, 'ri_facenet/models/20180402-114759')
        model_filter = "Protobuf (*.pb)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Model file", default_path, model_filter)
        if source_file:
            self.fn_model_file_lineEdit.setText(source_file)


    def browseClassifierFile(self):
        default_path = joinFPathFull(root_dir, 'ri_facenet/classifier')
        pkl_filter = "Pickle (*.pkl)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Classifier file", default_path, pkl_filter)
        if source_file:
            self.fn_classifier_file_lineEdit.setText(source_file)
