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
from pyppbox.utils.mytools import getAbsPathFDS, normalizePathFDS, joinFPathFull

root_dir = os.path.dirname(__file__)
cfg_dir = joinFPathFull(root_dir, 'cfg')


class Ui_DeepSORTForm(object):

    def setupUi(self, DeepSORTForm):
        DeepSORTForm.setObjectName("DeepSORTForm")
        DeepSORTForm.resize(390, 180)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(DeepSORTForm.sizePolicy().hasHeightForWidth())
        DeepSORTForm.setSizePolicy(sizePolicy)
        DeepSORTForm.setMinimumSize(QtCore.QSize(390, 180))
        DeepSORTForm.setMaximumSize(QtCore.QSize(390, 180))
        self.save_pushButton = QtWidgets.QPushButton(DeepSORTForm)
        self.save_pushButton.setGeometry(QtCore.QRect(150, 140, 91, 31))
        self.save_pushButton.setObjectName("save_pushButton")
        self.ds_max_overlap_lineEdit = QtWidgets.QLineEdit(DeepSORTForm)
        self.ds_max_overlap_lineEdit.setGeometry(QtCore.QRect(110, 40, 241, 21))
        self.ds_max_overlap_lineEdit.setObjectName("ds_max_overlap_lineEdit")
        self.ds_model_file_pushButton = QtWidgets.QPushButton(DeepSORTForm)
        self.ds_model_file_pushButton.setGeometry(QtCore.QRect(360, 100, 21, 24))
        self.ds_model_file_pushButton.setObjectName("ds_model_file_pushButton")
        self.ds_cosine_distance_label = QtWidgets.QLabel(DeepSORTForm)
        self.ds_cosine_distance_label.setGeometry(QtCore.QRect(10, 70, 91, 16))
        self.ds_cosine_distance_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ds_cosine_distance_label.setObjectName("ds_cosine_distance_label")
        self.ds_model_file_lineEdit = QtWidgets.QLineEdit(DeepSORTForm)
        self.ds_model_file_lineEdit.setGeometry(QtCore.QRect(110, 100, 241, 21))
        self.ds_model_file_lineEdit.setReadOnly(True)
        self.ds_model_file_lineEdit.setObjectName("ds_model_file_lineEdit")
        self.ds_nn_budget_lineEdit = QtWidgets.QLineEdit(DeepSORTForm)
        self.ds_nn_budget_lineEdit.setGeometry(QtCore.QRect(110, 10, 241, 21))
        self.ds_nn_budget_lineEdit.setObjectName("ds_nn_budget_lineEdit")
        self.ds_model_file_label = QtWidgets.QLabel(DeepSORTForm)
        self.ds_model_file_label.setGeometry(QtCore.QRect(10, 100, 91, 16))
        self.ds_model_file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ds_model_file_label.setObjectName("ds_model_file_label")
        self.ds_cosine_distance_lineEdit = QtWidgets.QLineEdit(DeepSORTForm)
        self.ds_cosine_distance_lineEdit.setGeometry(QtCore.QRect(110, 70, 241, 21))
        self.ds_cosine_distance_lineEdit.setObjectName("ds_cosine_distance_lineEdit")
        self.ds_nn_budget_label = QtWidgets.QLabel(DeepSORTForm)
        self.ds_nn_budget_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.ds_nn_budget_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ds_nn_budget_label.setObjectName("ds_nn_budget_label")
        self.ds_max_overlap_label = QtWidgets.QLabel(DeepSORTForm)
        self.ds_max_overlap_label.setGeometry(QtCore.QRect(10, 40, 91, 16))
        self.ds_max_overlap_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ds_max_overlap_label.setObjectName("ds_max_overlap_label")

        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)

        # custom 
        self.loadCFG()
        self.loadDS()

        self.ds_model_file_pushButton.clicked.connect(self.browseModelFile)
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(DeepSORTForm))

        self.retranslateUi(DeepSORTForm)
        QtCore.QMetaObject.connectSlotsByName(DeepSORTForm)


    def retranslateUi(self, DeepSORTForm):
        _translate = QtCore.QCoreApplication.translate
        DeepSORTForm.setWindowTitle(_translate("DeepSORTForm", "DeepSORT"))
        self.save_pushButton.setText(_translate("DeepSORTForm", "Save"))
        self.ds_model_file_pushButton.setText(_translate("DeepSORTForm", "..."))
        self.ds_cosine_distance_label.setText(_translate("DeepSORTForm", "cosine_distance"))
        self.ds_model_file_label.setText(_translate("DeepSORTForm", "model_file"))
        self.ds_nn_budget_label.setText(_translate("DeepSORTForm", "nn_budget"))
        self.ds_max_overlap_label.setText(_translate("DeepSORTForm", "max_overlap"))


    def loadCFG(self):
        self.mycfg = MyConfigurator()
        self.mycfg.loadTCFG()


    def loadDS(self):
        self.ds_cosine_distance_lineEdit.setText(str(self.mycfg.tcfg_deepsort.max_cosine_distance))
        self.ds_max_overlap_lineEdit.setText(str(self.mycfg.tcfg_deepsort.nms_max_overlap))
        self.ds_nn_budget_lineEdit.setText(str(self.mycfg.tcfg_deepsort.nn_budget))
        self.ds_model_file_lineEdit.setText(getAbsPathFDS(self.mycfg.tcfg_deepsort.model_file))


    def updateCFG(self, YOLOForm):
        deepsort_doc = {"tk_name": "DeepSORT",
                        "nn_budget": int(self.ds_nn_budget_lineEdit.text()),
                        "nms_max_overlap": float(self.ds_max_overlap_lineEdit.text()),
                        "max_cosine_distance": float(self.ds_cosine_distance_lineEdit.text()),
                        "model_file": normalizePathFDS(root_dir, self.ds_model_file_lineEdit.text())}
        centroid_doc = self.mycfg.tcfg_centroid.getDocument()
        sort_doc = self.mycfg.tcfg_sort.getDocument()
        cfgio = MyCFGIO()
        cfgio.dumpTrackersWithHeader([centroid_doc, sort_doc, deepsort_doc])
        YOLOForm.close()


    def browseModelFile(self):
        default_path = joinFPathFull(root_dir, 'tk_deepsort')
        model_filter = "Protobuf (*.pb)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Model file", default_path, model_filter)
        if source_file:
            self.ds_model_file_lineEdit.setText(source_file)
