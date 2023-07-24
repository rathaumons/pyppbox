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
        self.ds_max_overlap_lineEdit = QtWidgets.QLineEdit(deepsort_ui)
        self.ds_max_overlap_lineEdit.setGeometry(QtCore.QRect(110, 40, 241, 21))
        self.ds_max_overlap_lineEdit.setObjectName("ds_max_overlap_lineEdit")
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
        self.ds_model_file_lineEdit.setReadOnly(True)
        self.ds_model_file_lineEdit.setObjectName("ds_model_file_lineEdit")
        self.ds_nn_budget_lineEdit = QtWidgets.QLineEdit(deepsort_ui)
        self.ds_nn_budget_lineEdit.setGeometry(QtCore.QRect(110, 10, 241, 21))
        self.ds_nn_budget_lineEdit.setObjectName("ds_nn_budget_lineEdit")
        self.ds_model_file_label = QtWidgets.QLabel(deepsort_ui)
        self.ds_model_file_label.setGeometry(QtCore.QRect(10, 100, 91, 16))
        self.ds_model_file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                              QtCore.Qt.AlignmentFlag.AlignTrailing|
                                              QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ds_model_file_label.setObjectName("ds_model_file_label")
        self.ds_cosine_distance_lineEdit = QtWidgets.QLineEdit(deepsort_ui)
        self.ds_cosine_distance_lineEdit.setGeometry(QtCore.QRect(110, 70, 241, 21))
        self.ds_cosine_distance_lineEdit.setObjectName("ds_cosine_distance_lineEdit")
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
        self.ds_cosine_distance_lineEdit.setText(str(self.mycfg.tcfg_deepsort.max_cosine_distance))
        self.ds_max_overlap_lineEdit.setText(str(self.mycfg.tcfg_deepsort.nms_max_overlap))
        self.ds_nn_budget_lineEdit.setText(str(self.mycfg.tcfg_deepsort.nn_budget))
        self.ds_model_file_lineEdit.setText(getAbsPathFDS(self.mycfg.tcfg_deepsort.model_file))

    def updateCFG(self, YOLOForm):
        deepsort_doc = {
            "tk_name": unified_strings.getUnifiedFormat("DeepSORT"),
            "nn_budget": getInt(self.ds_nn_budget_lineEdit.text(), default_val=100),
            "nms_max_overlap": getFloat(self.ds_max_overlap_lineEdit.text(), default_val=0.5),
            "max_cosine_distance": getFloat(self.ds_cosine_distance_lineEdit.text(), default_val=0.1),
            "model_file": normalizePathFDS(root_dir, self.ds_model_file_lineEdit.text())
        }
        centroid_doc = self.mycfg.tcfg_centroid.getDocument()
        sort_doc = self.mycfg.tcfg_sort.getDocument()
        self.mycfg.dumpAllTCFG([centroid_doc, sort_doc, deepsort_doc])
        YOLOForm.close()

    def browseModelFile(self):
        default_path = getAncestorDir(self.mycfg.tcfg_deepsort.model_file)
        model_filter = "Protobuf (*.pb)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Model file", 
                                                               default_path, model_filter)
        if source_file:
            self.ds_model_file_lineEdit.setText(source_file)
