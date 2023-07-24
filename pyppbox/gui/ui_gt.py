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
from pyppbox.utils.commontools import (normalizePathFDS, getAbsPathFDS, 
                                       getGlobalRootDir, getAncestorDir)


unified_strings = UnifiedStrings()
root_dir = getGlobalRootDir()

class Ui_GT(object):

    def __init__(self, cfg_mode, cfg_dir):
        self.cfg_dir = cfg_dir
        if cfg_mode == 0:
            self.mycfg = MyCFG()
        else:
            self.mycfg = MyCFG(self.cfg_dir)

    def setupUi(self, gi_ui):
        gi_ui.setObjectName("gi_ui")
        gi_ui.resize(390, 120)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, 
                                           QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(gi_ui.sizePolicy().hasHeightForWidth())
        gi_ui.setSizePolicy(sizePolicy)
        gi_ui.setMinimumSize(QtCore.QSize(390, 120))
        gi_ui.setMaximumSize(QtCore.QSize(390, 120))
        self.gt_file_browse_pushButton = QtWidgets.QPushButton(gi_ui)
        self.gt_file_browse_pushButton.setGeometry(QtCore.QRect(360, 10, 21, 24))
        self.gt_file_browse_pushButton.setObjectName("gt_file_browse_pushButton")
        self.gt_map_browse_pushButton = QtWidgets.QPushButton(gi_ui)
        self.gt_map_browse_pushButton.setGeometry(QtCore.QRect(360, 40, 21, 24))
        self.gt_map_browse_pushButton.setObjectName("gt_map_browse_pushButton")
        self.save_pushButton = QtWidgets.QPushButton(gi_ui)
        self.save_pushButton.setGeometry(QtCore.QRect(150, 80, 91, 31))
        self.save_pushButton.setObjectName("save_pushButton")
        self.gt_file_lineEdit = QtWidgets.QLineEdit(gi_ui)
        self.gt_file_lineEdit.setGeometry(QtCore.QRect(110, 10, 241, 21))
        self.gt_file_lineEdit.setObjectName("gt_file_lineEdit")
        self.gt_file_label = QtWidgets.QLabel(gi_ui)
        self.gt_file_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.gt_file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                        QtCore.Qt.AlignmentFlag.AlignTrailing|
                                        QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.gt_file_label.setObjectName("gt_file_label")
        self.gt_map_lineEdit = QtWidgets.QLineEdit(gi_ui)
        self.gt_map_lineEdit.setGeometry(QtCore.QRect(110, 40, 241, 21))
        self.gt_map_lineEdit.setObjectName("gt_map_lineEdit")
        self.gt_map_label = QtWidgets.QLabel(gi_ui)
        self.gt_map_label.setGeometry(QtCore.QRect(10, 40, 91, 16))
        self.gt_map_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                       QtCore.Qt.AlignmentFlag.AlignTrailing|
                                       QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.gt_map_label.setObjectName("gt_map_label")
        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)
        self.retranslateUi(gi_ui)
        # custom
        self.loadGT()  
        self.gt_file_browse_pushButton.clicked.connect(self.browseGTFile)
        self.gt_map_browse_pushButton.clicked.connect(self.browseGTMap)
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(gi_ui))
        QtCore.QMetaObject.connectSlotsByName(gi_ui)

    def retranslateUi(self, gi_ui):
        _translate = QtCore.QCoreApplication.translate
        gi_ui.setWindowTitle(_translate("gi_ui", "GT"))
        self.gt_file_label.setText(_translate("gi_ui", "gt_file"))
        self.gt_file_browse_pushButton.setText(_translate("gi_ui", "..."))
        self.gt_map_label.setText(_translate("gi_ui", "gt_map_file"))
        self.gt_map_browse_pushButton.setText(_translate("gi_ui", "..."))
        self.save_pushButton.setText(_translate("gi_ui", "Save"))

    def loadGT(self):
        self.mycfg.setAllDCFG()
        self.gt_file_lineEdit.setText(getAbsPathFDS(self.mycfg.dcfg_gt.gt_file))
        self.gt_map_lineEdit.setText(getAbsPathFDS(self.mycfg.dcfg_gt.gt_map_file))

    def browseGTFile(self):
        default_path = getAncestorDir(self.mycfg.dcfg_gt.gt_file)
        file_filter = "Text (*.txt)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Input GT file", 
                                                               default_path, file_filter)
        if source_file:
            self.gt_file_lineEdit.setText(source_file)
    
    def browseGTMap(self):
        default_path = getAncestorDir(self.mycfg.dcfg_gt.gt_map_file)
        file_filter = "Text (*.txt)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Input GT Map file", 
                                                               default_path, file_filter)
        if source_file:
            self.gt_map_lineEdit.setText(source_file)

    def updateCFG(self, gi_ui):
        gt_doc = {"dt_name": unified_strings.getUnifiedFormat("GT"),
                  "gt_file": normalizePathFDS(root_dir, self.gt_file_lineEdit.text()),
                  "gt_map_file": normalizePathFDS(root_dir, self.gt_map_lineEdit.text())}
        yolo_doc = self.mycfg.dcfg_yolocs.getDocument()
        yolo_utlt_doc = self.mycfg.dcfg_yolout.getDocument()
        self.mycfg.dumpAllDCFG([yolo_doc, yolo_utlt_doc, gt_doc])
        gi_ui.close()
