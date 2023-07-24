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
from pyppbox.utils.commontools import getFloat, getInt


unified_strings = UnifiedStrings()

class Ui_SORT(object):

    def __init__(self, cfg_mode, cfg_dir):
        self.cfg_dir = cfg_dir
        if cfg_mode == 0:
            self.mycfg = MyCFG()
        else:
            self.mycfg = MyCFG(self.cfg_dir)

    def setupUi(self, sort_ui):
        sort_ui.setObjectName("sort_ui")
        sort_ui.resize(390, 150)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, 
                                           QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(sort_ui.sizePolicy().hasHeightForWidth())
        sort_ui.setSizePolicy(sizePolicy)
        sort_ui.setMinimumSize(QtCore.QSize(390, 150))
        sort_ui.setMaximumSize(QtCore.QSize(390, 150))
        self.save_pushButton = QtWidgets.QPushButton(sort_ui)
        self.save_pushButton.setGeometry(QtCore.QRect(150, 110, 91, 31))
        self.save_pushButton.setObjectName("save_pushButton")
        self.st_min_hits_label = QtWidgets.QLabel(sort_ui)
        self.st_min_hits_label.setGeometry(QtCore.QRect(10, 40, 91, 16))
        self.st_min_hits_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                            QtCore.Qt.AlignmentFlag.AlignTrailing|
                                            QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.st_min_hits_label.setObjectName("st_min_hits_label")
        self.st_iou_threshold_lineEdit = QtWidgets.QLineEdit(sort_ui)
        self.st_iou_threshold_lineEdit.setGeometry(QtCore.QRect(110, 70, 241, 21))
        self.st_iou_threshold_lineEdit.setObjectName("st_iou_threshold_lineEdit")
        self.st_min_hits_lineEdit = QtWidgets.QLineEdit(sort_ui)
        self.st_min_hits_lineEdit.setGeometry(QtCore.QRect(110, 40, 241, 21))
        self.st_min_hits_lineEdit.setObjectName("st_min_hits_lineEdit")
        self.st_max_age_label = QtWidgets.QLabel(sort_ui)
        self.st_max_age_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.st_max_age_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                           QtCore.Qt.AlignmentFlag.AlignTrailing|
                                           QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.st_max_age_label.setObjectName("st_max_age_label")
        self.st_iou_threshold_label = QtWidgets.QLabel(sort_ui)
        self.st_iou_threshold_label.setGeometry(QtCore.QRect(10, 70, 91, 16))
        self.st_iou_threshold_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                                 QtCore.Qt.AlignmentFlag.AlignTrailing|
                                                 QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.st_iou_threshold_label.setObjectName("st_iou_threshold_label")
        self.st_max_age_lineEdit = QtWidgets.QLineEdit(sort_ui)
        self.st_max_age_lineEdit.setGeometry(QtCore.QRect(110, 10, 241, 21))
        self.st_max_age_lineEdit.setObjectName("st_max_age_lineEdit")
        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)
        self.retranslateUi(sort_ui)
        # custom
        self.loadST()
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(sort_ui))
        QtCore.QMetaObject.connectSlotsByName(sort_ui)

    def retranslateUi(self, sort_ui):
        _translate = QtCore.QCoreApplication.translate
        sort_ui.setWindowTitle(_translate("sort_ui", "SORT"))
        self.save_pushButton.setText(_translate("sort_ui", "Save"))
        self.st_min_hits_label.setText(_translate("sort_ui", "min_hits"))
        self.st_max_age_label.setText(_translate("sort_ui", "max_age"))
        self.st_iou_threshold_label.setText(_translate("sort_ui", "iou_threshold"))

    def loadST(self):
        self.mycfg.setAllTCFG()
        self.st_iou_threshold_lineEdit.setText(str(self.mycfg.tcfg_sort.iou_threshold))
        self.st_max_age_lineEdit.setText(str(self.mycfg.tcfg_sort.max_age))
        self.st_min_hits_lineEdit.setText(str(self.mycfg.tcfg_sort.min_hits))

    def updateCFG(self, sort_ui):
        sort_doc = {
            "tk_name": unified_strings.getUnifiedFormat("SORT"),
            "max_age": getInt(self.st_max_age_lineEdit.text(), default_val=1),
            "min_hits": getInt(self.st_min_hits_lineEdit.text(), default_val=3),
            "iou_threshold": getFloat(self.st_iou_threshold_lineEdit.text(), default_val=0.3)
        }
        centroid_doc = self.mycfg.tcfg_centroid.getDocument()
        deepsort_doc = self.mycfg.tcfg_deepsort.getDocument()
        self.mycfg.dumpAllTCFG([centroid_doc, sort_doc, deepsort_doc])
        sort_ui.close()
