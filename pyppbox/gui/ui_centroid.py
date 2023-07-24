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
from pyppbox.utils.commontools import getInt


unified_strings = UnifiedStrings()

class Ui_Centroid(object):

    def __init__(self, cfg_mode, cfg_dir):
        self.cfg_dir = cfg_dir
        if cfg_mode == 0:
            self.mycfg = MyCFG()
        else:
            self.mycfg = MyCFG(self.cfg_dir)

    def setupUi(self, centroid_ui):
        centroid_ui.setObjectName("centroid_ui")
        centroid_ui.resize(390, 90)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, 
                                           QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(centroid_ui.sizePolicy().hasHeightForWidth())
        centroid_ui.setSizePolicy(sizePolicy)
        centroid_ui.setMinimumSize(QtCore.QSize(390, 90))
        centroid_ui.setMaximumSize(QtCore.QSize(390, 90))
        self.save_pushButton = QtWidgets.QPushButton(centroid_ui)
        self.save_pushButton.setGeometry(QtCore.QRect(150, 50, 91, 31))
        self.save_pushButton.setObjectName("save_pushButton")
        self.ct_max_distance_lineEdit = QtWidgets.QLineEdit(centroid_ui)
        self.ct_max_distance_lineEdit.setGeometry(QtCore.QRect(110, 10, 241, 21))
        self.ct_max_distance_lineEdit.setObjectName("ct_max_distance_lineEdit")
        self.ct_max_distance_label = QtWidgets.QLabel(centroid_ui)
        self.ct_max_distance_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.ct_max_distance_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|
                                                QtCore.Qt.AlignmentFlag.AlignTrailing|
                                                QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ct_max_distance_label.setObjectName("ct_max_distance_label")
        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)
        self.retranslateUi(centroid_ui)
        # custom 
        self.loadCT()
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(centroid_ui))
        QtCore.QMetaObject.connectSlotsByName(centroid_ui)

    def retranslateUi(self, centroid_ui):
        _translate = QtCore.QCoreApplication.translate
        centroid_ui.setWindowTitle(_translate("centroid_ui", "Centroid"))
        self.ct_max_distance_label.setText(_translate("centroid_ui", "max_spread"))
        self.save_pushButton.setText(_translate("centroid_ui", "Save"))

    def loadCT(self):
        self.mycfg.setAllTCFG()
        self.ct_max_distance_lineEdit.setText(str(self.mycfg.tcfg_centroid.max_spread))

    def updateCFG(self, centroid_ui):
        centroid_doc = {
            "tk_name": unified_strings.getUnifiedFormat("Centroid"),
            "max_spread": getInt(self.ct_max_distance_lineEdit.text(), default_val=50)
        }
        sort_doc = self.mycfg.tcfg_sort.getDocument()
        deepsort_doc = self.mycfg.tcfg_deepsort.getDocument()
        self.mycfg.dumpAllTCFG([centroid_doc, sort_doc, deepsort_doc])
        centroid_ui.close()
