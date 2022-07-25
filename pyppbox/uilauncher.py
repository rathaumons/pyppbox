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
import sys
import yaml
import webbrowser
import subprocess as sp
from yaml.loader import SafeLoader

from PyQt6 import QtCore, QtGui, QtWidgets
from pyppbox.ui_yolo import Ui_YOLOForm
from pyppbox.ui_openpose import Ui_OpenPoseForm
from pyppbox.ui_gt import Ui_GTForm
from pyppbox.ui_centroid import Ui_CentroidForm
from pyppbox.ui_sort import Ui_SORTForm
from pyppbox.ui_deepsort import Ui_DeepSORTForm
from pyppbox.ui_facenet import Ui_FacenetForm
from pyppbox.ui_deepreid import Ui_DeepReIDForm
from pyppbox.config import MyConfigurator, MyCFGIO
from pyppbox.utils.mytools import getAbsPathFDS, normalizePathFDS, joinFPathFull, getBool

root_dir = os.path.dirname(__file__)
cfg_dir = joinFPathFull(root_dir, 'cfg')
main_yaml = os.path.join(cfg_dir, "main.yaml")




class Ui_PPTSLauncher(object):

    def setupUi(self, PPTSLauncher):
        PPTSLauncher.setObjectName("PPTSLauncher")
        PPTSLauncher.resize(490, 550)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(PPTSLauncher.sizePolicy().hasHeightForWidth())
        PPTSLauncher.setSizePolicy(sizePolicy)
        PPTSLauncher.setMinimumSize(QtCore.QSize(490, 550))
        PPTSLauncher.setMaximumSize(QtCore.QSize(490, 550))
        self.centralwidget = QtWidgets.QWidget(PPTSLauncher)
        self.centralwidget.setObjectName("centralwidget")
        self.launch_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.launch_pushButton.setGeometry(QtCore.QRect(110, 500, 271, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.launch_pushButton.setFont(font)
        self.launch_pushButton.setDefault(True)
        self.launch_pushButton.setObjectName("launch_pushButton")
        self.detector_comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.detector_comboBox.setGeometry(QtCore.QRect(110, 240, 271, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.detector_comboBox.setFont(font)
        self.detector_comboBox.setObjectName("detector_comboBox")
        self.detector_comboBox.addItem("")
        self.detector_comboBox.addItem("")
        self.detector_comboBox.addItem("")
        self.detector_comboBox.addItem("")
        self.detector_label = QtWidgets.QLabel(self.centralwidget)
        self.detector_label.setGeometry(QtCore.QRect(10, 240, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.detector_label.setFont(font)
        self.detector_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.detector_label.setObjectName("detector_label")
        self.browse_input_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.browse_input_pushButton.setGeometry(QtCore.QRect(390, 390, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.browse_input_pushButton.setFont(font)
        self.browse_input_pushButton.setObjectName("browse_input_pushButton")
        self.input_video_file_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.input_video_file_lineEdit.setGeometry(QtCore.QRect(110, 390, 271, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.input_video_file_lineEdit.setFont(font)
        self.input_video_file_lineEdit.setText("")
        self.input_video_file_lineEdit.setReadOnly(True)
        self.input_video_file_lineEdit.setObjectName("input_video_file_lineEdit")
        self.tracker_label = QtWidgets.QLabel(self.centralwidget)
        self.tracker_label.setGeometry(QtCore.QRect(10, 280, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.tracker_label.setFont(font)
        self.tracker_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.tracker_label.setObjectName("tracker_label")
        self.tracker_comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.tracker_comboBox.setGeometry(QtCore.QRect(110, 280, 271, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.tracker_comboBox.setFont(font)
        self.tracker_comboBox.setObjectName("tracker_comboBox")
        self.tracker_comboBox.addItem("")
        self.tracker_comboBox.addItem("")
        self.tracker_comboBox.addItem("")
        self.tracker_comboBox.addItem("")
        self.reider_label = QtWidgets.QLabel(self.centralwidget)
        self.reider_label.setGeometry(QtCore.QRect(10, 320, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.reider_label.setFont(font)
        self.reider_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.reider_label.setObjectName("reider_label")
        self.reider_comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.reider_comboBox.setGeometry(QtCore.QRect(110, 320, 271, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.reider_comboBox.setFont(font)
        self.reider_comboBox.setObjectName("reider_comboBox")
        self.reider_comboBox.addItem("")
        self.reider_comboBox.addItem("")
        self.reider_comboBox.addItem("")
        self.input_bl_line = QtWidgets.QFrame(self.centralwidget)
        self.input_bl_line.setGeometry(QtCore.QRect(10, 470, 471, 21))
        self.input_bl_line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.input_bl_line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.input_bl_line.setObjectName("input_bl_line")
        self.input_video_file_label = QtWidgets.QLabel(self.centralwidget)
        self.input_video_file_label.setGeometry(QtCore.QRect(10, 390, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        # font.setBold(True)
        self.input_video_file_label.setFont(font)
        self.input_video_file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.input_video_file_label.setObjectName("input_video_file_label")
        self.input_force_hd_label = QtWidgets.QLabel(self.centralwidget)
        self.input_force_hd_label.setGeometry(QtCore.QRect(10, 430, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.input_force_hd_label.setFont(font)
        self.input_force_hd_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.input_force_hd_label.setObjectName("input_force_hd_label")
        self.input_force_hd_comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.input_force_hd_comboBox.setGeometry(QtCore.QRect(110, 430, 271, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.input_force_hd_comboBox.setFont(font)
        self.input_force_hd_comboBox.setObjectName("input_force_hd_comboBox")
        self.input_force_hd_comboBox.addItem("")
        self.input_force_hd_comboBox.addItem("")
        self.input_ab_line = QtWidgets.QFrame(self.centralwidget)
        self.input_ab_line.setGeometry(QtCore.QRect(10, 360, 471, 21))
        self.input_ab_line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.input_ab_line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.input_ab_line.setObjectName("input_ab_line")
        self.config_dt_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.config_dt_pushButton.setGeometry(QtCore.QRect(390, 240, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.config_dt_pushButton.setFont(font)
        self.config_dt_pushButton.setObjectName("config_dt_pushButton")
        self.config_tk_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.config_tk_pushButton.setGeometry(QtCore.QRect(390, 280, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.config_tk_pushButton.setFont(font)
        self.config_tk_pushButton.setObjectName("config_tk_pushButton")
        self.config_ri_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.config_ri_pushButton.setGeometry(QtCore.QRect(390, 320, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.config_ri_pushButton.setFont(font)
        self.config_ri_pushButton.setObjectName("config_ri_pushButton")
        self.treid_figure_label = QtWidgets.QLabel(self.centralwidget)
        self.treid_figure_label.setGeometry(QtCore.QRect(10, 10, 471, 171))
        self.treid_figure_label.setText("")
        self.treid_figure_label.setPixmap(QtGui.QPixmap("gui/TReID.png"))
        self.treid_figure_label.setScaledContents(True)
        self.treid_figure_label.setObjectName("treid_figure_label")
        self.treid_info_label = QtWidgets.QLabel(self.centralwidget)
        self.treid_info_label.setGeometry(QtCore.QRect(10, 190, 471, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setStrikeOut(False)
        self.treid_info_label.setFont(font)
        self.treid_info_label.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.treid_info_label.setStyleSheet("color: rgb(0, 0, 255)")
        self.treid_info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.treid_info_label.setObjectName("treid_info_label")
        self.first_line = QtWidgets.QFrame(self.centralwidget)
        self.first_line.setGeometry(QtCore.QRect(10, 210, 471, 21))
        self.first_line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.first_line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.first_line.setObjectName("first_line")
        PPTSLauncher.setCentralWidget(self.centralwidget)

        self.retranslateUi(PPTSLauncher)

        # custom defs:

        self.treid_info_label.mousePressEvent = self.addEffectOnTReIDLabelColor 
        self.treid_info_label.mouseReleaseEvent = self.takeMeToGitHub     

        self.detector_comboBox.setCurrentIndex(-1)
        self.tracker_comboBox.setCurrentIndex(-1)
        self.reider_comboBox.setCurrentIndex(-1)

        self.detector_comboBox.currentIndexChanged.connect(self.detectorFilter)
        self.tracker_comboBox.currentIndexChanged.connect(self.trackerFilter)
        self.reider_comboBox.currentIndexChanged.connect(self.reiderFilter)

        self.config_dt_pushButton.clicked.connect(self.loadDTQDialog)
        self.config_tk_pushButton.clicked.connect(self.loadTKQDialog)
        self.config_ri_pushButton.clicked.connect(self.loadRIQDialog)
        self.browse_input_pushButton.clicked.connect(self.browseInputFile)

        self.launch_pushButton.clicked.connect(self.launchNow)
        
        self.loadMCFG()
        self.setMainCFG()

        QtCore.QMetaObject.connectSlotsByName(PPTSLauncher)


    def retranslateUi(self, PPTSLauncher):
        _translate = QtCore.QCoreApplication.translate
        PPTSLauncher.setWindowTitle(_translate("PPTSLauncher", "PPTS Launcher"))
        self.launch_pushButton.setText(_translate("PPTSLauncher", "LAUNCH"))
        self.detector_comboBox.setItemText(0, _translate("PPTSLauncher", "None"))
        self.detector_comboBox.setItemText(1, _translate("PPTSLauncher", "YOLO"))
        self.detector_comboBox.setItemText(2, _translate("PPTSLauncher", "OpenPose"))
        self.detector_comboBox.setItemText(3, _translate("PPTSLauncher", "GT"))
        self.detector_label.setText(_translate("PPTSLauncher", "Detector"))
        self.browse_input_pushButton.setText(_translate("PPTSLauncher", "Browse"))
        self.tracker_label.setText(_translate("PPTSLauncher", "Tracker"))
        self.tracker_comboBox.setItemText(0, _translate("PPTSLauncher", "None"))
        self.tracker_comboBox.setItemText(1, _translate("PPTSLauncher", "Centroid"))
        self.tracker_comboBox.setItemText(2, _translate("PPTSLauncher", "SORT"))
        self.tracker_comboBox.setItemText(3, _translate("PPTSLauncher", "DeepSORT"))
        self.reider_label.setText(_translate("PPTSLauncher", "Re-IDer"))
        self.reider_comboBox.setItemText(0, _translate("PPTSLauncher", "None"))
        self.reider_comboBox.setItemText(1, _translate("PPTSLauncher", "Facenet"))
        self.reider_comboBox.setItemText(2, _translate("PPTSLauncher", "DeepReID"))
        self.input_video_file_label.setText(_translate("PPTSLauncher", "Input file"))
        self.input_force_hd_label.setText(_translate("PPTSLauncher", "Force HD"))
        self.input_force_hd_comboBox.setItemText(0, _translate("PPTSLauncher", "True"))
        self.input_force_hd_comboBox.setItemText(1, _translate("PPTSLauncher", "False"))
        self.config_dt_pushButton.setText(_translate("PPTSLauncher", "Config..."))
        self.config_tk_pushButton.setText(_translate("PPTSLauncher", "Config..."))
        self.config_ri_pushButton.setText(_translate("PPTSLauncher", "Config..."))
        self.treid_info_label.setText(_translate("PPTSLauncher", "<< Click here for our papers & GitHub repo >>"))


    def loadMCFG(self):
        self.mycfg = MyConfigurator()
        self.mycfg.loadMCFG()


    def detectorFilter(self, val):
        if val == 0:
            self.config_dt_pushButton.setDisabled(True)
        else:
            self.config_dt_pushButton.setDisabled(False)


    def trackerFilter(self, val):
        if val == 0:
            self.config_tk_pushButton.setDisabled(True)
        else:
            self.config_tk_pushButton.setDisabled(False)


    def reiderFilter(self, val):
        if val == 0:
            self.config_ri_pushButton.setDisabled(True)
        else:
            self.config_ri_pushButton.setDisabled(False)


    def setMainCFG(self):

        if self.mycfg.mcfg.detector.lower() == "none":
            self.detector_comboBox.setCurrentIndex(0)
        elif self.mycfg.mcfg.detector.lower() == "yolo":
            self.detector_comboBox.setCurrentIndex(1)
        elif self.mycfg.mcfg.detector.lower() == "openpose":
            self.detector_comboBox.setCurrentIndex(2)
        elif self.mycfg.mcfg.detector.lower() == "gt":
            self.detector_comboBox.setCurrentIndex(3)
        
        if self.mycfg.mcfg.tracker.lower() == "none":
            self.tracker_comboBox.setCurrentIndex(0)
        elif self.mycfg.mcfg.tracker.lower() == "centroid":
            self.tracker_comboBox.setCurrentIndex(1)
        elif self.mycfg.mcfg.tracker.lower() == "sort":
            self.tracker_comboBox.setCurrentIndex(2)
        elif self.mycfg.mcfg.tracker.lower() == "deepsort":
            self.tracker_comboBox.setCurrentIndex(3)

        if self.mycfg.mcfg.reider.lower() == "none":
            self.reider_comboBox.setCurrentIndex(0)
        elif self.mycfg.mcfg.reider.lower() == "facenet":
            self.reider_comboBox.setCurrentIndex(1)
        elif self.mycfg.mcfg.reider.lower() == "deepreid":
            self.reider_comboBox.setCurrentIndex(2)

        full_path = getAbsPathFDS(self.mycfg.mcfg.input_video)
        self.input_video_file_lineEdit.setText(full_path)

        if self.mycfg.mcfg.force_hd is True:
            self.input_force_hd_comboBox.setCurrentIndex(0)
        else:
            self.input_force_hd_comboBox.setCurrentIndex(1)


    def browseInputFile(self):
        default_path = joinFPathFull(root_dir, "tmp/demo")
        video_filter = "Video (*.avi *.mkv *.mov *.mp4)"
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Input video file", default_path, video_filter)
        if source_file:
            self.input_video_file_lineEdit.setText(source_file)


    def launchNow(self):
        self.updateMCFG(self.detector_comboBox.currentText(), self.tracker_comboBox.currentText(), self.reider_comboBox.currentText(), self.input_video_file_lineEdit.text(), self.input_force_hd_comboBox.currentText())
        launcher.hide()
        programName = "uidemo.cmd"
        p = sp.Popen([programName])
        stdout, stderr = p.communicate()
        self.loadMCFG()
        launcher.show()


    def updateMCFG(self, detector, tracker, reider, input, force_hd):

        print("DT = " + str(detector))
        print("TK = " + str(tracker))
        print("RI = " + str(reider))

        if detector.lower() == "openpose":
            detector = "OpenPose"
        elif detector.lower() == "yolo":
            detector = "YOLO"
        elif detector.lower() == "gt":
            detector = "GT"
        else:
            detector = "None"

        if tracker.lower() == "centroid":
            tracker = "Centroid"
        elif tracker.lower() == "sort":
            tracker = "SORT"
        elif tracker.lower() == "deepsort":
            tracker = "DeepSORT"
        else:
            tracker = "None"

        if reider.lower() == "facenet":
            reider = "Facenet"
        elif reider.lower() == "deepreid":
            reider = "DeepReID"
        else:
            reider = "None"

        main_data = {'detector': detector, 
                     'tracker': tracker, 
                     'reider': reider, 
                     'input_video': normalizePathFDS(root_dir, input), 
                     'force_hd': getBool(force_hd)}
                    
        cfgIO = MyCFGIO()
        cfgIO.dumpMainWithHeader(main_data)


    def loadDTQDialog(self):
        tmp_QDialog = QtWidgets.QDialog(None, QtCore.Qt.WindowType.WindowCloseButtonHint)
        tmp_QDialog.setWindowIcon(QtGui.QIcon(joinFPathFull(root_dir, "gui/settings.ico")))
        if self.detector_comboBox.currentText().lower() == "yolo":
            ui = Ui_YOLOForm()
            ui.setupUi(tmp_QDialog)
            tmp_QDialog.exec()
        elif self.detector_comboBox.currentText().lower() == "openpose":
            ui = Ui_OpenPoseForm()
            ui.setupUi(tmp_QDialog)
            tmp_QDialog.exec()
        elif self.detector_comboBox.currentText().lower() == "gt":
            ui = Ui_GTForm()
            ui.setupUi(tmp_QDialog)
            tmp_QDialog.exec()
        elif self.detector_comboBox.currentText().lower() == "none":
            pass


    def loadTKQDialog(self):
        tmp_QDialog = QtWidgets.QDialog(None, QtCore.Qt.WindowType.WindowCloseButtonHint)
        tmp_QDialog.setWindowIcon(QtGui.QIcon(joinFPathFull(root_dir, "gui/settings.ico")))
        if self.tracker_comboBox.currentText().lower() == "centroid":
            ui = Ui_CentroidForm()
            ui.setupUi(tmp_QDialog)
            tmp_QDialog.exec()
        elif self.tracker_comboBox.currentText().lower() == "sort":
            ui = Ui_SORTForm()
            ui.setupUi(tmp_QDialog)
            tmp_QDialog.exec()
        elif self.tracker_comboBox.currentText().lower() == "deepsort":
            ui = Ui_DeepSORTForm()
            ui.setupUi(tmp_QDialog)
            tmp_QDialog.exec()
        elif self.tracker_comboBox.currentText().lower() == "none":
            pass


    def loadRIQDialog(self):
        tmp_QDialog = QtWidgets.QDialog(None, QtCore.Qt.WindowType.WindowCloseButtonHint)
        tmp_QDialog.setWindowIcon(QtGui.QIcon(joinFPathFull(root_dir, "gui/settings.ico")))
        if self.reider_comboBox.currentText().lower() == "facenet":
            ui = Ui_FacenetForm()
            ui.setupUi(tmp_QDialog)
            tmp_QDialog.exec()
        elif self.reider_comboBox.currentText().lower() == "deepreid":
            ui = Ui_DeepReIDForm()
            ui.setupUi(tmp_QDialog)
            tmp_QDialog.exec()
        elif self.reider_comboBox.currentText().lower() == "none":
            pass


    def addEffectOnTReIDLabelColor(self, _):
        self.treid_info_label.setStyleSheet("color: rgb(170, 0, 255)")


    def takeMeToGitHub(self, _):
        self.treid_info_label.setStyleSheet("color: rgb(0, 0, 255)")
        webbrowser.open('https://github.com/rathaumons/pyppbox.git')



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    launcher = QtWidgets.QMainWindow()
    launcher.setWindowIcon(QtGui.QIcon(joinFPathFull(root_dir, "gui/settings.ico")))
    ui = Ui_PPTSLauncher()
    ui.setupUi(launcher)
    launcher.show()
    sys.exit(app.exec())

