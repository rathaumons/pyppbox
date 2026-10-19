# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


from PyQt6 import QtCore, QtWidgets
from pyppbox.gui.configwidgets import config_form
from pyppbox.gui.numericwidgets import IntSpinBox
from pyppbox.config.unifiedstrings import UnifiedStrings
from pyppbox.config.myconfig import MyConfigurator as MyCFG


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
        self.save_pushButton = QtWidgets.QPushButton(centroid_ui)
        self.save_pushButton.setObjectName("save_pushButton")
        self.ct_max_distance_spinBox = IntSpinBox(centroid_ui)
        self.ct_max_distance_spinBox.setObjectName("ct_max_distance_spinBox")
        self.ct_max_distance_label = QtWidgets.QLabel(centroid_ui)
        self.ct_max_distance_label.setObjectName("ct_max_distance_label")
        self.retranslateUi(centroid_ui)
        # custom 
        self.loadCT()
        config_form(centroid_ui, [
            (self.ct_max_distance_label, self.ct_max_distance_spinBox),
        ], self.save_pushButton)
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(centroid_ui))
        QtCore.QMetaObject.connectSlotsByName(centroid_ui)

    def retranslateUi(self, centroid_ui):
        _translate = QtCore.QCoreApplication.translate
        centroid_ui.setWindowTitle(_translate("centroid_ui", "Centroid"))
        self.ct_max_distance_label.setText(_translate("centroid_ui", "max_spread"))
        self.save_pushButton.setText(_translate("centroid_ui", "Save"))

    def loadCT(self):
        self.mycfg.setAllTCFG()
        self.ct_max_distance_spinBox.setValue(self.mycfg.tcfg_centroid.max_spread)

    def updateCFG(self, centroid_ui):
        centroid_doc = {
            "tk_name": unified_strings.getUnifiedFormat("Centroid"),
            "max_spread": self.ct_max_distance_spinBox.value()
        }
        try:
            self.mycfg.dumpTrackerCFG(centroid_doc)
        except (ValueError, OSError) as error:
            QtWidgets.QMessageBox.warning(centroid_ui, 'Cannot save settings', str(error))
            return
        centroid_ui.close()
