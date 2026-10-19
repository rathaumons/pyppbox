# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


from PyQt6 import QtCore, QtWidgets
from pyppbox.gui.configwidgets import config_form
from pyppbox.gui.numericwidgets import FloatSpinBox, IntSpinBox
from pyppbox.config.unifiedstrings import UnifiedStrings
from pyppbox.config.myconfig import MyConfigurator as MyCFG


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
        self.save_pushButton = QtWidgets.QPushButton(sort_ui)
        self.save_pushButton.setObjectName("save_pushButton")
        self.st_min_hits_label = QtWidgets.QLabel(sort_ui)
        self.st_min_hits_label.setObjectName("st_min_hits_label")
        self.st_iou_threshold_spinBox = FloatSpinBox(sort_ui)
        self.st_iou_threshold_spinBox.setObjectName("st_iou_threshold_spinBox")
        self.st_min_hits_spinBox = IntSpinBox(sort_ui)
        self.st_min_hits_spinBox.setObjectName("st_min_hits_spinBox")
        self.st_max_age_label = QtWidgets.QLabel(sort_ui)
        self.st_max_age_label.setObjectName("st_max_age_label")
        self.st_iou_threshold_label = QtWidgets.QLabel(sort_ui)
        self.st_iou_threshold_label.setObjectName("st_iou_threshold_label")
        self.st_max_age_spinBox = IntSpinBox(sort_ui)
        self.st_max_age_spinBox.setObjectName("st_max_age_spinBox")
        self.retranslateUi(sort_ui)
        # custom
        self.loadST()
        config_form(sort_ui, [
            (self.st_max_age_label, self.st_max_age_spinBox),
            (self.st_min_hits_label, self.st_min_hits_spinBox),
            (self.st_iou_threshold_label, self.st_iou_threshold_spinBox),
        ], self.save_pushButton)
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
        self.st_iou_threshold_spinBox.setValue(self.mycfg.tcfg_sort.iou_threshold)
        self.st_max_age_spinBox.setValue(self.mycfg.tcfg_sort.max_age)
        self.st_min_hits_spinBox.setValue(self.mycfg.tcfg_sort.min_hits)

    def updateCFG(self, sort_ui):
        sort_doc = {
            "tk_name": unified_strings.getUnifiedFormat("SORT"),
            "max_age": self.st_max_age_spinBox.value(),
            "min_hits": self.st_min_hits_spinBox.value(),
            "iou_threshold": self.st_iou_threshold_spinBox.value()
        }
        try:
            self.mycfg.dumpTrackerCFG(sort_doc)
        except (ValueError, OSError) as error:
            QtWidgets.QMessageBox.warning(sort_ui, 'Cannot save settings', str(error))
            return
        sort_ui.close()
