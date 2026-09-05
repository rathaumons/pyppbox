"""Offscreen GUI save/load and real Ultralytics argument validation."""
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt6 import QtWidgets
from ultralytics.cfg import get_cfg

from pyppbox.config.configtools import internal_cfg_dir, loadDocumentList, dumpDocDictList
from pyppbox.config.myconfig import MyConfigurator
from pyppbox.gui.ui_yoloult import Ui_YOLOULT


class RealGUIConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_legacy_boolean_gui_round_trip_and_backend_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            docs = loadDocumentList(Path(internal_cfg_dir) / 'detectors.yaml')
            path = Path(directory) / 'detectors.yaml'
            for legacy, index, expected in (('True', 0, True), ('False', 1, False)):
                with self.subTest(legacy=legacy):
                    for doc in docs:
                        if doc['dt_name'] == 'YOLO_Ultralytics':
                            doc['show_boxes'] = legacy
                    dumpDocDictList(path, docs, '')
                    dialog = QtWidgets.QDialog()
                    ui = Ui_YOLOULT(cfg_mode=1, cfg_dir=directory)
                    ui.setupUi(dialog)
                    self.assertEqual(ui.boxes_comboBox.currentIndex(), index)
                    self.assertIs(get_cfg(overrides={'show_boxes': ui.mycfg.dcfg_yolout.show_boxes}).show_boxes, expected)
                    # Exercise both choices using real Qt widgets and the Save handler.
                    for selected_index, selected_value in ((0, True), (1, False)):
                        ui.boxes_comboBox.setCurrentIndex(selected_index)
                        ui.updateCFG(dialog)
                        saved = next(doc for doc in loadDocumentList(path) if doc['dt_name'] == 'YOLO_Ultralytics')
                        self.assertIs(saved['show_boxes'], selected_value)
                        reloaded = MyConfigurator(directory)
                        reloaded.setAllDCFG()
                        self.assertIs(get_cfg(overrides={'show_boxes': reloaded.dcfg_yolout.show_boxes}).show_boxes,
                                      selected_value)
                    dialog.close()


if __name__ == '__main__':
    unittest.main()
