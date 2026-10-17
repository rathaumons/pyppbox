"""Offscreen RF-DETR dialog and preservation checks for existing detector dialogs."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt6 import QtWidgets

from pyppbox.config.configtools import internal_cfg_dir, dumpDocDictList, loadDocumentList
from pyppbox.config.myconfig import MyConfigurator
from pyppbox.gui.ui_rfdetr import Ui_RFDETR
from pyppbox.gui.ui_yolocls import Ui_YOLOCLS
from pyppbox.gui.ui_yoloult import Ui_YOLOULT
from pyppbox.gui.ui_gt import Ui_GT


class RFDETRGUIConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_new_and_legacy_dialogs_preserve_other_detectors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'detectors.yaml'
            docs = [doc for doc in loadDocumentList(Path(internal_cfg_dir) / 'detectors.yaml')
                    if doc['dt_name'] != 'RF_DETR']
            dumpDocDictList(path, docs, '')
            dialog = QtWidgets.QDialog()
            ui = Ui_RFDETR(1, directory)
            ui.setupUi(dialog)
            self.assertEqual(ui.person_class_spinBox.value(), 1)
            self.assertEqual(ui.device_comboBox.currentText(), '0')
            ui.model_size_comboBox.setCurrentText('small')
            self.assertTrue(ui.model_file_lineEdit.text().endswith('data/modules/rf_detr/rf-detr-small.pth'))
            ui.device_comboBox.setCurrentText('cuda:1')
            ui.person_class_spinBox.setValue(0)
            ui.conf_spinBox.setValue(.65)
            ui.optimize_comboBox.setCurrentText('False')
            ui.updateCFG(dialog)
            saved = loadDocumentList(path)
            self.assertEqual(saved[:3], docs)
            rf_doc = saved[-1]
            self.assertEqual(rf_doc['dt_name'], 'RF_DETR')
            self.assertEqual(rf_doc['person_class_id'], 0)
            self.assertIs(rf_doc['optimize'], False)
            cfg = MyConfigurator(directory)
            cfg.setAllDCFG()
            self.assertEqual(cfg.dcfg_rfdetr.device, 'cuda:1')
            self.assertEqual(cfg.dcfg_rfdetr.conf, .65)
            for dialog_type in (Ui_YOLOCLS, Ui_YOLOULT, Ui_GT):
                with self.subTest(dialog=dialog_type.__name__):
                    dialog = QtWidgets.QDialog()
                    other = dialog_type(1, directory)
                    other.setupUi(dialog)
                    other.updateCFG(dialog)
                    self.assertEqual(next(doc for doc in loadDocumentList(path) if doc['dt_name'] == 'RF_DETR'), rf_doc)

    def test_invalid_device_does_not_save_or_close(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'detectors.yaml'
            docs = loadDocumentList(Path(internal_cfg_dir) / 'detectors.yaml')
            dumpDocDictList(path, docs, '')
            before = path.read_bytes()
            dialog = QtWidgets.QDialog()
            ui = Ui_RFDETR(1, directory)
            ui.setupUi(dialog)
            ui.device_comboBox.setCurrentText('cuda:-1')
            with patch.object(QtWidgets.QMessageBox, 'warning') as warning, patch.object(dialog, 'close') as close:
                ui.updateCFG(dialog)
                warning.assert_called_once()
                close.assert_not_called()
            self.assertEqual(path.read_bytes(), before)

    def test_device_zero_round_trip_and_custom_or_empty_checkpoint_on_size_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'detectors.yaml'
            dumpDocDictList(path, loadDocumentList(Path(internal_cfg_dir) / 'detectors.yaml'), '')
            dialog = QtWidgets.QDialog()
            ui = Ui_RFDETR(1, directory)
            ui.setupUi(dialog)
            ui.updateCFG(dialog)
            saved = next(doc for doc in loadDocumentList(path) if doc['dt_name'] == 'RF_DETR')
            self.assertEqual(saved['device'], 0)
            self.assertIsInstance(saved['device'], int)
            ui.model_file_lineEdit.setText('my-models/custom-nano.pth')
            ui.model_size_comboBox.setCurrentText('medium')
            self.assertEqual(ui.model_file_lineEdit.text(), 'my-models/custom-nano.pth')
            ui.model_file_lineEdit.clear()
            ui.model_size_comboBox.setCurrentText('large')
            self.assertEqual(ui.model_file_lineEdit.text(), '')


if __name__ == '__main__':
    unittest.main()
