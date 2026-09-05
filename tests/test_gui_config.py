"""Exercise the real GUI save/load handlers with lightweight widget stand-ins."""
import importlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pyppbox.config.configtools import internal_cfg_dir, loadDocumentList, dumpDocDictList
from pyppbox.config.myconfig import MyConfigurator


class GUIConfigTests(unittest.TestCase):
    def test_yolo_gui_saves_boolean_and_restores_selected_value(self):
        with patch.dict(sys.modules, {'PyQt6': SimpleNamespace(QtCore=Mock(), QtGui=Mock(), QtWidgets=Mock())}):
            module = importlib.import_module('pyppbox.gui.ui_yoloult')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'detectors.yaml'
            docs = loadDocumentList(Path(internal_cfg_dir) / 'detectors.yaml')
            for text, expected, index in (('False', False, 1), ('True', True, 0)):
                with self.subTest(selection=text):
                    # Begin with a legacy quoted value, as reported from the GUI.
                    for doc in docs:
                        if doc['dt_name'] == 'YOLO_Ultralytics':
                            doc['show_boxes'] = text
                    dumpDocDictList(path, docs, '')
                    configurator = MyConfigurator(directory)
                    configurator.setAllDCFG()
                    ui = module.Ui_YOLOULT.__new__(module.Ui_YOLOULT)
                    ui.mycfg = configurator
                    values = {'conf': '0.4', 'iou': '0.7', 'imgsz': '640', 'device': 'cpu',
                              'max_det': '100', 'model_file': configurator.dcfg_yolout.model_file,
                              'repspoint_calib': '0.25'}
                    for name, value in values.items():
                        setattr(ui, name + '_lineEdit', Mock(text=Mock(return_value=value)))
                    ui.boxes_comboBox = Mock(currentText=Mock(return_value=text))
                    ui.loadComboBoxes()
                    ui.boxes_comboBox.setCurrentIndex.assert_called_once_with(index)
                    dialog = Mock()
                    ui.updateCFG(dialog)
                    dialog.close.assert_called_once_with()
                    saved = next(doc for doc in loadDocumentList(path) if doc['dt_name'] == 'YOLO_Ultralytics')
                    self.assertIs(saved['show_boxes'], expected)
                    configurator.setAllDCFG()
                    self.assertIs(configurator.dcfg_yolout.show_boxes, expected)


if __name__ == '__main__':
    unittest.main()
