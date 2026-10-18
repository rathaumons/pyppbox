"""Real numeric controls must preserve existing configs and save arrow edits."""
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt6 import QtCore, QtTest, QtWidgets

from pyppbox.config.configtools import internal_cfg_dir, loadDocumentList, dumpDocDictList
from pyppbox.config.myconfig import MyConfigurator
from pyppbox.gui.ui_yolocls import Ui_YOLOCLS
from pyppbox.gui.ui_yoloult import Ui_YOLOULT
from pyppbox.gui.ui_rfdetr import Ui_RFDETR
from pyppbox.gui.ui_gt import Ui_GT
from pyppbox.gui.ui_centroid import Ui_Centroid
from pyppbox.gui.ui_sort import Ui_SORT
from pyppbox.gui.ui_deepsort import Ui_DeepSORT
from pyppbox.gui.ui_facenet import Ui_FaceNet
from pyppbox.gui.ui_torchreid import Ui_Torchreid


PANELS = (Ui_YOLOCLS, Ui_YOLOULT, Ui_RFDETR, Ui_GT, Ui_Centroid,
          Ui_SORT, Ui_DeepSORT, Ui_FaceNet, Ui_Torchreid)
CONFIGS = ('dcfg_yolocs', 'dcfg_yolout', 'dcfg_rfdetr', 'dcfg_gt',
           'tcfg_centroid', 'tcfg_sort', 'tcfg_deepsort', 'rcfg_facenet', 'rcfg_torchreid')


class NumericGUIConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def copy_configs(self, folder):
        for name in ('main.yaml', 'detectors.yaml', 'trackers.yaml', 'reiders.yaml'):
            shutil.copy2(Path(internal_cfg_dir) / name, Path(folder) / name)

    def snapshot(self, folder):
        cfg = MyConfigurator(str(folder), set_all_modules=True)
        return {name: getattr(cfg, name).getDocument() for name in CONFIGS}

    def test_all_panels_preserve_numeric_precision_zeroes_and_nullable_budget(self):
        edits = {
            'YOLO_Classic': dict(conf=.12345678901234566, repspoint_calibration=-.125),
            'YOLO_Ultralytics': dict(conf=.12345678901234566, max_det=100001),
            'RF_DETR': dict(conf=.12345678901234566, max_det=100001),
            'Centroid': dict(max_spread=0),
            'SORT': dict(max_age=0, min_hits=0, iou_threshold=.3333333333333333),
            'DeepSORT': dict(nn_budget=None, max_cosine_distance=1.25),
            'FaceNet': dict(gpu_mem=.585, batch_size=1001, min_confidence=.75000123,
                            yl_h_calibration=[-125, 75], yl_w_calibration=[-55, 55]),
            'Torchreid': dict(min_confidence=.35000123),
        }
        for panel in PANELS:
            with self.subTest(panel=panel.__name__), tempfile.TemporaryDirectory() as folder:
                self.copy_configs(folder)
                for filename, key in (('detectors.yaml', 'dt_name'), ('trackers.yaml', 'tk_name'),
                                      ('reiders.yaml', 'ri_name')):
                    path = Path(folder) / filename
                    docs = loadDocumentList(path)
                    for doc in docs:
                        doc.update(edits.get(doc[key], {}))
                    dumpDocDictList(path, docs, '')
                before = self.snapshot(folder)
                dialog = QtWidgets.QDialog()
                ui = panel(1, folder)
                ui.setupUi(dialog)
                ui.updateCFG(dialog)
                self.assertEqual(self.snapshot(folder), before)
                dialog.close()

    def test_arrow_keys_and_typed_decimals_are_saved_as_numbers(self):
        cases = (
            (Ui_YOLOCLS, 'yl_model_imgsize_spinBox', 'dcfg_yolocs', 'model_image_size', 1056),
            (Ui_YOLOULT, 'max_det_spinBox', 'dcfg_yolout', 'max_det', 101),
            (Ui_RFDETR, 'max_det_spinBox', 'dcfg_rfdetr', 'max_det', 101),
            (Ui_Centroid, 'ct_max_distance_spinBox', 'tcfg_centroid', 'max_spread', 65),
            (Ui_SORT, 'st_max_age_spinBox', 'tcfg_sort', 'max_age', 2),
            (Ui_DeepSORT, 'ds_nn_budget_spinBox', 'tcfg_deepsort', 'nn_budget', 101),
            (Ui_FaceNet, 'fn_gpu_mem_spinBox', 'rcfg_facenet', 'gpu_mem', .595),
            (Ui_Torchreid, 'dr_min_confidence_spinBox', 'rcfg_torchreid', 'min_confidence', .4),
        )
        for panel, field, config, key, expected in cases:
            with self.subTest(panel=panel.__name__), tempfile.TemporaryDirectory() as folder:
                self.copy_configs(folder)
                dialog = QtWidgets.QDialog()
                ui = panel(1, folder)
                ui.setupUi(dialog)
                dialog.show()
                self.app.processEvents()
                spin = getattr(ui, field)
                spin.setFocus()
                QtTest.QTest.keyClick(spin, QtCore.Qt.Key.Key_Up)
                QtTest.QTest.mouseClick(ui.save_pushButton, QtCore.Qt.MouseButton.LeftButton)
                saved = self.snapshot(folder)[config][key]
                self.assertEqual(saved, expected)
                self.assertIs(type(saved), type(expected))
                dialog.close()
        with tempfile.TemporaryDirectory() as folder:
            self.copy_configs(folder)
            dialog = QtWidgets.QDialog()
            ui = Ui_FaceNet(1, folder)
            ui.setupUi(dialog)
            dialog.show()
            self.app.processEvents()
            ui.fn_gpu_mem_spinBox.lineEdit().setText('0.6325123456789')
            ui.fn_yl_h_calib_pair.first_spinBox.setFocus()
            QtTest.QTest.keyClick(ui.fn_yl_h_calib_pair.first_spinBox, QtCore.Qt.Key.Key_Up)
            QtTest.QTest.mouseClick(ui.save_pushButton, QtCore.Qt.MouseButton.LeftButton)
            saved = self.snapshot(folder)['rcfg_facenet']
            self.assertEqual(saved['gpu_mem'], .6325123456789)
            self.assertEqual(saved['yl_h_calibration'], [-124, 75])
            dialog.close()

    def test_decimal_steps_stay_readable_and_unlimited_budget_remains_editable(self):
        with tempfile.TemporaryDirectory() as folder:
            self.copy_configs(folder)
            dialog = QtWidgets.QDialog()
            ui = Ui_YOLOULT(1, folder)
            ui.setupUi(dialog)
            for _ in range(4):
                ui.conf_spinBox.stepUp()
            self.assertEqual(ui.conf_spinBox.value(), .6)
            self.assertEqual(ui.conf_spinBox.text(), '0.6')
            dialog.close()
            dialog = QtWidgets.QDialog()
            ui = Ui_DeepSORT(1, folder)
            ui.setupUi(dialog)
            ui.ds_nn_budget_spinBox.setValue(None)
            self.assertEqual(ui.ds_nn_budget_spinBox.text(), 'None')
            ui.updateCFG(dialog)
            self.assertIsNone(self.snapshot(folder)['tcfg_deepsort']['nn_budget'])
            ui.ds_nn_budget_spinBox.stepUp()
            ui.updateCFG(dialog)
            self.assertEqual(self.snapshot(folder)['tcfg_deepsort']['nn_budget'], 0)

    def test_every_panel_preserves_extensions_custom_modules_and_document_order(self):
        for panel in PANELS:
            with self.subTest(panel=panel.__name__), tempfile.TemporaryDirectory() as folder:
                self.copy_configs(folder)
                before={}
                for filename, key in (('detectors.yaml','dt_name'),('trackers.yaml','tk_name'),('reiders.yaml','ri_name')):
                    path=Path(folder)/filename; docs=loadDocumentList(path)
                    for doc in docs:
                        # These schemas intentionally reject unknown options.
                        if doc[key] not in ('RF_DETR', 'ArcFace', 'AdaFace'):
                            doc['extension']={'keep':['001',False,42]}
                    docs.insert(1,{key:'CustomModule','opaque':{'keep':'value'}})
                    dumpDocDictList(path,docs,''); before[filename]=docs
                dialog=QtWidgets.QDialog(); ui=panel(1,folder); ui.setupUi(dialog)
                effective_before=self.snapshot(folder)
                target=effective_before[CONFIGS[PANELS.index(panel)]]
                key=next(key for key in ('dt_name','tk_name','ri_name') if key in target)
                selected=target[key]
                ui.updateCFG(dialog)
                for filename, docs in before.items():
                    after=loadDocumentList(Path(folder)/filename)
                    self.assertEqual(len(after),len(docs))
                    for old,new in zip(docs,after):
                        if old.get(key)==selected:
                            self.assertEqual(new.get('extension'),old.get('extension'))
                        else:
                            self.assertEqual(new,old)
                self.assertEqual(self.snapshot(folder),effective_before)
                dialog.close()

    def test_save_failures_leave_all_dialogs_open_and_original_bytes_intact(self):
        from pyppbox.config import configtools
        for panel in PANELS:
            with self.subTest(panel=panel.__name__), tempfile.TemporaryDirectory() as folder:
                self.copy_configs(folder)
                before={p.name:p.read_bytes() for p in Path(folder).glob('*.yaml')}
                dialog=QtWidgets.QDialog(); ui=panel(1,folder); ui.setupUi(dialog)
                with patch.object(configtools.os,'replace',side_effect=OSError('injected disk error')), \
                        patch.object(QtWidgets.QMessageBox,'warning') as warning, patch.object(dialog,'close') as close:
                    ui.updateCFG(dialog)
                    warning.assert_called_once(); close.assert_not_called()
                self.assertEqual({p.name:p.read_bytes() for p in Path(folder).glob('*.yaml')},before)
                dialog.close()

    def test_indexed_devices_roundtrip_and_invalid_devices_are_not_saved(self):
        for panel, field in ((Ui_YOLOULT,'device_lineEdit'),(Ui_Torchreid,'dr_device_lineEdit')):
            with self.subTest(panel=panel.__name__), tempfile.TemporaryDirectory() as folder:
                self.copy_configs(folder)
                dialog=QtWidgets.QDialog(); ui=panel(1,folder); ui.setupUi(dialog)
                getattr(ui,field).setText('cuda:1'); ui.updateCFG(dialog)
                name='dcfg_yolout' if panel is Ui_YOLOULT else 'rcfg_torchreid'
                self.assertEqual(self.snapshot(folder)[name]['device'],'cuda:1')
                before={p.name:p.read_bytes() for p in Path(folder).glob('*.yaml')}
                getattr(ui,field).setText('garbage')
                with patch.object(QtWidgets.QMessageBox,'warning') as warning, patch.object(dialog,'close') as close:
                    ui.updateCFG(dialog)
                    warning.assert_called_once(); close.assert_not_called()
                self.assertEqual({p.name:p.read_bytes() for p in Path(folder).glob('*.yaml')},before)


if __name__ == '__main__':
    unittest.main()
