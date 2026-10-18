# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""RF-DETR settings dialog; editing settings does not load the optional backend."""
from PyQt6 import QtCore, QtGui, QtWidgets
from pyppbox.gui.configwidgets import config_form, device_dropdown
from pyppbox.gui.numericwidgets import FloatSpinBox, IntSpinBox

from pyppbox.config.myconfig import DCFGRFDETR, MyConfigurator


class Ui_RFDETR:
    def __init__(self, cfg_mode, cfg_dir):
        self.mycfg = MyConfigurator() if cfg_mode == 0 else MyConfigurator(cfg_dir)

    def setupUi(self, dialog):
        self.mycfg.setAllDCFG()
        cfg = self.mycfg.dcfg_rfdetr
        dialog.setObjectName('rfdetr_ui')
        dialog.setWindowTitle('RF-DETR')
        dialog.setFixedSize(390, 300)
        dialog.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                             QtWidgets.QSizePolicy.Policy.Fixed)

        self.model_size_comboBox = QtWidgets.QComboBox(dialog)
        self.model_size_comboBox.addItems(['nano', 'small', 'medium', 'large'])
        self.model_size_comboBox.setCurrentText(cfg.model_size)
        self.model_size_comboBox.setToolTip('Pretrained models use their native resolution and return boxes only.')
        self.device_comboBox = device_dropdown(dialog, ['0', 'cpu', 'cuda:0', 'mps', 'auto'])
        self.device_comboBox.setCurrentText(str(cfg.device))
        self.conf_spinBox = FloatSpinBox(dialog)
        self.conf_spinBox.setRange(0, 1)
        self.conf_spinBox.setSingleStep(0.05)
        self.conf_spinBox.setValue(cfg.conf)
        self.max_det_spinBox = IntSpinBox(dialog, minimum=1)
        self.max_det_spinBox.setValue(cfg.max_det)
        self.person_class_spinBox = IntSpinBox(dialog)
        self.person_class_spinBox.setValue(cfg.person_class_id)
        self.person_class_spinBox.setToolTip('Pretrained COCO: 1. Custom checkpoints: use their people class ID.')
        self.model_file_lineEdit = QtWidgets.QLineEdit(cfg.model_file, dialog)
        self.model_file_lineEdit.setPlaceholderText('Empty: use pretrained weights')
        self.model_file_lineEdit.setToolTip('Leave empty to download/use pretrained weights, or select a checkpoint.')
        self.model_file_lineEdit.setCursorPosition(0)
        self._model_size = cfg.model_size
        self.model_size_comboBox.currentTextChanged.connect(self.updateModelSize)
        self.calibration_spinBox = FloatSpinBox(dialog)
        self.calibration_spinBox.setRange(0, 1)
        self.calibration_spinBox.setSingleStep(0.05)
        self.calibration_spinBox.setValue(cfg.repspoint_calibration)
        self.optimize_comboBox = QtWidgets.QComboBox(dialog)
        self.optimize_comboBox.addItems(['True', 'False'])
        self.optimize_comboBox.setCurrentText(str(cfg.optimize))
        self.optimize_comboBox.setToolTip('Prepare the model for inference when it is loaded.')

        fields = [
            ('model_size', self.model_size_comboBox),
            ('device', self.device_comboBox),
            ('conf', self.conf_spinBox),
            ('max_det', self.max_det_spinBox),
            ('person_class_id', self.person_class_spinBox),
            ('model_file', self.model_file_lineEdit),
            ('repspoint_calib', self.calibration_spinBox),
            ('optimize', self.optimize_comboBox),
        ]
        for index, (name, field) in enumerate(fields):
            top = 10 + 30 * index
            label = QtWidgets.QLabel(name, dialog)
            label.setObjectName(name + '_label')
            label.setGeometry(10, top, 91, 16)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight |
                               QtCore.Qt.AlignmentFlag.AlignVCenter)
            field.setObjectName(name + '_field')
            field.setGeometry(110, top, 241, 21)
            label.setBuddy(field)
            setattr(self, name + '_label', label)

        self.model_file_pushButton = QtWidgets.QPushButton('...', dialog)
        self.model_file_pushButton.setObjectName('model_file_pushButton')
        self.model_file_pushButton.setGeometry(360, 160, 21, 24)
        self.model_file_pushButton.setToolTip('Browse for a model checkpoint')
        self.model_file_pushButton.clicked.connect(self.browseModelFile)
        self.save_pushButton = QtWidgets.QPushButton('Save', dialog)
        self.save_pushButton.setObjectName('save_pushButton')
        self.save_pushButton.setGeometry(150, 260, 91, 31)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.save_pushButton.setFont(font)
        self.save_pushButton.setDefault(True)
        self.save_pushButton.setFocus()
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(dialog))
        rows = [(getattr(self, name + '_label'), field, self.model_file_pushButton)
                if name == 'model_file' else (getattr(self, name + '_label'), field)
                for name, field in fields]
        config_form(dialog, rows, self.save_pushButton)

    def updateModelSize(self, size):
        path = self.model_file_lineEdit.text().strip().replace('\\', '/')
        old_default = 'data/modules/rf_detr/' + DCFGRFDETR.MODEL_FILES[self._model_size]
        if path == old_default or path.endswith('/' + old_default):
            directory = path.rsplit('/', 1)[0]
            self.model_file_lineEdit.setText(directory + '/' + DCFGRFDETR.MODEL_FILES[size])
            self.model_file_lineEdit.setCursorPosition(0)
        self._model_size = size

    def browseModelFile(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, 'RF-DETR checkpoint', self.model_file_lineEdit.text(), 'Checkpoint (*.pth *.pt)')
        if path:
            self.model_file_lineEdit.setText(path)

    def updateCFG(self, dialog):
        document = dict(dt_name='RF_DETR', model_size=self.model_size_comboBox.currentText(),
                        device=self.device_comboBox.currentText(), conf=self.conf_spinBox.value(),
                        max_det=self.max_det_spinBox.value(), person_class_id=self.person_class_spinBox.value(),
                        model_file=self.model_file_lineEdit.text().strip(),
                        repspoint_calibration=self.calibration_spinBox.value(),
                        optimize=self.optimize_comboBox.currentText() == 'True')
        try:
            cfg = DCFGRFDETR(self.mycfg.relative_to_pyppbox_root)
            cfg.set(document)
            self.mycfg.dumpDetectorCFG(cfg.getDocument())
        except (ValueError, OSError) as error:
            QtWidgets.QMessageBox.warning(dialog, 'Cannot save RF-DETR settings', str(error))
            return
        dialog.close()
