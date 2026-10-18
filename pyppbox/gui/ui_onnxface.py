# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Face recognizer settings, without importing the optional inference backend."""
from PyQt6 import QtWidgets
from pyppbox.gui.numericwidgets import FloatSpinBox, IntSpinBox, IntPairEdit
from pyppbox.config.myconfig import MyConfigurator, RCFGArcFace, RCFGAdaFace


class Ui_ONNXFace:
    def __init__(self, cfg_mode, cfg_dir, name):
        self.mycfg = MyConfigurator() if cfg_mode == 0 else MyConfigurator(cfg_dir)
        self.cls = RCFGArcFace if name == 'ArcFace' else RCFGAdaFace
        self.name = name

    def setupUi(self, dialog):
        self.mycfg.setAllRCFG()
        cfg = getattr(self.mycfg, 'rcfg_' + self.name.lower())
        dialog.setWindowTitle(self.name)
        form = QtWidgets.QFormLayout(dialog)
        self.fields = {}
        for key in self.cls.DEFAULTS:
            if key == 'ri_name':
                continue
            value = getattr(cfg, key)
            if key in ('min_confidence', 'det_thresh'):
                widget = FloatSpinBox(dialog)
                widget.setRange(0, 1)
                widget.setValue(value)
            elif key in ('input_mean', 'input_std'):
                widget = QtWidgets.QDoubleSpinBox(dialog)
                widget.setRange(0.000001 if key == 'input_std' else -1000000, 1000000)
                widget.setDecimals(6)
                widget.setValue(value)
            elif key in ('num_threads', 'det_size'):
                widget = IntSpinBox(dialog, minimum=1)
                widget.setValue(value)
                if key == 'det_size':
                    widget.setSingleStep(32)
            elif key in ('yl_h_calibration', 'yl_w_calibration'):
                widget = IntPairEdit(dialog)
                widget.setValue(value)
            elif key in ('device', 'input_color'):
                widget = QtWidgets.QComboBox(dialog)
                widget.addItems(['cpu', 'cuda:0'] if key == 'device' else ['RGB', 'BGR'])
                widget.setEditable(key == 'device')
                widget.setCurrentText(str(value))
            else:
                widget = QtWidgets.QLineEdit(str(value), dialog)
                widget.setCursorPosition(0)
            widget.setObjectName(key + '_field')
            self.fields[key] = widget
            if key in ('model_file', 'model_det', 'classifier_pkl', 'train_data'):
                row = QtWidgets.QWidget(dialog)
                layout = QtWidgets.QHBoxLayout(row)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(widget)
                button = QtWidgets.QPushButton('...', row)
                button.setFixedWidth(28)
                button.clicked.connect(lambda checked=False, field=key: self.browse(field))
                layout.addWidget(button)
                form.addRow(key, row)
            else:
                form.addRow(key, widget)
        self.fields['min_confidence'].setToolTip('Identity SVM probability threshold; requires a new classifier for this model.')
        self.fields['input_color'].setToolTip('Must match the exported model. CVLFace IR18 export: RGB; original AdaFace: BGR.')
        self.save_pushButton = QtWidgets.QPushButton('Save', dialog)
        self.save_pushButton.clicked.connect(lambda: self.updateCFG(dialog))
        form.addRow(self.save_pushButton)
        dialog.resize(620, dialog.sizeHint().height())

    def browse(self, field):
        widget = self.fields[field]
        if field == 'train_data':
            path = QtWidgets.QFileDialog.getExistingDirectory(None, 'Training images', widget.text())
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(None, field, widget.text())
        if path:
            widget.setText(path)

    def updateCFG(self, dialog):
        document = {'ri_name': self.name}
        for key, widget in self.fields.items():
            if isinstance(widget, QtWidgets.QComboBox):
                document[key] = widget.currentText()
            elif isinstance(widget, QtWidgets.QLineEdit):
                document[key] = widget.text().strip()
            else:
                document[key] = widget.value()
        try:
            cfg = self.cls(self.mycfg.relative_to_pyppbox_root)
            cfg.set(document)
            self.mycfg.dumpReiderCFG(cfg.getDocument())
        except (ValueError, OSError) as error:
            QtWidgets.QMessageBox.warning(dialog, 'Cannot save face settings', str(error))
            return
        dialog.close()
