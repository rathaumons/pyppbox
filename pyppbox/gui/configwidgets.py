# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Shared controls for the module configuration dialogs."""
from PyQt6 import QtCore, QtGui, QtWidgets


def device_dropdown(parent, choices):
    """Create an editable device selector with backend-specific presets."""
    dropdown = QtWidgets.QComboBox(parent)
    dropdown.setEditable(True)
    dropdown.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
    dropdown.addItems(choices)
    dropdown.setToolTip('Choose a preset or type a supported device, such as cuda:1.')
    return dropdown


def path_input_row(parent, line_edit, browse_button):
    """Pair an editable path and a browse button with matching heights."""
    row = QtWidgets.QWidget(parent)
    row.setObjectName(line_edit.objectName() + '_row')
    layout = QtWidgets.QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    line_edit.setReadOnly(False)
    line_edit.setCursorPosition(0)
    browse_button.setAutoDefault(False)
    browse_button.setFixedSize(28, line_edit.sizeHint().height())
    layout.addWidget(line_edit)
    layout.addWidget(browse_button)
    row.setFocusProxy(line_edit)
    return row


def config_form(dialog, rows, save_button, automatic_device=None):
    """Lay out (label, field[, browse_button]) rows and a Save button.

    automatic_device adds a read-only device row for OpenCV/TensorFlow."""
    form = QtWidgets.QFormLayout(dialog)
    form.setContentsMargins(10, 10, 10, 10)
    form.setHorizontalSpacing(9)
    form.setVerticalSpacing(6)
    form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    alignment = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
    form.setLabelAlignment(alignment)
    parameter_rows = list(rows)
    if automatic_device:
        device_note = QtWidgets.QLabel(f'auto by {automatic_device}', dialog)
        device_note.setObjectName('device_note_label')
        parameter_rows.append(('device', device_note))
    for label, field, *browse in parameter_rows:
        if isinstance(label, str):
            label = QtWidgets.QLabel(label, dialog)
        label.setFixedWidth(91)
        label.setAlignment(alignment)
        label.setBuddy(field)
        if browse:
            field = path_input_row(dialog, field, browse[0])
        form.addRow(label, field)

    save_button.setFixedSize(91, 31)
    font = QtGui.QFont()
    font.setPointSize(12)
    save_button.setFont(font)
    save_button.setDefault(True)
    actions = QtWidgets.QHBoxLayout()
    actions.setContentsMargins(0, 14, 0, 0)
    actions.addStretch()
    actions.addWidget(save_button)
    actions.addStretch()
    form.addRow(actions)
    dialog.setFixedSize(390, dialog.sizeHint().height())
    return form
