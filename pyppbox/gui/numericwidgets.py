# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Numeric config controls with compact text and lossless float round trips."""
from decimal import Decimal
import math

from PyQt6 import QtCore, QtWidgets


class FloatSpinBox(QtWidgets.QDoubleSpinBox):
    """Display concise decimals without rounding loaded configuration values."""

    def __init__(self, parent=None, minimum=0.0, maximum=1.0, step=0.05):
        super().__init__(parent)
        self.setLocale(QtCore.QLocale.c())
        self.setDecimals(323)
        self.setRange(minimum, maximum)
        self.setSingleStep(step)

    def textFromValue(self, value):
        text = format(Decimal(str(value)), 'f')
        return text.rstrip('0').rstrip('.') if '.' in text else text

    def setValue(self, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('A numeric setting must be finite.')
        # Keep pre-existing values outside the suggested editor range unchanged.
        self.setRange(min(self.minimum(), value), max(self.maximum(), value))
        super().setValue(value)

    def stepBy(self, steps):
        self.interpretText()
        value = Decimal(str(self.value())) + Decimal(str(self.singleStep())) * steps
        self.setValue(min(self.maximum(), max(self.minimum(), float(value))))


class IntSpinBox(QtWidgets.QSpinBox):
    """Edit integer settings and preserve loaded values outside nominal bounds."""

    def __init__(self, parent=None, minimum=0, maximum=2147483647, step=1):
        super().__init__(parent)
        self.setRange(minimum, maximum)
        self.setSingleStep(step)

    def setValue(self, value):
        value = int(value)
        self.setRange(min(self.minimum(), value), max(self.maximum(), value))
        super().setValue(value)


class OptionalIntSpinBox(IntSpinBox):
    """Represent an unlimited budget using the existing None configuration value."""

    def __init__(self, parent=None):
        super().__init__(parent, minimum=-1)
        self.setSpecialValueText('None')
        self.setToolTip('None: keep an unlimited feature budget')

    def setValue(self, value):
        super().setValue(-1 if value is None else value)

    def value(self):
        value = super().value()
        return None if value == -1 else value


class IntPairEdit(QtWidgets.QWidget):
    """Edit two signed crop offsets while saving the original two-item list."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.first_spinBox = IntSpinBox(self, minimum=-2147483648)
        self.second_spinBox = IntSpinBox(self, minimum=-2147483648)
        for widget in (self.first_spinBox, self.second_spinBox):
            widget.setFixedHeight(21)
            layout.addWidget(widget)
        self.first_spinBox.setToolTip('First crop offset in pixels')
        self.second_spinBox.setToolTip('Second crop offset in pixels')
        self.setFocusProxy(self.first_spinBox)

    def setValue(self, values):
        self.first_spinBox.setValue(values[0])
        self.second_spinBox.setValue(values[1])

    def value(self):
        return [self.first_spinBox.value(), self.second_spinBox.value()]
