# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""A link label that stays readable when the application theme changes."""
from PyQt6 import QtCore, QtGui, QtWidgets


def _luminance(color):
    channels = (color.redF(), color.greenF(), color.blueF())
    linear = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4
              for value in channels]
    return sum(value * weight for value, weight in zip(linear, (.2126, .7152, .0722)))


class ThemedLinkLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        QtWidgets.QApplication.instance().paletteChanged.connect(self._update_color)
        self._update_color()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.Type.PaletteChange,
                            QtCore.QEvent.Type.ApplicationPaletteChange,
                            QtCore.QEvent.Type.StyleChange):
            self._update_color()

    def _update_color(self, *_):
        parent = self.parentWidget()
        # The transparent background is painted by the parent.
        palette = parent.palette() if parent is not None else QtWidgets.QApplication.palette()
        background = _luminance(palette.color(QtGui.QPalette.ColorRole.Window))

        def contrast(color):
            foreground = _luminance(color)
            return (max(background, foreground) + .05) / (min(background, foreground) + .05)

        # Prefer the theme's link color, with accessible blue fallbacks for
        # palettes that retain a dark blue link color on a dark background.
        candidates = [palette.color(QtGui.QPalette.ColorRole.Link),
                      QtGui.QColor('#80c4ff'), QtGui.QColor('#005a9e'),
                      palette.color(QtGui.QPalette.ColorRole.WindowText),
                      QtGui.QColor('black'), QtGui.QColor('white')]
        color = next(candidate for candidate in candidates if contrast(candidate) >= 4.5)
        if self.palette().color(QtGui.QPalette.ColorRole.WindowText) != color:
            foreground = QtGui.QPalette()
            foreground.setColor(QtGui.QPalette.ColorRole.WindowText, color)
            self.setPalette(foreground)
