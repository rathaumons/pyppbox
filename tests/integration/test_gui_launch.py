"""Launcher subprocess failures must remain visible even without a console."""
import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

from pyppbox.gui import guihub

sys.path.insert(0, str(Path(guihub.__file__).parent))
launcher_module = importlib.import_module('ui_launcher')


class GUILaunchTests(unittest.TestCase):
    def launcher(self):
        ui = launcher_module.Ui_PYPPBOXLauncher.__new__(launcher_module.Ui_PYPPBOXLauncher)
        ui.updateMCFG = Mock()
        ui.mycfg = Mock()
        for name in ('detector_comboBox', 'tracker_comboBox', 'reider_comboBox',
                     'input_video_file_lineEdit', 'input_force_hd_comboBox'):
            setattr(ui, name, Mock())
        return ui

    def test_failed_demo_shows_backend_error_and_restores_launcher(self):
        def failed_child(*args, **kwargs):
            kwargs['stderr'].write(b'FileNotFoundError: missing checkpoint.pth\n')
            return Mock(wait=Mock(return_value=1))
        ui = self.launcher()
        with patch.object(launcher_module, 'launcher', create=True) as window, \
             patch.object(launcher_module.sp, 'Popen', side_effect=failed_child), \
             patch.object(launcher_module.QtWidgets, 'QMessageBox') as message:
            ui.launchNow()
            window.show.assert_called_once()
            message.return_value.setDetailedText.assert_called_once_with(
                'FileNotFoundError: missing checkpoint.pth')
            message.return_value.exec.assert_called_once()

    def test_successful_demo_does_not_show_error(self):
        ui = self.launcher()
        with patch.object(launcher_module, 'launcher', create=True) as window, \
             patch.object(launcher_module.sp, 'Popen', return_value=Mock(wait=Mock(return_value=0))), \
             patch.object(launcher_module.QtWidgets, 'QMessageBox') as message:
            ui.launchNow()
            window.show.assert_called_once()
            message.assert_not_called()

    def test_state_or_config_save_failure_is_visible_without_starting_demo(self):
        ui = self.launcher()
        ui.updateMCFG.side_effect = OSError('Settings directory is read-only')
        with patch.object(launcher_module, 'launcher', create=True) as window, \
             patch.object(launcher_module.sp, 'Popen') as popen, \
             patch.object(launcher_module.QtWidgets, 'QMessageBox') as message:
            ui.launchNow()
            window.hide.assert_not_called(); popen.assert_not_called()
            message.critical.assert_called_once_with(window, 'Cannot save launch settings',
                                                     'Settings directory is read-only')

    def test_process_start_failure_restores_launcher_and_shows_error(self):
        ui = self.launcher()
        with patch.object(launcher_module, 'launcher', create=True) as window, \
             patch.object(launcher_module.sp, 'Popen', side_effect=OSError('Cannot start Python')), \
             patch.object(launcher_module.QtWidgets, 'QMessageBox') as message:
            ui.launchNow()
            window.show.assert_called_once()
            message.return_value.setDetailedText.assert_called_once_with('Cannot start Python')

    @unittest.skipUnless(sys.platform == 'win32', 'Windows pythonw subprocess behavior')
    def test_pythonw_uses_same_environment_without_a_console(self):
        ui = self.launcher()
        with patch.object(launcher_module, 'launcher', create=True), \
             patch.object(launcher_module.sys, 'executable', 'C:/test/.venv/Scripts/pythonw.exe'), \
             patch.object(launcher_module.sys, 'stdout', None), \
             patch.object(launcher_module.sp, 'Popen', return_value=Mock(wait=Mock(return_value=0))) as popen:
            ui.launchNow()
            self.assertEqual(Path(popen.call_args.args[0][0]), Path('C:/test/.venv/Scripts/python.exe'))
            options = popen.call_args.kwargs
            self.assertEqual(options['creationflags'], launcher_module.sp.CREATE_NO_WINDOW)
            self.assertIs(options['stdout'], options['stderr'])


if __name__ == '__main__':
    unittest.main()
