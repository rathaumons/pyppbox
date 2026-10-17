# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Ownership failures must close handles and never signal the host session."""
import os
import subprocess
import unittest
from unittest.mock import Mock, patch

from pyppbox.gui import processes


class GUIProcessOwnershipTests(unittest.TestCase):
    def test_monitor_ignores_missing_invalid_or_nonleader_ownership(self):
        for parent in ('', 'not-a-pid', '-1', '0', '1234'):
            with self.subTest(parent=parent), patch.dict(os.environ,{'PYPPBOX_GUI_PARENT_PID':parent},clear=True), \
                 patch.object(processes.os,'name','posix'), \
                 patch.object(processes.os,'getsid',return_value=-1,create=True), \
                 patch('threading.Thread') as thread:
                processes.watch_gui_parent()
                thread.assert_not_called()

    def test_failed_spawn_closes_the_private_job(self):
        with patch.object(processes.os,'name','nt'),patch.object(processes,'_WindowsJob') as jobs:
            start=Mock(side_effect=OSError('Cannot create process'))
            with self.assertRaisesRegex(OSError,'Cannot create'):
                with processes.gui_process(['python','child.py'],popen=start):
                    self.fail('A failed spawn must not yield a process')
            jobs.return_value.close.assert_called_once()

    def test_failed_job_assignment_stops_the_suspended_child(self):
        child=Mock(wait=Mock(side_effect=[subprocess.TimeoutExpired('child',5),0]))
        with patch.object(processes.os,'name','nt'),patch.object(processes,'_WindowsJob') as jobs:
            jobs.return_value.attach_and_resume.side_effect=OSError('Cannot assign child')
            with self.assertRaisesRegex(OSError,'Cannot assign'):
                with processes.gui_process(['python','child.py'],popen=Mock(return_value=child)):
                    self.fail('An unowned suspended child must not be returned')
            jobs.return_value.close.assert_called_once()
            child.kill.assert_called_once()


if __name__=='__main__': unittest.main()
