# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Real Qt launcher/demo process ownership, with deterministic backend fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import psutil


class GUIProcessCleanupTests(unittest.TestCase):
    def cleanup_fixtures(self,root,helper):
        owned={}
        for path in root.glob('*.json'):
            record=json.loads(path.read_text())
            for pid in (record['pid'],record['parent']):
                try:
                    process=psutil.Process(pid)
                    if str(root) in process.cmdline() and any(Path(arg).name==helper.name for arg in process.cmdline()):
                        owned[pid]=process
                except (psutil.NoSuchProcess,psutil.AccessDenied): pass
        for process in owned.values():
            try: process.kill()
            except psutil.NoSuchProcess: pass
        psutil.wait_procs(list(owned.values()),timeout=3)

    def test_cleanup_does_not_stop_an_unrelated_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); other=root/'other'; other.mkdir()
            helper=Path(__file__).with_name('gui_process_probe.py')
            unrelated=subprocess.Popen([sys.executable,str(helper),'leaf',str(other),'unrelated'],
                                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYPPBOX_RUNTIME_DIR=str(root/'runtime'),
                         PYPPBOX_DISABLE_FILE_LOG='1',PYPPBOX_DISABLE_TERMINAL_LOG='1')
                with (root/'output.log').open('w') as output:
                    result=subprocess.run([sys.executable,str(helper),'api',directory,'interrupt'],env=env,
                                          stdout=output,stderr=subprocess.STDOUT,timeout=30)
                self.assertEqual(result.returncode,0,(root/'output.log').read_text(errors='replace'))
                self.assertIsNone(unrelated.poll())
            finally:
                unrelated.kill(); unrelated.wait(timeout=5)
                self.cleanup_fixtures(root,helper)

    def test_api_interrupt_launcher_exit_and_demo_failure_leave_no_descendants(self):
        for mode,scenario in (('api','interrupt'),('api','launcher-exit'),('api','demo-error'),
                              ('launcher','demo-error'),('api','api-exit'),('api','stubborn-interrupt')):
            with self.subTest(mode=mode,scenario=scenario), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYPPBOX_RUNTIME_DIR=str(root/'runtime'),
                         PYPPBOX_DISABLE_FILE_LOG='1',PYPPBOX_DISABLE_TERMINAL_LOG='1',PYTHONDONTWRITEBYTECODE='1')
                helper=Path(__file__).with_name('gui_process_probe.py')
                processes=[]
                try:
                    with (root/'output.log').open('w',encoding='utf-8') as output:
                        result=subprocess.run([sys.executable,str(helper),mode,directory,scenario],env=env,
                                              stdout=output,stderr=subprocess.STDOUT,text=True,timeout=30)
                    details=(root/'output.log').read_text(encoding='utf-8',errors='replace')
                    self.assertEqual(result.returncode,0,details)
                    self.assertTrue((root/'leaf.json').is_file(),details)
                    for path in root.glob('*.json'):
                        pid=json.loads(path.read_text())['pid']
                        try: processes.append(psutil.Process(pid))
                        except psutil.NoSuchProcess: pass
                    deadline=time.monotonic()+3
                    def alive():
                        found=[]
                        for process in processes:
                            try:
                                if process.is_running() and process.status()!=psutil.STATUS_ZOMBIE: found.append(process.pid)
                            except psutil.NoSuchProcess: pass
                        return found
                    while alive() and time.monotonic()<deadline: time.sleep(.02)
                    self.assertEqual(alive(),[],'Owned GUI descendants survived')
                    if mode=='api' and scenario!='api-exit':
                        self.assertEqual(list((root/'runtime/sessions').iterdir()),[])
                finally:
                    self.cleanup_fixtures(root,helper)


if __name__ == '__main__':
    unittest.main()
