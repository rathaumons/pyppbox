# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Private real-process fixture; never used by the application launcher."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def main():
    mode, directory, scenario = sys.argv[1:]
    try:
        from pyppbox.gui.processes import watch_gui_parent
    except ModuleNotFoundError:
        pass  # Frozen pre-fix baseline has no ownership helper.
    else:
        if mode in ('launcher','demo'): watch_gui_parent()
    root = Path(directory)
    (root / f'{mode}.json').write_text(json.dumps(dict(pid=os.getpid(),parent=os.getppid())),encoding='utf-8')
    original_popen = subprocess.Popen
    def child(kind):
        return original_popen([sys.executable, __file__, kind, directory, scenario], env=os.environ.copy())
    def ready():
        deadline = time.monotonic()+15
        while True:
            try:
                if json.loads((root/'leaf.json').read_text())['pid']>0: return
            except (OSError,ValueError,KeyError): pass
            if time.monotonic() > deadline:
                raise TimeoutError('Fixture descendant did not start')
            time.sleep(.02)
    if mode == 'leaf':
        if scenario=='stubborn-interrupt' and os.name=='posix':
            import signal
            signal.signal(signal.SIGTERM,signal.SIG_IGN)
        time.sleep(60)
    elif mode == 'demo':
        child('leaf')
        ready()
        if scenario == 'demo-error':
            raise RuntimeError('Controlled fixture demo failure')
        time.sleep(60)
    elif mode == 'launcher':
        from PyQt6 import QtWidgets
        from pyppbox.gui import guihub
        sys.path.insert(0,str(Path(guihub.__file__).parent))
        module = importlib.import_module('ui_launcher')
        app = QtWidgets.QApplication([])
        module.launcher = QtWidgets.QMainWindow()
        ui = module.Ui_PYPPBOXLauncher.__new__(module.Ui_PYPPBOXLauncher)
        ui.updateMCFG = lambda *args:None
        ui.mycfg = SimpleNamespace(setMCFG=lambda:None)
        for name in ('detector_comboBox','tracker_comboBox','reider_comboBox','input_force_hd_comboBox'):
            combo = QtWidgets.QComboBox(); combo.addItem('None'); setattr(ui,name,combo)
        ui.input_video_file_lineEdit = QtWidgets.QLineEdit()
        module.QtWidgets.QMessageBox = lambda *args:SimpleNamespace(
            setIcon=lambda *args:None,setWindowTitle=lambda *args:None,setText=lambda *args:None,
            setDetailedText=lambda *args:None,exec=lambda:None)
        module.QtWidgets.QMessageBox.Icon = SimpleNamespace(Critical=1)
        def popen(command, **options):
            if Path(command[1]).name == 'guidemo.py':
                process = original_popen([command[0],__file__,'demo',directory,scenario],**options)
                if scenario == 'launcher-exit':
                    import threading
                    def exit_when_ready():
                        ready(); os._exit(0)
                    threading.Thread(target=exit_when_ready,daemon=True).start()
                return process
            return original_popen(command,**options)
        module.sp.Popen = popen
        ui.launchNow()
    elif mode == 'api':
        from pyppbox.gui import guitools
        def popen(command, **options):
            process = original_popen([command[0],__file__,'launcher',directory,scenario],**options)
            if scenario=='api-exit':
                import threading
                def exit_when_ready():
                    ready(); os._exit(0)
                threading.Thread(target=exit_when_ready,daemon=True).start()
            if scenario in ('interrupt','stubborn-interrupt'):
                original_wait = process.wait
                first = True
                def wait(*args,**kwargs):
                    nonlocal first
                    if first:
                        first=False; ready(); raise KeyboardInterrupt()
                    return original_wait(*args,**kwargs)
                process.wait=wait
            return process
        guitools.sp.Popen = popen
        try:
            guitools.launchGUI()
        except KeyboardInterrupt:
            if scenario not in ('interrupt','stubborn-interrupt'): raise


if __name__ == '__main__':
    main()
