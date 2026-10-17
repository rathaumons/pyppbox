# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Host logging, owned runtime files, and independent GUI process handoffs."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import Mock, patch

from pyppbox.gui import guihub, guitools
from pyppbox.utils import runtime


class RuntimeIsolationTests(unittest.TestCase):
    def run_python(self, code, directory, **environment):
        env = dict(os.environ, PYPPBOX_RUNTIME_DIR=str(directory),
                   PYPPBOX_DISABLE_FILE_LOG='0', PYPPBOX_DISABLE_TERMINAL_LOG='1',
                   PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYPPBOX_GUI_SESSION_DIR', None)
        env.update(environment)
        result = subprocess.run([sys.executable, '-c', textwrap.dedent(code)],
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_file_handler_does_not_require_newer_logging_private_closed_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_python('''
                import logging
                from pathlib import Path
                from pyppbox.utils import logtools as logs
                logs.add_info_log('first')
                handler = next(h for h in logs.__logger__.handlers if getattr(h, '_pyppbox_file', False))
                handler.__dict__.pop('_closed', None)  # Python 3.9 FileHandler has no such field.
                logs.add_info_log('second')
                path = Path(handler.baseFilename)
                before = path.read_bytes()
                assert b'first' in before and b'second' in before
                handler.close()
                handler.__dict__.pop('_closed', None)
                handler.emit(logging.LogRecord('pyppbox', logging.INFO, '', 0, 'after close', (), None))
                assert handler.stream is None and path.read_bytes() == before
            ''', directory)

    def test_enabled_logging_preserves_root_and_keeps_foreign_records_out_of_own_file(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_python('''
                import logging
                from pathlib import Path
                root = logging.getLogger()
                before = root.level, list(root.handlers), root.disabled
                from pyppbox.utils import logtools as logs
                assert not Path(logs.__log_dir__).exists()  # lazy even when enabled
                logs.add_info_log('person identity α')
                logging.getLogger('host.application').warning('host only')
                assert before == (root.level, list(root.handlers), root.disabled)
                file = Path(logs.__log_txt_path__)
                assert file.is_relative_to(Path(__import__('os').environ['PYPPBOX_RUNTIME_DIR']))
                text = file.read_text(encoding='utf-8')
                assert 'person identity α' in text and 'host only' not in text
            ''', directory)

    def test_host_handler_receives_named_records_even_without_file_logging(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_python('''
                import io, logging
                from pathlib import Path
                output = io.StringIO()
                handler = logging.StreamHandler(output)
                root = logging.getLogger(); root.addHandler(handler); root.setLevel(logging.INFO)
                from pyppbox.utils import logtools as logs
                logs.add_info_log('named event')
                assert 'named event' in output.getvalue()
                assert root.handlers == [handler] and root.level == logging.INFO
                assert not Path(logs.__log_dir__).exists()
                assert logs.__logger__.name == 'pyppbox'
            ''', directory, PYPPBOX_DISABLE_FILE_LOG='1')

    def test_terminal_toggles_and_explicit_override_keep_original_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.run_python('''
                from pyppbox.utils import logtools as logs
                logs.add_info_log('hidden')
                logs.add_warning_log('explicit', terminal_log=True)
                logs.enable_terminal_log(); logs.add_info_log('enabled')
                logs.add_error_log('hidden', terminal_log=False)
                logs.disable_terminal_log(); logs.add_info_log('hidden')
            ''', directory, PYPPBOX_DISABLE_FILE_LOG='1')
            self.assertEqual(output, 'explicit\nenabled\n')

    def test_toggle_reload_and_host_logger_settings_do_not_duplicate_or_truncate(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_python('''
                import importlib, logging, os
                from pathlib import Path
                os.environ['PYPPBOX_RUNTIME_DIR'] = os.path.join(os.environ['PYPPBOX_RUNTIME_DIR'], 'unused', '..')
                from pyppbox.utils import logtools as logs
                host_handler = logging.NullHandler()
                logs.__logger__.addHandler(host_handler)
                logs.add_info_log('first')
                original = Path(logs.__log_txt_path__)
                importlib.reload(logs); logs.add_info_log('second')
                assert Path(logs.__log_txt_path__) == original
                assert len([h for h in logs.__logger__.handlers if getattr(h, '_pyppbox_file', False)]) == 1
                assert original.read_text().count('first') == 1
                assert original.read_text().count('second') == 1
                os.environ['PYPPBOX_DISABLE_FILE_LOG'] = '1'; logs.set_file_log_from_env()
                assert host_handler in logs.__logger__.handlers
                assert not any(getattr(h, '_pyppbox_file', False) for h in logs.__logger__.handlers)
                logs.__logger__.setLevel(logging.ERROR)
                os.environ['PYPPBOX_DISABLE_FILE_LOG'] = '0'; logs.set_file_log_from_env()
                logs.add_error_log('third')
                assert logs.__logger__.level == logging.ERROR
                assert Path(logs.__log_txt_path__) != original
            ''', directory)

    def test_unwritable_sink_and_disk_error_are_nonfatal_and_leave_host_handler(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_python('''
                import io, logging, os
                from pathlib import Path
                from unittest.mock import patch
                folder = Path(os.environ['PYPPBOX_RUNTIME_DIR']); blocked = folder / 'blocked'
                blocked.write_text('not a directory')
                os.environ['PYPPBOX_RUNTIME_DIR'] = str(blocked)
                output = io.StringIO(); host_handler = logging.StreamHandler(output)
                root = logging.getLogger(); root.addHandler(host_handler); root.setLevel(logging.INFO)
                from pyppbox.utils import logtools as logs
                logs.add_error_log('original backend failure')
                logs.add_error_log('second backend failure')
                assert output.getvalue().count('File logging disabled') == 1
                assert 'original backend failure' in output.getvalue()
                os.environ['PYPPBOX_RUNTIME_DIR'] = str(folder)
                logs.set_file_log_from_env(); logs.add_info_log('recovered')
                owned = next(h for h in logs.__logger__.handlers if getattr(h, '_pyppbox_file', False))
                with patch.object(owned.stream, 'write', side_effect=OSError('disk full')):
                    logs.add_error_log('inference error still visible')
                assert not logs.__file_log__
                assert 'inference error still visible' in output.getvalue()
                assert root.handlers == [host_handler]
            ''', directory)

    def test_concurrent_process_logs_have_unique_files_even_in_one_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            code = '''
                import json
                from pathlib import Path
                from unittest.mock import patch
                with patch('time.strftime', return_value='20261017_010203'):
                    from pyppbox.utils import logtools as logs
                    logs.add_info_log('owned process event')
                    print(json.dumps(logs.__log_txt_path__))
            '''
            with ThreadPoolExecutor(max_workers=4) as pool:
                files = list(pool.map(lambda _: json.loads(self.run_python(code, directory)), range(4)))
            self.assertEqual(len(set(files)), 4)
            for file in files:
                self.assertEqual(Path(file).read_text().count('owned process event'), 1)

    def test_threaded_first_messages_use_one_sink_and_preserve_all_records(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_python('''
                from concurrent.futures import ThreadPoolExecutor
                from pathlib import Path
                from pyppbox.utils import logtools as logs
                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(lambda i: logs.add_info_log('event-' + str(i)), range(64)))
                assert len(list(Path(logs.__log_dir__).glob('*.txt'))) == 1
                assert len(Path(logs.__log_txt_path__).read_text().splitlines()) == 64
            ''', directory)

    @unittest.skipUnless(hasattr(os, 'fork'), 'POSIX fork lock/session ownership')
    def test_fork_reinitializes_held_lock_and_opens_a_child_owned_log(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_python('''
                import os, signal, threading
                from pathlib import Path
                from pyppbox.utils import logtools as logs
                logs.add_info_log('parent record')
                ready, release = threading.Event(), threading.Event()
                def hold():
                    with logs._file_lock:
                        ready.set(); release.wait(10)
                thread = threading.Thread(target=hold); thread.start(); assert ready.wait(5)
                child = os.fork()
                if child == 0:
                    signal.alarm(5)
                    logs.add_info_log('child record')
                    os._exit(0)
                release.set(); thread.join(5)
                _, status = os.waitpid(child, 0)
                assert os.waitstatus_to_exitcode(status) == 0
                records = [path.read_text() for path in Path(logs.__log_dir__).glob('*.txt')]
                assert len(records) == 2
                assert sum('parent record' in text for text in records) == 1
                assert sum('child record' in text for text in records) == 1
                assert all(not ('parent record' in text and 'child record' in text) for text in records)
            ''', directory)

    def test_readonly_package_guard_allows_logging_and_gui_handoff_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_python('''
                import os, sys
                from pathlib import Path
                import pyppbox
                package = Path(pyppbox.__file__).resolve().parent
                def audit(event, args):
                    if event == 'open':
                        path, mode, flags = args
                        paths = [path] if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND) else []
                    elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod'):
                        paths = args[:1]
                    elif event == 'os.rename':
                        paths = args[:2]
                    else:
                        return
                    for path in paths:
                        if isinstance(path, (str, bytes, os.PathLike)) and Path(os.fsdecode(path)).resolve().is_relative_to(package):
                            raise PermissionError('Package is read-only')
                sys.addaudithook(audit)
                from pyppbox.utils import logtools as logs
                from pyppbox.gui import guihub
                logs.add_info_log('read-only installation works')
                with guihub._gui_session(str(package / 'config/cfg')) as env:
                    os.environ['PYPPBOX_GUI_SESSION_DIR'] = env['PYPPBOX_GUI_SESSION_DIR']
                    guihub.writeInputTMP('outside-package.mp4', True)
                    assert guihub.loadInputTMP() == ('outside-package.mp4', True)
                    assert guihub.loadUITMP() is None, (guihub.loadUITMP(), guihub.default_cfg)
                assert Path(logs.__log_txt_path__).exists()
            ''', directory)

    def test_internal_config_selection_recognizes_the_same_windows_path_case(self):
        if os.name != 'nt':
            self.skipTest('Windows case-insensitive paths')
        with tempfile.TemporaryDirectory() as directory:
            selection = str(Path(guihub.default_cfg).resolve()).swapcase()
            with patch.dict(os.environ, {'PYPPBOX_RUNTIME_DIR':directory}, clear=True):
                guihub.writeUITMP(selection)
                self.assertIsNone(guihub.loadUITMP())

    def test_retention_skips_current_process_files_and_handles_permission_errors(self):
        from pyppbox.utils import logtools
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / f'log_20200101_010101_{os.getpid()}_1234abcd.txt'
            old = root / 'log_20200101_010101.txt'
            for file in (active, old): file.write_text('old'); os.utime(file, (0, 0))
            with patch.object(logtools, '__log_dir__', directory):
                logtools.cleanup_old_logs()
                self.assertTrue(active.exists()); self.assertFalse(old.exists())
                with patch.object(logtools.os, 'listdir', side_effect=PermissionError('denied')):
                    logtools.cleanup_old_logs()

    def test_runtime_paths_are_lazy_platform_specific_and_overrideable(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with patch.dict(os.environ, {'PYPPBOX_RUNTIME_DIR': str(base / 'override')}, clear=True):
                self.assertEqual(Path(runtime.get_runtime_dir()), base / 'override')
                self.assertEqual(Path(runtime.get_gui_state_dir()), base / 'override/gui')
                self.assertFalse((base / 'override').exists())
                os.environ['PYPPBOX_RUNTIME_DIR'] = str(base / 'unused' / '..' / 'override')
                self.assertEqual(Path(runtime.get_runtime_dir()), base / 'override')
            for platform, env, suffix in (
                ('win32', {'LOCALAPPDATA': str(base)}, 'pyppbox'),
                ('darwin', {}, 'Library/Application Support/pyppbox'),
                ('linux', {'XDG_STATE_HOME': str(base)}, 'pyppbox'),
                ('linux', {}, '.local/state/pyppbox')):
                with self.subTest(platform=platform, env=env), patch.dict(os.environ, env, clear=True), \
                     patch.object(runtime.sys, 'platform', platform), patch.object(runtime.Path, 'home', return_value=base):
                    self.assertEqual(Path(runtime.get_runtime_dir()), base / suffix)

    def test_legacy_reads_new_utf8_writes_and_missing_or_malformed_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); legacy = root / 'installed/tmp'; legacy.mkdir(parents=True)
            (legacy / 'ui.tmp').write_text(str(root / 'old-config') + '\n', encoding='utf-8')
            (legacy / 'input.tmp').write_text('old-video.mp4\nTrue\n', encoding='utf-8')
            original = {p:p.read_bytes() for p in legacy.iterdir()}
            with patch.dict(os.environ, {'PYPPBOX_RUNTIME_DIR': str(root / 'user')}, clear=True), \
                 patch.object(guihub, 'current_dir', str(legacy.parent)):
                self.assertEqual(guihub.loadUITMP(), (root / 'old-config').as_posix())
                self.assertEqual(guihub.loadInputTMP(), ('old-video.mp4', True))
                guihub.writeUITMP(guihub.default_cfg); guihub.writeInputTMP('vidéo-α.mp4', False)
                self.assertIsNone(guihub.loadUITMP())
                self.assertEqual(guihub.loadInputTMP(), ('vidéo-α.mp4', False))
                for file, content in original.items(): self.assertEqual(file.read_bytes(), content)
                session = root / 'isolated'; session.mkdir()
                with patch.dict(os.environ, {'PYPPBOX_GUI_SESSION_DIR': str(session)}):
                    self.assertIsNone(guihub.loadUITMP()); self.assertEqual(guihub.loadInputTMP(), ('', False))
                    for content in ('', 'truncated', 'video\ninvalid', 'bad\x00path\nTrue', 'x' * 65537):
                        (session / 'input.tmp').write_text(content)
                        self.assertEqual(guihub.loadInputTMP(), ('', False))
                    (session / 'ui.tmp').write_text('\x00\n')
                    self.assertIsNone(guihub.loadUITMP())

    def test_failed_atomic_replace_keeps_complete_state_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'PYPPBOX_RUNTIME_DIR': directory}, clear=True):
            guihub.writeInputTMP('original.mp4', True)
            state = Path(directory) / 'gui/input.tmp'; original = state.read_bytes()
            with patch.object(guihub.os, 'replace', side_effect=OSError('replace interrupted')):
                with self.assertRaisesRegex(OSError, 'interrupted'): guihub.writeInputTMP('replacement.mp4', False)
            self.assertEqual(state.read_bytes(), original)
            self.assertEqual(list(state.parent.iterdir()), [state])
            for value in ('bad\npath', 'bad\rpath'):
                with self.assertRaises(ValueError): guihub.writeUITMP(value)
                with self.assertRaises(ValueError): guihub.writeInputTMP(value, False)

    def test_remember_failure_does_not_break_active_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'gui').write_text('blocked preferences')
            with patch.dict(os.environ, {'PYPPBOX_RUNTIME_DIR': str(root), 'PYPPBOX_GUI_SESSION_DIR': str(root / 'session')}, clear=True):
                guihub.writeUITMP(str(root / 'config')); guihub.writeInputTMP('live.mp4', True)
                self.assertEqual(guihub.loadUITMP(), (root / 'config').as_posix())
                self.assertEqual(guihub.loadInputTMP(), ('live.mp4', True))

    def test_launch_environments_are_independent_and_only_owned_state_is_cleaned(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'PYPPBOX_RUNTIME_DIR': directory}, clear=True):
            original = dict(os.environ)
            with guihub._gui_session(str(Path(directory) / 'config-a')) as first:
                first_path = Path(first['PYPPBOX_GUI_SESSION_DIR'])
                with guihub._gui_session(str(Path(directory) / 'config-b')) as second:
                    second_path = Path(second['PYPPBOX_GUI_SESSION_DIR'])
                    self.assertNotEqual(first_path, second_path)
                    for env, video in ((first, 'a.mp4'), (second, 'b.mp4')):
                        with patch.dict(os.environ, env, clear=True):
                            guihub.writeInputTMP(video, True)
                    with patch.dict(os.environ, first, clear=True):
                        self.assertEqual(guihub.loadInputTMP(), ('a.mp4', True))
                        self.assertTrue(guihub.loadUITMP().endswith('config-a'))
                        with guihub._gui_session(env=first, reuse=True) as inherited:
                            self.assertEqual(inherited, first)
                        self.assertTrue(first_path.exists())
                    self.assertEqual(dict(os.environ), original)
                self.assertFalse(second_path.exists()); self.assertTrue(first_path.exists())
            self.assertFalse(first_path.exists()); self.assertTrue((Path(directory) / 'gui/input.tmp').exists())

    def test_launch_gui_passes_session_to_child_and_cleans_up_on_process_start_failure(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'PYPPBOX_RUNTIME_DIR': directory}, clear=True):
            seen = []
            def child(*args, **kwargs):
                session = Path(kwargs['env']['PYPPBOX_GUI_SESSION_DIR']); seen.append(session)
                self.assertTrue((session / 'ui.tmp').is_file())
                self.assertNotIn('PYPPBOX_GUI_SESSION_DIR', os.environ)
                return Mock(wait=Mock(return_value=0))
            with patch.object(guitools.sp, 'Popen', side_effect=child): guitools.launchGUI()
            self.assertFalse(seen[0].exists())
            with patch.object(guitools.sp, 'Popen', side_effect=OSError('start failed')):
                with self.assertRaisesRegex(OSError, 'start failed'): guitools.launchGUI()
            self.assertEqual(list((Path(directory) / 'sessions').iterdir()), [])

    def test_structure_points_at_runtime_state_without_changing_config_paths(self):
        from pyppbox.config.configtools import PYPPBOXStructure, internal_cfg_dir
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'PYPPBOX_RUNTIME_DIR': directory}, clear=True):
            structure = PYPPBOXStructure()
            self.assertEqual(structure.cfg_dir, internal_cfg_dir)
            self.assertEqual(Path(structure.gui_tmp_dir), Path(directory) / 'gui')
            self.assertFalse((Path(directory) / 'gui').exists())

    def test_interrupted_gui_wait_stops_owned_launcher_before_cleaning_handoff(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'PYPPBOX_RUNTIME_DIR': directory}, clear=True):
            child = Mock(wait=Mock(side_effect=[KeyboardInterrupt(), 0]))
            with patch.object(guitools.sp, 'Popen', return_value=child):
                with self.assertRaises(KeyboardInterrupt): guitools.launchGUI()
            child.terminate.assert_called_once(); child.kill.assert_not_called()
            self.assertEqual(list((Path(directory) / 'sessions').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
