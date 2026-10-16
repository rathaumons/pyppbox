# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Fresh-process import boundaries and the standalone facade's shared state."""
import os
import subprocess
import sys
import textwrap
import unittest


BLOCKER = '''
import importlib.abc
import sys
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in forbidden):
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockImports())
'''


class StartupTests(unittest.TestCase):
    def run_python(self, code, forbidden=()):
        script = 'forbidden = ' + repr(forbidden) + '\n' + BLOCKER + textwrap.dedent(code)
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
            timeout=30, env=dict(os.environ, PYPPBOX_DISABLE_FILE_LOG='1', PYPPBOX_DISABLE_TERMINAL_LOG='1'))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_config_and_catalog_metadata_do_not_import_runtime_backends(self):
        self.run_python('''
            from pyppbox.config.myconfig import MyConfigurator
            cfg = MyConfigurator(set_all_modules=True)
            assert cfg.rcfg_torchreid.model_wh == (128, 256)
            assert cfg.rcfg_torchreid.model_dict.getWH('OSNET_AIN_MS_D_C.PTH.TAR') == (128, 256)
            assert cfg.rcfg_torchreid.model_dict.getWH('unknown.pt') == (0, 0)
            assert cfg.getMCFG()['reider']
        ''', ('pyppbox.modules', 'sklearn', 'scipy', 'torch', 'tensorflow', 'ultralytics', 'rfdetr'))

    def test_sort_runtime_does_not_import_demo_plotting_or_image_libraries(self):
        self.run_python('''
            import numpy as np
            np.random.seed(837)
            expected = np.random.random(4)
            np.random.seed(837)
            from types import SimpleNamespace
            from pyppbox.modules.trackers.sort import MySORT
            from pyppbox.utils.persontools import Person
            np.testing.assert_array_equal(np.random.random(4), expected)
            tracker = MySORT(SimpleNamespace(max_age=2, min_hits=1, iou_threshold=.3))
            def person():
                return Person(0, -1, box_xywh=np.array([0, 0, 20, 40]), box_xyxy=np.array([0, 0, 20, 40]))
            first = tracker.update([person()])[0]
            first.deepid = 'Alice'
            returned = tracker.update([person()])[0]
            assert returned.cid == first.cid and returned.deepid == 'Alice'
            assert tracker.update([]) == []
        ''', ('matplotlib', 'skimage'))

    def test_importing_facade_functions_does_not_import_the_pipeline(self):
        self.run_python('''
            import inspect
            import pyppbox.ppb as api
            import pyppbox.standalone as facade
            from pyppbox.standalone import getConfig, detectPeople
            assert api._mt is None
            assert getConfig is api.getConfig and detectPeople is api.detectPeople
            assert set(api.__all__).issubset(dir(facade))
            assert set(api.__all__).issubset(dir(api))
            assert inspect.signature(facade.detectPeople) == inspect.signature(api.detectPeople)
            assert api._mt is None
        ''', ('pyppbox.ppb.mt', 'cv2', 'numpy', 'scipy', 'sklearn', 'torch', 'tensorflow'))

    def test_legacy_catalog_imports_and_pickle_names_remain_valid(self):
        self.run_python('''
            import pickle
            from pathlib import Path
            import tempfile
            from pyppbox.config import torchreidmodels as metadata
            from pyppbox.modules.reiders.torchreid import model_dict as legacy
            assert legacy.TorchreidModel is metadata.TorchreidModel
            assert legacy.TorchreidModelDict is metadata.TorchreidModelDict
            assert legacy.default_model_dict_yaml == metadata.default_model_dict_yaml
            assert Path(legacy.default_model_dict_yaml).is_file()
            assert pickle.loads(b'cpyppbox.modules.reiders.torchreid.model_dict\\nTorchreidModel\\n.') is metadata.TorchreidModel
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'models.yaml'
                path.write_text('name: first\\narch: old\\nheight: 256\\nwidth: 128\\nmodel_files: [custom.pth]\\n'
                    '---\\nname: second\\narch: new\\nheight: 100\\nwidth: 50\\nmodel_files: [CUSTOM.PTH]\\n')
                catalog = metadata.TorchreidModelDict(str(path))
                assert catalog.getWH('Custom.pth') == (50, 100)
                assert catalog.findModelArch('custom.pth') == 'new'
                assert catalog.getWH('missing.pth') == (0, 0)
                assert catalog.findModelArch('missing.pth') == ''
        ''')

    def test_concurrent_first_calls_create_one_shared_pipeline(self):
        self.run_python('''
            from concurrent.futures import ThreadPoolExecutor
            from types import SimpleNamespace
            import time
            from unittest.mock import Mock, patch
            import pyppbox.ppb as api
            import pyppbox.ppb.mt as runtime
            shared = object()
            def construct():
                time.sleep(.02)
                return SimpleNamespace(getConfig=lambda: shared)
            factory = Mock(side_effect=construct)
            with patch.object(runtime, 'MT', factory):
                with ThreadPoolExecutor(max_workers=8) as pool:
                    results = list(pool.map(lambda _: api.getConfig(), range(16)))
                assert all(result is shared for result in results)
                assert factory.call_count == 1
        ''')

    def test_failed_first_construction_can_be_retried(self):
        self.run_python('''
            from types import SimpleNamespace
            from unittest.mock import patch
            import pyppbox.ppb as api
            import pyppbox.ppb.mt as runtime
            fake = SimpleNamespace(getMainConfig=lambda current=True: {'detector': 'None'})
            with patch.object(runtime, 'MT', side_effect=[RuntimeError('injected'), fake]) as factory:
                try:
                    api.getMainConfig()
                except RuntimeError as error:
                    assert str(error) == 'injected'
                else:
                    raise AssertionError('Constructor failure was swallowed')
                assert api._mt is None
                assert api.getMainConfig() == {'detector': 'None'}
                assert factory.call_count == 2
        ''')

    def test_standalone_aliases_share_state_and_explicit_MT_instances_are_independent(self):
        self.run_python('''
            import numpy as np
            import pyppbox.ppb as api
            import pyppbox.standalone as facade
            from pyppbox.standalone import *
            from pyppbox.ppb.mt import MT as RealMT
            from pyppbox.utils.persontools import Person
            assert MT is facade.MT is api.MT is RealMT
            assert all(getattr(facade, name) is getattr(api, name) for name in api.__all__)
            first, second = MT(), MT()
            assert first.getConfig() is not second.getConfig()
            shared = getConfig()
            assert shared is api.getConfig() is facade.getConfig()
            assert shared is not first.getConfig()
            setMainModules({'detector': 'None', 'tracker': 'None', 'reider': 'None'}, load_all=False)
            assert api.getMainConfig(current=False) == {'detector': 'None', 'tracker': 'None', 'reider': 'None'}
            facade.setMainTracker({'tk_name': 'Centroid', 'max_spread': 30})
            assert api.getMainConfig()['tracker'] == 'Centroid'
            frame = np.zeros((60, 120, 3), dtype=np.uint8)
            def person(x):
                return Person(0, -1, box_xywh=np.array([x, 0, 20, 40]), box_xyxy=np.array([x, 0, x+20, 40]))
            previous = trackPeople(frame, [person(10)], img_is_mat=True)[0]
            previous.deepid = 'Alice'
            returned = api.trackPeople(frame, [person(11)], img_is_mat=True)[0]
            assert returned.cid == previous.cid and returned.deepid == 'Alice'
            assert first.getMainConfig()['tracker'] == 'None'
        ''')


if __name__ == '__main__':
    unittest.main()
