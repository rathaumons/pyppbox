# Release Notes 

## **pyppbox V4 - Even Smaller**

* `pyppbox` v4.0b1 (COMING SOON 🚀)

  - `pyppbox` will leverage [`vsensebox`](https://github.com/rathaumons/vsensebox) for detection and tracking
  - `pyppbox.utils.persontools` will change to adapt [`vsensebox`](https://github.com/rathaumons/vsensebox)
  - `pyppbox.ppb` will replace `pyppbox.standalone`
  - All configuration files will also change
  - More changes will be added here! Stay tuned!

## **pyppbox V3 - Make Simpler and Faster**

* `pyppbox` [v3.13.0](https://github.com/rathaumons/pyppbox/tree/v3.13.0) - The Last V3

  - Fix a bug and improve performance of SORT tracker:
    - Refactor and optimize SORT tracking algorithm
    - Fix a bug where unmatched tracks are not cleared after timeout
    - Improve and enhance other internal functions
    - Switch default assignment solver to the new `lapx`'s [`lapjvxa()`](https://github.com/rathaROG/lapx#3-the-new-function-lapjvxa)
  - Improve performance of Centroid tracker by using the new `lapx`'s [`lapjvxa()`](https://github.com/rathaROG/lapx#3-the-new-function-lapjvxa)
  - Improve the internal logging control:
    - Add environment variable `PYPPBOX_DISABLE_FILE_LOG` to allow users to disable internal temporary text log files in `pyppbox/data/logs` (disabled by default); can be activated as follows:
      - On Linux terminal: `export PYPPBOX_DISABLE_FILE_LOG=1`
      - On Windows terminal: `set PYPPBOX_DISABLE_FILE_LOG=1`
    - Add environment varible `PYPPBOX_DISABLE_TERMINAL_LOG` to allow users to disable the terminal log completely; can be toggled with the built-in functions:
      - Enable (default): `pyppbox.enable_terminal_log()` or export/set `PYPPBOX_DISABLE_TERMINAL_LOG=0` in terminal
      - Disable: `pyppbox.disable_terminal_log()` or export/set `PYPPBOX_DISABLE_TERMINAL_LOG=1` in terminal
    - Add enviroment variable adaptation for all GUI modules and related functions
    - Add more logging [functions](https://rathaumons.github.io/pyppbox/pyppbox/utils.html#module-pyppbox.utils.logtools)
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.12.0](https://github.com/rathaumons/pyppbox/tree/v3.12.0) - ~~The Last V3~~

  - Add comprehensive type hints across the major codebase
  - Add `getMainConfig(current=True)` to retrieve current main modules
  - Improve multithreading core (`pyppbox.ppb.mt.MT`)
  - Overhaul configuration system including helper renames:
    - `getListCFGDoc()` -> `getCFGDictList()`
    - `loadListDocument()` -> `loadDocumentList()`
    - `dumpListDocDict()` -> `dumpDocDictList()`
  - Fix and enhance the main tools and utilities:
    - Fix `ResIO.addPeople()` to correctly store metadata
    - Add `GTIO.map_dict` for O(1) video → GT lookups
    - Enhance `GTIO.getGTFileName()` by using the new  `GTIO.map_dict`
    - Enhance visualization tools (`visualizetools.py`)
  - Restructure GitHub tests:
    - Move `.githubtest/` -> `.github/test/`
    - Update the related test .py files
    - Update CI workflows (Linux/macOS/Windows)
  - Update copyrights to 2025
  - Update and improve documentation
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.11.0](https://github.com/rathaumons/pyppbox/tree/v3.11.0) - ~~The Last V3~~

  - Remove the unused `line_width` from `YOLO_Ultralytics` configuration
  - Set `yolov8s-pose.pt` as default for `YOLO_Ultralytics` detector
  - Optimize the default parameters of the detectors for better detection
  - Set `pyppbox-data` [v1.3.0](https://github.com/rathaumons/pyppbox-data/releases/tag/v1.3.0) as default in documentation and workflows
  - Clean up redundant direct dependencies and related modules
  - Update all related workflows, GUI, example, and config files
  - Update and improve documentation
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.10.0](https://github.com/rathaumons/pyppbox/tree/v3.10.0) - ~~The Last V3~~

  - Add miscellaneous `misc` options to all dump functions in `ResIO`
  - Remove the legacy License classifier for [PEP 639](https://peps.python.org/pep-0639/) compliance
  - Detect and dodge fake CUDA setups in OpenCV
  - Change `setuptools` minimum version to v67.8.0
  - Fix most if not all typos
  - Update and improve `GETSTARTED.md`
  - Update and improve GitHub workflows
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.9.0](https://github.com/rathaumons/pyppbox/tree/v3.9.0) - ~~The Last V3~~

  - Fix `trainReIDClassifier()` for default input
  - Add an example of retraining all internal reider classifiers for GTA V dataset
  - Rearrange some examples
  - Update and improve documentation
  - Optimize GitHub workflows
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.8b1](https://github.com/rathaumons/pyppbox/tree/v3.8b1) - ~~The Last V3~~

  - Add `show_footnote` option to `visualizePeople()`
  - Optimize and enhance general performance
  - Correct typos and improve documentation
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.7b1](https://github.com/rathaumons/pyppbox/tree/v3.7b1) - ~~The Last V3~~

  - Sync `pyppbox.utils.persontools.Person`'s misc across frames in all trackers
  - Improve and add `--use-numid` for using number id in MOT as the real id in `pyppbox` format
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.6b9](https://github.com/rathaumons/pyppbox/tree/v3.6b9) - ~~The Last V3~~

  - Fix `setMainTracker()` and `setMainReIDer()` for default input
  - Correct `show_ids` documentation in `visualizePeople()`
  - Correct an error message in `visualizePeople()`
  - Update and improve documentation
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.6b8](https://github.com/rathaumons/pyppbox/tree/v3.6b8) - ~~The Last V3~~

  - Fix FaceNet alignment issue in data preparation for training
  - Update and improve documentation
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.6b7](https://github.com/rathaumons/pyppbox/tree/v3.6b7) - ~~The Last V3~~

  - Improve examples
  - Improve requirements
  - Update and improve `GETSTARTED.md`
  - Update and improve documentation
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.6b6](https://github.com/rathaumons/pyppbox/tree/v3.6b6) - ~~The Last V3~~

  - Add proper supports for YOLOv9, YOLOv10, and YOLOv11
  - Update requirements and documentation
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.6b5](https://github.com/rathaumons/pyppbox/tree/v3.6b5)- ~~The Last V3~~

  - Fix file filter in Torchreid GUI
  - Switch from [`pyppbox-ultralytics`](https://github.com/rathaumons/ultralytics-for-pyppbox) to [`vsensebox-ultralytics`](https://github.com/numediart/ultralytics-for-vsensebox)
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.6b4](https://github.com/rathaumons/pyppbox/tree/v3.6b4) - ~~The Last V3~~

  - Integrate [SFPS](https://github.com/rathaROG/smooth-fps) for better FPS calculation
  - Update GitHub workflows
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.6b3](https://github.com/rathaumons/pyppbox/tree/v3.6b3) - ~~The Last V3~~

  - Fix typo and update GitHub workflows
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.6b2](https://github.com/rathaumons/pyppbox/tree/v3.6b2)

  - Add a warning for the changes in the coming major version 4
  - Update documentation
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.6b1](https://github.com/rathaumons/pyppbox/tree/v3.6b1)

  - Replace `pyunpack` & `patool` with `shutil`
  - Improve Example 9 - example_09_eva_offline.py
  - Add `pyppbox.gui.guitools` to the documentation
  - Fix `useInternalConfigDir()` in `pyppbox.gui.guitools`
  - Fix sphinx-build warning for utils.rst
  - Fix and improve documentation
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.5b2](https://github.com/rathaumons/pyppbox/tree/v3.5b2)

  - Increase default random ID range in evatools
  - Add exception to `generateStaticID()` in `pyppbox.utils.evatools.TKOReider`
  - Update documentation
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.5b1](https://github.com/rathaumons/pyppbox/tree/v3.5b1)

  - Add Python 3.12 support
  - Change GUI Demo title
  - Update documentation
  - Update and improve GitHub workflows
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.4b2](https://github.com/rathaumons/pyppbox/tree/v3.4b2)

  - Add freedom of input video without GT (Ground-truth) in GUI demo
  - Update GitHub workflows
  - Update and improve documentation
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.4b1](https://github.com/rathaumons/pyppbox/tree/v3.4b1)

  - Remove `findRepspointList()` from `pyppbox.utils.persontools`
  - Add `findRepspointBB()` and `findRepspointUP()` to `pyppbox.utils.persontools`
  - Add float support for the private `convertStringToNPL()` of `pyppbox.utils.gttools`
  - Add `min_width_filter` support for `detectPeople()` of `ppbox.standalone`
  - Add alternative repspoint support for `detectPeople()` of `ppbox.standalone` and all detectors
  - Add [MOT Challenge ground-truth](https://motchallenge.net/instructions/) converter -> `pyppbox.utils.mot2pyppbox`
  - Improve documentation
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.3b1](https://github.com/rathaumons/pyppbox/tree/v3.3b1)

  - Remove `ontracked` and its related methods from `pyppbox.utils.persontools.Person`
  - Add `misc` to `pyppbox.utils.persontools.Person`
  - Improve documentation
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.2b2](https://github.com/rathaumons/pyppbox/tree/v3.2b2)

  - Add a hotfix for issue [#21](https://github.com/rathaumons/pyppbox/issues/21)
  - Improve documentation
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.2b1](https://github.com/rathaumons/pyppbox/tree/v3.2b1)

  - Update some default configurations
  - Update configurations/GUI/examples for `pyppbox-ultralytics>=8.0.218`
  - Update GitHub tests
  - Update GitHub workflows
  - Update requirements
  - Update and improve `GETSTARTED.md`
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.1b5](https://github.com/rathaumons/pyppbox/tree/v3.1b5)

  - Fix minor visaul bug
  - Add `install_req_py3_cuda121.cmd` for CUDA 12.1
  - Improve documentation
  - Drop support for Python 3.8
  - **Known issue/limitation**:
    - [Issue] YOLO Ultralytics: May throw `CUDA error: an illegal memory access was encountered` in multithread application

* `pyppbox` [v3.1b4](https://github.com/rathaumons/pyppbox/tree/v3.1b4)

  - Improve support for new `pyppbox-ultralytics`
  - Improve documentation
  - **Known issue/limitation**:
    - [Issue] YOLO Ultralytics: May throw `CUDA error: an illegal memory access was encountered` in multithread application

* `pyppbox` [v3.1b3](https://github.com/rathaumons/pyppbox/tree/v3.1b3)

  - Update and improve visualizetools for new `pyppbox-ultralytics`
  - Update requirements
  - Improve documentation
  - **Known issue/limitation**:
    - [Issue] YOLO Ultralytics: May throw `CUDA error: an illegal memory access was encountered` in multithread application

* `pyppbox` [v3.1b2](https://github.com/rathaumons/pyppbox/tree/v3.1b2)

  - Improve all supported trackers
  - Improve evatools
  - Change some default configurations
  - Update and improve examples
  - Improve documentation
  - **Known issue/limitation**:
    - [Issue] YOLO Ultralytics: May throw `CUDA error: an illegal memory access was encountered` in multithread application

* `pyppbox` [v3.1b1](https://github.com/rathaumons/pyppbox/tree/v3.1b1)

  - Add multithreading support for standalone -> `ppbox.standalone.mt.MT`
  - Add multithreading example -> See example 13
  - Add CPU support for `Torchreid`
  - Implement install requirements/dependencies -> See setup.py
  - Simplify and improve requirements
  - Update default config files
  - Update and improve documentation
  - Update and improve GUI
  - **Known issue/limitation**:
    - [Issue] YOLO Ultralytics: May throw `CUDA error: an illegal memory access was encountered` in multithread application

* `pyppbox` [v3.0b5](https://github.com/rathaumons/pyppbox/tree/v3.0b5)

  - Fix a critical bug in GUI of FaceNet which can cause missing `train_data` configuration
  - Fix and improve documentation
  - **Known issue/limitation**:
    - [Issue] YOLO Ultralytics: May throw `CUDA error: an illegal memory access was encountered` in multithread application

* `pyppbox` [v3.0b4](https://github.com/rathaumons/pyppbox/tree/v3.0b4)

  - Add hotfix for command `python` in `subprocess` when running on Linux
  - Update and improve dependencies/requirements for macOS and Linux
  - Update and improve `GETSTARTED.md` for macOS and Linux
  - Add core stability test for macOS
  - **Known issue/limitation**:
    - [Issue] YOLO Ultralytics: May throw `CUDA error: an illegal memory access was encountered` in multithread application

* `pyppbox` [v3.0b3](https://github.com/rathaumons/pyppbox/tree/v3.0b3)

  - Fix GUI for GT
  - Improve supports for Python [3.8-3.11]
  - Update and improve dependencies/requirements
  - Update and improve `GETSTARTED.md`
  - Update test workflows for Python [3.8-3.11]
  - **Known issue/limitation**:
    - [Issue] YOLO Ultralytics: May throw `CUDA error: an illegal memory access was encountered` in multithread application

* `pyppbox` [v3.0b2](https://github.com/rathaumons/pyppbox/tree/v3.0b2)

  - Add Linux and macOS supports
  - Add workflow for Windows/Linux core stability tests
  - Add workflow for PyPI build -> `pyppbox` is now available on PyPI
  - Improve independency of the modules
  - Improve supports for CPU-Only
  - Improve setup quality
  - Improve GUI stability
  - Update and normalize dependencies/requirements
  - Update and improve documentation
  - **Known issue/limitation**:
    - [Issue] YOLO Ultralytics: May throw `CUDA error: an illegal memory access was encountered` in multithread application

* `pyppbox` [v3.0b1](https://github.com/rathaumons/pyppbox/tree/v3.0b1)

  - Introduce new structure of `pyppbox` -> Cleaner and simpler
  - Introduce [`pyppbox-data`](https://github.com/rathaumons/pyppbox-data) and [`pyppbox-data-gta5`](https://github.com/rathaumons/PoseTReID_DATASET#-introducing-pyppbox-data-gta5) -> Size of `pyppbox` is now 99.9% smaller, easier to build and easier to install independently, with the freedom of choice for the modules you need
  - Introduce new standard `Person` class for `pyppbox` -> `pyppbox.utils.persontools.Person`
  - Introduce standalone functions for easy detect, track, and reid people -> `pyppbox.standalone` to replace `PManager`
  - Introduce new powerful and dynamic ***configuration classes***, `pyppbox.config` -> No more separated internal GLOBAL/LOCAL .py files
  - Introduce new supported configuration formats -> All YAML/JSON, raw string, ready dictionary, and file
  - Introduce standalone visual function `visualizePeople()` for easy visualize people -> `pyppbox.utils.visualizetools.visualizePeople`
  - Introduce standalone function `trainReIDClassifier()` for easy train classifier of the supported reiders -> `pyppbox.standalone.trainReIDClassifier`
  - Introduce new result I/O class `ResIO` for easy create new ground-truth and dump result into text file -> `pyppbox.utils.restools.ResIO`
  - Introduce new evaluation class `MyEVA`, and instead of represented point, bounding box is now used for matching and comparing the result -> `pyppbox.utils.evatools.MyEVA`
  - standalone comparison function `compareRes2Ref()` for supported datasets -> `pyppbox.utils.evatools.compareRes2Ref`
  - Introduce new ground truth tools `GTIO` and `GTInterpreter` for supported datasets -> `pyppbox.utils.gttools`
  - Introduce new internal logging -> `pyppbox.logtools`
  - Introduce new in-code `numpydoc` documentation ***methods/functions/classes***
  - Introduce new online `Sphinx` documentation -> [https://rathaumons.github.io/pyppbox](https://rathaumons.github.io/pyppbox)
  - Remove unnecessary import from submodules and their `__init__.py` files
  - Remove `input_video` and `force_hd` from main configurations, and other unused parameters from other configuration files -> New defaults `{pyppbox root}/config/cfg`
  - Add and update [examples](https://github.com/rathaumons/pyppbox/tree/main/examples) for `pyppbox` V3+ -> No longer compatible with older versions of `pyppbox`
  - Update and improve all supported modules
  - Update and improve GUI
  - Update ***requirements***
  - **Known issue/limitation**:
    - [Issue] YOLO Ultralytics: May throw `CUDA error: an illegal memory access was encountered` in multithread application


## **pyppbox V2 - Hello Ultralytics YOLOv8**

* `pyppbox` [v2.0b2](https://github.com/rathaumons/pyppbox/tree/v2.0b2)

  - Fix person's keypoint issue in `PManager` when using YOLO Ultralytics with pose estimation model
  - Add support for the new keypoint data format of YOLO Ultralytics's pose estimation model
  - Remove unnecessary imports
  - Update requirements
  - **Known issue/limitation**:
    - [Issue] YOLO Ultralytics: May throw `CUDA error: an illegal memory access was encountered` in multithread application

* `pyppbox` [v2.0b1](https://github.com/rathaumons/pyppbox/tree/v2.0b1)

  - Integrate PyTorch Ultralytics YOLOv8 -> `pyppbox-ultralytics` as package name
  - ***`torchreid` for pyppbox*** is changed to `pyppbox-torchreid` as package name
  - ***`opencv-contrib-python` for pyppbox*** is changed to `pyppbox-opencv` as package name
  - Enhance `Person` class & add `keypoints` attribute
  - Improve & generalize the names of some ***methods/functions/classes***
  - Enhance `PManager` & introduce `__` for private ***classes/methods***
  - Improve all related ***configurators*** & change some default configurations
  - Introduce ***lite*** & ***full*** edition of the extra models/weights
  - Update ***requirements*** & drop supports for ***Python 3.9 & CUDA 11.6/11.7***
  - Update the `FacNet` & `Torchreid` pretrained classifier ***PKLs*** for ***GTA5 dataset***
  - Update examples
  - Change to `pypa/build` for `setup.py`
  - **Known issue/limitation**:
    - [Issue] YOLO Ultralytics: May throw `CUDA error: an illegal memory access was encountered` in multithread application


## **pyppbox V1.1 - Multithreading Ready**

* `pyppbox` [v1.1b5](https://github.com/rathaumons/pyppbox/tree/v1.1b5)

  - Add support for **CUDA 11.8** -  Ready for RTX 4000 series :)

* `pyppbox` [v1.1b4](https://github.com/rathaumons/pyppbox/tree/v1.1b4)

  - Correct minor mistakes in default configuration related files and classes
  - Improve `setup.py` and change from `pip` to `bdist_wheel`
  - Update requirements - February 2023 update
  - Update `DeepSORT` for `numpy==1.24.2`
  - Clean up unused codes and improve some README.md files

* `pyppbox` [v1.1b3](https://github.com/rathaumons/pyppbox/tree/v1.1b3)

  - Fix a bug in UI demo `uidemo.py` when a given input video file does not exist in `GT` dictionary
  - Check pull [#6](https://github.com/rathaumons/pyppbox/pull/6) for more details

* `pyppbox` [v1.1b2](https://github.com/rathaumons/pyppbox/tree/v1.1b2)

  - Fix minor bugs of `SORT` module and `PManager()` in a rare/simulated scenario when a person appears and disappears from frame to frame
  - Check pull [#5](https://github.com/rathaumons/pyppbox/pull/5) for more details

* `pyppbox` [v1.1b1](https://github.com/rathaumons/pyppbox/tree/v1.1b1)

  - `PManager()` now has the default `__init__(enableEval=False, localConfig=False)`
  - The `enableEval=False` means all related `EVA` classes are disabled -> Check `example_advanced.py`
  - The `localConfig=False` means PManager uses the **GLOBAL** `cfg` dir inside the `pyppbox` package
  - When `localConfig=True`, you must call `setLocalConfig(local_cfg_dir)` in order to set your new **LOCAL** `cfg` dir
  - The idea of **GLOBAL** & **LOCAL** enables `pyppbox` to be used in multi-threading without interfering with the **GLOBAL** `cfg` dir
  - Your **LOCAL** `cfg` dir requires 4 configuration files: **`main.yaml`**, **`detectors.yaml`**, **`trackers.yaml`**, and **`reiders.yaml`**
  - Make sure the all input files such as pre-trained weights/models and others exist according to your **LOCAL** `cfg`
  - Check the **LOCAL** `cfg` and `example_local_cfg.py` as an example
  - Check pull [#4](https://github.com/rathaumons/pyppbox/pull/4) for more details


## **pyppbox V1**

* The GitHub repo was reinitiated from version 1.0b9 where the complete history is available here [33da563](https://github.com/rathaumons/pyppbox/tree/33da56302d27204931337b44d9a6a5adc1eb5257)

* [`OpenPose`](https://github.com/CMU-Perceptual-Computing-Lab/openpose) submodule was removed due to [its complicated license](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE) -> Check this repo [`pyppbox-paper`](https://github.com/rathaumons/pyppbox-paper) if you need `OpenPose` or need to reproduce the results in the paper

