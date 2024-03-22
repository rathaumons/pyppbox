# Release Notes 

## **pyppbox V3 - Make Simpler and Faster**

* `pyppbox` [v3.5b1](https://github.com/rathaumons/pyppbox/tree/v3.5b1)

  - Add Python 3.12 support
  - Change GUI Demo title
  - Update documentations
  - Update and improve GitHub workflows
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.4b2](https://github.com/rathaumons/pyppbox/tree/v3.4b2)

  - Add freedom of input video without GT (Ground-truth) in GUI demo
  - Update GitHub workflows
  - Update and improve documentations
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.4b1](https://github.com/rathaumons/pyppbox/tree/v3.4b1)

  - Remove `findRepspointList()` from `pyppbox.utils.persontools`
  - Add `findRepspointBB()` and `findRepspointUP()` to `pyppbox.utils.persontools`
  - Add float support for the private `convertStringToNPL()` of `pyppbox.utils.gttools`
  - Add `min_width_filter` support for `detectPeople()` of `ppbox.standalone`
  - Add alternative repspoint support for `detectPeople()` of `ppbox.standalone` and all detectors
  - Add [MOT Challenge ground-truth](https://motchallenge.net/instructions/) converter -> `pyppbox.utils.mot2pyppbox`
  - Improve documentations
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.3b1](https://github.com/rathaumons/pyppbox/tree/v3.3b1)

  - Remove `ontracked` and its related methods from `pyppbox.utils.persontools.Person`
  - Add `misc` to `pyppbox.utils.persontools.Person`
  - Improve documentations
  - **Known issue/limitation**:
    - You tell me :)

* `pyppbox` [v3.2b2](https://github.com/rathaumons/pyppbox/tree/v3.2b2)

  - Add a hotfix for issue [#21](https://github.com/rathaumons/pyppbox/issues/21)
  - Improve documentations
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
  - Improve documentations
  - Drop support for Python 3.8
  - **Known issue/limitation**:
    - [Issue] YOLO Ultralytics: May throw `CUDA error: an illegal memory access was encountered` in multithread application

* `pyppbox` [v3.1b4](https://github.com/rathaumons/pyppbox/tree/v3.1b4)

  - Improve support for new `pyppbox-ultralytics`
  - Improve documentations
  - **Known issue/limitation**:
    - [Issue] YOLO Ultralytics: May throw `CUDA error: an illegal memory access was encountered` in multithread application

* `pyppbox` [v3.1b3](https://github.com/rathaumons/pyppbox/tree/v3.1b3)

  - Update and improve visualizetools for new `pyppbox-ultralytics`
  - Update requirements
  - Improve documentations
  - **Known issue/limitation**:
    - [Issue] YOLO Ultralytics: May throw `CUDA error: an illegal memory access was encountered` in multithread application

* `pyppbox` [v3.1b2](https://github.com/rathaumons/pyppbox/tree/v3.1b2)

  - Improve all supported trackers
  - Improve evatools
  - Change some default configurations
  - Update and improve examples
  - Improve documentations
  - **Known issue/limitation**:
    - [Issue] YOLO Ultralytics: May throw `CUDA error: an illegal memory access was encountered` in multithread application

* `pyppbox` [v3.1b1](https://github.com/rathaumons/pyppbox/tree/v3.1b1)

  - Add multithreading support for standalone -> `ppbox.standalone.mt.MT`
  - Add multithreading example -> See example 13
  - Add CPU support for `Torchreid`
  - Implement install requirements/dependencies -> See setup.py
  - Simplify and improve requirements
  - Update default config files
  - Update and improve documentations
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
  - Update and improve documentations
  - **Known issue/limitation**:
    - [Issue] YOLO Ultralytics: May throw `CUDA error: an illegal memory access was encountered` in multithread application

* `pyppbox` [v3.0b1](https://github.com/rathaumons/pyppbox/tree/v3.0b1)

  - Introduce new structure of `pyppbox` -> Cleaner and simplier
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
  - Introduce new in-code `numpydoc` documentation ***methods/funtions/classes***
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
  - Improve & generalize the names of some ***methods/funtions/classes***
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

