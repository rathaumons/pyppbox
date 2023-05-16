## Release Notes for **`pyppbox` V2**

* `pyppbox` [v2.0b1](https://github.com/rathaumons/pyppbox/tree/v2.0b1) <!-- - [-------](https://github.com/rathaumons/pyppbox/commit/-------) -->
  - Integrate PyTorch Ultralytics YOLOv8 -> `pyppbox-ultralytics` as package name
  - ***`torchreid` for pyppbox*** is changed to `pyppbox-torchreid` as package name
  - ***`opencv-contrib-python` for pyppbox*** is changed to `pyppbox-opencv` as package name
  - Enhance `Person` object & add `keypoints` attribute
  - Improve & generalize the names of some ***methods/funtions/objects***
  - Enhance `PManager()` & introduce `__` for private ***objects/methods***
  - Improve all related ***configurators*** & change some default configurations
  - Introduce ***lite*** & ***full*** edition of the extra models/weights
  - Update ***requirements*** & drop supports for ***Python 3.9 & CUDA 11.6/11.7***
  - Update the `FacNet` & `Torchreid` pretrained classifier ***PKLs*** for ***GTA5 dataset***
  - Update examples
  - Change to `pypa/build` for `setup.py`
  - **Known issue/limitation**: 
    - [Issue] In multithread application, Pytorch Ultralytics may throw `CUDA error: an illegal memory access was encountered`

## Release Notes for **`pyppbox` V1**

* `pyppbox` [v1.1b5](https://github.com/rathaumons/pyppbox/tree/v1.1b5) <!-- - [9f119f8](https://github.com/rathaumons/pyppbox/commit/9f119f8b31ff49fef13f44619655a35afbc2c27b) -->
  - Add support for **CUDA 11.8** -  Ready for RTX 4000 series :)

* `pyppbox` [v1.1b4](https://github.com/rathaumons/pyppbox/tree/v1.1b4) <!-- - [ba9b925](https://github.com/rathaumons/pyppbox/commit/ba9b925d838b2891240343b24de9d2ad9b8e63eb) -->
  - Correct minor mistakes in default configuration related files and objects
  - Improve `setup.py` and change from `pip` to `bdist_wheel`
  - Update requirements - February 2023 update
  - Update `DeepSORT` for `numpy==1.24.2`
  - Clean up unused codes and improve some README.md files

* `pyppbox` [v1.1b3](https://github.com/rathaumons/pyppbox/tree/v1.1b3) <!-- - [a4dc5ea](https://github.com/rathaumons/pyppbox/commit/a4dc5eaf190db68b2e877f56827dc8a9d776ae33) -->
  - Fix a bug in UI demo `uidemo.py` when a given input video file does not exist in `GT` dictionary.
  - Check pull https://github.com/rathaumons/pyppbox/pull/6 for more details.

* `pyppbox` [v1.1b2](https://github.com/rathaumons/pyppbox/tree/v1.1b2) <!-- - [da311c4](https://github.com/rathaumons/pyppbox/commit/da311c40aae5689d3516c43bcce57b2c5f5a10c2) -->
  - Fix minor bugs of `SORT` module and `PManager()` in a rare/simulated scenario when a person appears and disappears from frame to frame.
  - Check pull https://github.com/rathaumons/pyppbox/pull/5 for more details.

* `pyppbox` [v1.1b1](https://github.com/rathaumons/pyppbox/tree/v1.1b1) <!-- - [bedb41f](https://github.com/rathaumons/pyppbox/commit/bedb41f5f755c4eb82e663a22f83728ed2145c5a) -->
  - `PManager()` now has the default `__init__(enableEval=False, localConfig=False)`.
  - The `enableEval=False` means all related `EVA` objects are disabled -> Check [`example_advanced.py`](example_advanced.py)
  - The `localConfig=False` means PManager uses the **GLOBAL** `cfg` dir inside the `pyppbox` package.
  - When `localConfig=True`, you must call `setLocalConfig(local_cfg_dir)` in order to set your new **LOCAL** `cfg` dir.
  - The idea of **GLOBAL** & **LOCAL** enables `pyppbox` to be used in multi-threading without interfering with the **GLOBAL** `cfg` dir.
  - Your **LOCAL** `cfg` dir requires 4 configuration files: **`main.yaml`**, **`detectors.yaml`**, **`trackers.yaml`**, and **`reiders.yaml`**.
  - Make sure the all input files such as pre-trained weights/models and others exist according to your **LOCAL** `cfg`.
  - Check the **LOCAL** [`cfg`](examples/cfg) and [`example_local_cfg.py`](examples/example_local_cfg.py) as an example.
  - Check pull https://github.com/rathaumons/pyppbox/pull/4 for more details.

* This repo was reinitiated from version 1.0b9 where the complete history is available here [33da563](https://github.com/rathaumons/pyppbox/tree/33da56302d27204931337b44d9a6a5adc1eb5257).

* [`OpenPose`](https://github.com/CMU-Perceptual-Computing-Lab/openpose) submodule was removed due to [its complicated license](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE) -> Check this repo [`pyppbox-paper`](https://github.com/rathaumons/pyppbox-paper) if you need `OpenPose` or need to reproduce the results in the paper.
