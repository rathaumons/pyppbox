## Release Notes 

* pyppbox [v1.1b5](https://github.com/rathaumons/pyppbox/tree/v1.1b5) [9f119f8](https://github.com/rathaumons/pyppbox/commit/9f119f8b31ff49fef13f44619655a35afbc2c27b)
  - Add support for **CUDA 11.8** -  Ready for RTX 4000 series :)

* pyppbox [v1.1b4](https://github.com/rathaumons/pyppbox/tree/v1.1b4) [ba9b925](https://github.com/rathaumons/pyppbox/commit/ba9b925d838b2891240343b24de9d2ad9b8e63eb)
  - Correct minor mistakes in default configuration related files and objects
  - Improve `setup.py` and change from `pip` to `bdist_wheel`
  - Update requirements - February 2023 update
  - Update `DeepSORT` for `numpy==1.24.2`
  - Clean up unused codes and improve some README.md files

* pyppbox [v1.1b3](https://github.com/rathaumons/pyppbox/tree/v1.1b3) [a4dc5ea](https://github.com/rathaumons/pyppbox/commit/a4dc5eaf190db68b2e877f56827dc8a9d776ae33)
  - Fix a bug in UI demo `uidemo.py` when a given input video file does not exist in `GT` dictionary.
  - Check pull https://github.com/rathaumons/pyppbox/pull/6 for more details.

* pyppbox [v1.1b2](https://github.com/rathaumons/pyppbox/tree/v1.1b2) [da311c4](https://github.com/rathaumons/pyppbox/commit/da311c40aae5689d3516c43bcce57b2c5f5a10c2)
  - Fix minor bugs of `SORT` module and `PManager()` in a rare/simulated scenario when a person appears and disappears from frame to frame.
  - Check pull https://github.com/rathaumons/pyppbox/pull/5 for more details.

* pyppbox [v1.1b1](https://github.com/rathaumons/pyppbox/tree/v1.1b1) [bedb41f](https://github.com/rathaumons/pyppbox/commit/bedb41f5f755c4eb82e663a22f83728ed2145c5a)
  - `PManager()` now has the default `__init__(enableEval=False, localConfig=False)`.
  - The `enableEval=False` means all related `EVA` objects are disabled -> Check [`example_advanced.py`](example_advanced.py)
  - The `localConfig=False` means PManager uses the **GLOBAL** `cfg` dir inside the `pyppbox` package.
  - When `localConfig=True`, you must call `setLocalConfig(local_cfg_dir)` in order to set your new **LOCAL** `cfg` dir.
  - The idea of **GLOBAL** & **LOCAL** enables `pyppbox` to be used in multi-threading without interfering with the **GLOBAL** `cfg` dir.
  - Your **LOCAL** `cfg` dir requires 4 configuration files: **`main.yaml`**, **`detectors.yaml`**, **`trackers.yaml`**, and **`reiders.yaml`**.
  - Make sure the all input files such as pre-trained weights/models and others exist according to your **LOCAL** `cfg`.
  - Check the **LOCAL** [`cfg`](examples/cfg) and [`example_local_cfg.py`](examples/example_local_cfg.py) as an example.
  - Check pull https://github.com/rathaumons/pyppbox/pull/4 for more details.

