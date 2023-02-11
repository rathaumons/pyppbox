# 📗 Examples

* There are **important changes** from version **1.1**:
  - `PManager()` now has the default `__init__(enableEval=False, localConfig=False)`.
  - The `enableEval=False` means all related `EVA` objects are disabled -> Check [`example_advanced.py`](example_advanced.py)
  - The `localConfig=False` means PManager uses the **GLOBAL** `cfg` dir inside the `pyppbox` package.
  - When `localConfig=True`, you must call `setLocalConfig(local_cfg_dir)` in order to set your new **LOCAL** `cfg` dir.
  - The idea of **GLOBAL** & **LOCAL** enables `pyppbox` to be used in multi-threading without interfering with the **GLOBAL** `cfg` dir.
  - Your **LOCAL** `cfg` dir requires 4 configuration files: **`main.yaml`**, **`detectors.yaml`**, **`trackers.yaml`**, and **`reiders.yaml`**.
  - Make sure the all input files such as pre-trained weights/models and others exist according to your **LOCAL** `cfg`.
  - Check the **LOCAL** [`cfg`](cfg) and [`example_local_cfg.py`](example_local_cfg.py) as an example.
