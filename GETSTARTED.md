# 🚀 Getting Started

Installing `pyppbox` is very easy and straightforward. You can install it from [PyPI](https://pypi.org/project/pyppbox/) directly or use the prebuilt `.whl` files on [GitHub releases](https://github.com/rathaumons/pyppbox/releases) or install from GitHub directly or build it from source on your own machine. However, in order to get it work, you need to install all the necessary dependencies or requirements for the modules you need.

This checkout contains **pyppbox V4 development** on `dev`. The package version remains **3.15.0** until the release bump. Existing V3 projects can keep their configuration files; see the [V4 upgrade guide](https://rathaumons.github.io/pyppbox/pyppbox/v4.html) for compatibility and runtime changes.


## ⚙️ Requirements

`YOLO_Classic` uses [OpenCV DNN](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html) to load Darknet `.cfg`/`.weights` models and requires OpenCV 4.x. OpenCV 5 removed the Darknet importer, so pyppbox requires `opencv-contrib-python<5` to preserve existing models and configurations. For GPU (CUDA) support, use a compatible OpenCV 4.x build from source or our [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox); the official `opencv-contrib-python` wheels provide CPU support only.

* Prerequisite: 
  - Python [[3.9-3.13]](https://www.python.org/downloads/) (For ***macOS*** GUI troubleshooting, try Python 3.11)
  - Local pyppbox repo: `git clone https://github.com/rathaumons/pyppbox.git`

* Before you install dependencies/requirements:
  - For Linux, recommend changing `python3` to `python`: `sudo apt install python-is-python3`
  - If you prefer conda + Python [3.9-3.13]: `conda create --name pyppbox_env python=3.11`
    Then activate it with `conda activate pyppbox_env` before installing packages.
  - Upgrade `pip` and `setuptools`:
    ```
    python -m pip install --upgrade pip
    pip install "setuptools>=67.8.0"
    ```
  - Recommend removing the official `ultralytics`:
    ```
    pip uninstall -y ultralytics
    ```

* Install dependencies/requirements under [`pyppbox/requirements/`](https://github.com/rathaumons/pyppbox/tree/main/requirements):
  - For CPU-only on any platform, skip this and go straight to Setup section below.
  - For GPU (CUDA) on ***Windows***:
    - Run the `cmd` installer [`install_req_py3_cuda121.cmd`](https://github.com/rathaumons/pyppbox/blob/main/requirements/install_req_py3_cuda121.cmd) (Or [`install_req_py3_cuda.cmd`](https://github.com/rathaumons/pyppbox/blob/main/requirements/install_req_py3_cuda.cmd) for CUDA 11.8)
    - These scripts install CUDA-enabled **PyTorch**. FaceNet and DeepSORT use **TensorFlow**: native Windows GPU support ended with TensorFlow 2.10. For newer TensorFlow GPU builds, follow the [official Linux/WSL2 instructions](https://www.tensorflow.org/install/pip); installing CUDA-enabled PyTorch does not enable TensorFlow GPU support.
  - For GPU (CUDA) on ***Linux***:
    - Install the CUDA version of [TensorFlow](https://www.tensorflow.org/install) and [PyTorch](https://pytorch.org/get-started/locally/).
    - Install the [`requirements.txt`](https://github.com/rathaumons/pyppbox/blob/main/requirements/requirements.txt):
      ```
      python -m pip install -r requirements/requirements.txt
      ```
  - For GPU (CUDA) on ***macOS***:
    - Not available

* (Optional) For GPU-Only (CUDA) -> Verify the installed dependencies:
  - Execute the `test_gpu.py`
    - From the repository root on Windows -> `requirements\test_gpu.cmd`
    - From the repository root on Linux -> `python requirements/test_gpu.py`
  - If there is no error, then you are all good and ready to go.
  - For OpenCV, the official `opencv-contrib-python` (No GPU support) is set in the `requirements.txt` file. If you need GPU support, check our [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox) or build one from source by yourself.


## 💽 Setup

You need to install the main package which is `pyppbox` and the data for the modules you need `pyppbox-data-xxx`. If you want to have some fun with the demo on our [GTA_V_DATASET](https://github.com/rathaumons/PoseTReID_DATASET), you also need to install `pyppbox-data-gta5`.

* Install `pyppbox`
  - Use the latest `.whl` from [releases](https://github.com/rathaumons/pyppbox/releases) or install from [PyPI](https://pypi.org/project/pyppbox/):
    ```
    pip install pyppbox
    ``` 
  - Or install directly from GitHub:
    ```
    pip install git+https://github.com/rathaumons/pyppbox.git@dev
    ```
  - Or build from source:
    ```
    pip install setuptools wheel build PyYAML
    python -m build --wheel --skip-dependency-check --no-isolation
    ```
    Run these commands from the repository root, then install the generated wheel in `dist/` with `python -m pip install` followed by its path.

* (Optional, V4) Install RF-DETR on Python 3.10+ from this checkout:
  ```
  python -m pip install ".[rfdetr]"
  ```
  For a release wheel containing RF-DETR, install it with the `[rfdetr]` extra. Earlier V3 releases do not provide this extra. Select `RF_DETR` in the GUI or with `setMainDetector()`. Like YOLO, the default config uses GPU `device: 0` and a local model path: `data/modules/rf_detr/rf-detr-nano.pth` under the installed pyppbox package. Set `device: cpu` for CPU inference. Install the optional `pyppbox-data-rfdetr` wheel below to populate the standard paths, browse to a downloaded checkpoint, or set `model_file: ''` to use RF-DETR's automatic download/cache. Existing configurations work without migration.

  **RF-DETR weights:** [Nano](https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth), [Small](https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth), [Medium](https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth), [Large](https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth). Nano/Small/Medium downloads have the same server filename; save them separately as `rf-detr-nano.pth`, `rf-detr-small.pth`, and `rf-detr-medium.pth`. Keep Large as `rf-detr-large-2026.pth`. [Example 15](https://rathaumons.github.io/pyppbox/examples/example_15.html#downloading-pretrained-weights) lists filenames, destinations, offline setup, and custom class IDs. Roboflow's [pretrained aliases](https://docs.roboflow.com/models/pretrained-aliases#rf-detr) describe its Inference service; pyppbox loads these local PyTorch checkpoints through `rfdetr`.

* (Optional, V4) Install ArcFace and AdaFace from the `dev` checkout on Python 3.10+:
  ```
  python -m pip install ".[faces]"
  ```
  Earlier pyppbox releases do not include these adapters. Install either face data wheel below, then select `ArcFace` or `AdaFace` in the GUI or with `setMainReIDer()`. Each wheel supplies its own detector, embedding model, and identity classifier; CPU is the default. See [face recognizer setup](https://rathaumons.github.io/pyppbox/pyppbox/modules/reiders.html#arcface-and-adaface) for inference, training, CUDA configuration, and model terms.

* Install [`pyppbox-data-xxx`](https://github.com/rathaumons/pyppbox-data/)
  - V4 uses all eight **[v1.6.0](https://github.com/rathaumons/pyppbox-data/releases/tag/v1.6.0)** module data wheels. Install only the modules you need. Data wheels supply assets; install the `rfdetr` or `faces` runtime extra separately when using those adapters. The data version is independent of the pyppbox package version.
  - Download from the release above or install directly:
    ```
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.6.0/pyppbox_data_yolocls-1.6.0-py3-none-any.whl
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.6.0/pyppbox_data_yoloult-1.6.0-py3-none-any.whl
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.6.0/pyppbox_data_rfdetr-1.6.0-py3-none-any.whl
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.6.0/pyppbox_data_deepsort-1.6.0-py3-none-any.whl
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.6.0/pyppbox_data_facenet-1.6.0-py3-none-any.whl
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.6.0/pyppbox_data_torchreid-1.6.0-py3-none-any.whl
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.6.0/pyppbox_data_arcface-1.6.0-py3-none-any.whl
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.6.0/pyppbox_data_adaface-1.6.0-py3-none-any.whl
    ```
  - [Read more about the data packages](https://github.com/rathaumons/pyppbox-data/).

* Install [`pyppbox-data-gta5`](https://github.com/rathaumons/PoseTReID_DATASET#-introducing-pyppbox-data-gta5)
  - Download the latest from [releases](https://github.com/rathaumons/PoseTReID_DATASET/releases) or install directly:
    ```
    pip install https://github.com/numediart/PoseTReID_DATASET/releases/download/v2.0/pyppbox_data_gta5-2.0-py3-none-any.whl
    ```

* Quick Test
  - In your Python terminal:
    ```
    import pyppbox
    pyppbox.launchGUI()
    ```
    Now you should see the GUI demo like this screenshot:
    <img src="https://raw.githubusercontent.com/rathaROG/screenshot/master/pyppbox/pyppbox_gui.jpg">
  - For related GUI functions and other configurations, check the [Configurations page](https://rathaumons.github.io/pyppbox/pyppbox/config.html).
  - Check the [Examples page](https://rathaumons.github.io/pyppbox/examples.html) for some real coding!
  - Device settings:
    * RF-DETR, YOLO Ultralytics, Torchreid, ArcFace, and AdaFace have editable device dropdowns. Choose a preset or type a supported value such as `cuda:1`; presets reflect each backend's capabilities.
    * Select `cpu` when using a module without its GPU runtime. The bundled RF-DETR and YOLO Ultralytics defaults request GPU index `0`; Torchreid defaults to `cuda`, while ArcFace/AdaFace default to `cpu`.
    * YOLO Classic, FaceNet, and DeepSORT show a final read-only `device` row before Save: `auto by OpenCV` or `auto by TensorFlow`. FaceNet's `gpu_mem` controls GPU memory allocation, not CPU/GPU selection.
    * See [GUI and device selection](https://rathaumons.github.io/pyppbox/pyppbox/config.html#device-selection) for presets, typed values, automatic selection, and hardware/runtime requirements. Selecting a preset does not install a GPU backend.

* Troubleshooting
  - For ***macOS***, if the GUI does not work, you may try Python 3.11 as suggested in Prerequisite section above.
  - For ***Linux***, if the GUI does not work, you might need to install these:
    ```
    sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
    ```
  - For ***Ubuntu on WSL 2***, you need to install these:
    ```
    sudo apt-get install libgl1-mesa-glx xdg-utils libegl1
    ```

## 📢 FYI

### 1️⃣ Customized OpenCV

OpenCV is widely used in many well-known packages, but the majority of the prebuilt WHLs on the Internet including the official one on PyPI do not include GPU support. Thus, we build our custom one which includes NVIDIA [CUDA](https://developer.nvidia.com/cuda-downloads) & [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) supports for the [OpenCV DNN module](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html). In order to well distinguish from the rest, we decided to build and change the package name from `opencv-contrib-python` to [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox) -> [[Repo]](https://github.com/rathaumons/opencv-for-pyppbox) [[WHL]](https://github.com/rathaumons/opencv-for-pyppbox/releases)

### 2️⃣ Customized Torchreid

Similar to [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox), our custom `torchreid` is changed to [`pyppbox-torchreid`](https://github.com/rathaumons/torchreid-for-pyppbox). More than the normal package rename, the module name is also changed from `torchreid` to `pyppbox_torchreid` which means the `import` in the code must be also changed. Find out more why `pyppbox` needs the customized `pyppbox-torchreid` -> [[Repo]](https://github.com/rathaumons/torchreid-for-pyppbox) [[PyPI]](https://pypi.org/project/pyppbox-torchreid/)

### 3️⃣ Customized Ultralytics

The current [requirements file](https://github.com/rathaumons/pyppbox/blob/main/requirements/requirements.txt) installs [`pyppbox-ultralytics>=8.4.144`](https://github.com/rathaumons/ultralytics-for-pyppbox), which provides the `ultralytics` import used by pyppbox. Existing `YOLO_Ultralytics` configurations continue to work.

The official `ultralytics`, the previous `vsensebox-ultralytics`, and `pyppbox-ultralytics` distributions share that import namespace. When migrating an existing environment, remove all three before reinstalling the current fork:

```sh
python -m pip uninstall -y ultralytics vsensebox-ultralytics pyppbox-ultralytics
python -m pip install "pyppbox-ultralytics>=8.4.144"
```

On Windows, `requirements/verify_packages.cmd` performs this replacement. Fresh installations of this pyppbox version select the current fork automatically.
