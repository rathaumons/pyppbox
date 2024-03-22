# 🚀 Getting Started

Installing `pyppbox` is very easy and straightforward. You can install it from [PyPI](https://pypi.org/project/pyppbox/) directly or use the prebuilt `.whl` files on [GitHub releases](https://github.com/rathaumons/pyppbox/releases) or install from GitHub directly or build it from source on your own machine. However, in order to get it work, you need to install all the necessary dependencies or requirements for the modules you needs.

🆕 `pyppbox` `v3.0b2+` supports all Windows, Linux and macOS. ***Use the most recent version of `pyppbox` for smoother experience!***

## ⚙️ Requirements

All requirements are not strictly limited. However, some specific modules might need some special dependencies. For example, `YOLO_Classic` (With `.weights` model) relies on [OpenCV DNN](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html) in order to make use of GPU power. In this case, you might need to build OpenCV from source by yourself or use our [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox) instead of the official `opencv-contrib-python` which does not include GPU support.

* Prerequisite: 
  - (Optional) For NVIDIA GPU: [CUDA Toolkit 11.x/12.x](https://developer.nvidia.com/cuda-downloads) with default installation path
  - (Optional) For NVIDIA GPU: [cuDNN 8.x.x](https://developer.nvidia.com/rdp/cudnn-download) with default installation path
  - Python [[3.9-3.12]](https://www.python.org/downloads/)
  - Local pyppbox repo: `git clone https://github.com/rathaumons/pyppbox.git`

* Before you install dependencies/requirements:
  - For Linux, recommend changing `python3` to `python`: `sudo apt install python-is-python3`
  - If you prefer conda + Python [3.9-3.12]: `conda create --name pyppbox_env python=3.11`
  - If you don't know whether to install only Tensorflow or PyTorch or both -> Check [Supported Modules](https://rathaumons.github.io/pyppbox/pyppbox/modules.html)
  - For Tensorflow with GPU support -> [See here](https://www.tensorflow.org/install/pip)

* Install dependencies/requirments under `pyppbox/requirements/`: 
  - On Windows, recommend using the `cmd` installer:
    - For GPU: `install_req_py3_cuda121.cmd` (Or `install_req_py3_cuda.cmd` for CUDA 11.8)
    - For CPU-only: `install_req_py3_cpu.cmd`
  - On Linux:
    - For GPU:
      ```
      python -m pip install --upgrade pip
      pip uninstall -y ultralytics # Remove the official ultralytics
      pip install "setuptools>=67.2.0"
      pip install torch torchvision torchaudio
      pip install -r requirements.txt
      ```
    - For CPU-only:
      ```
      python -m pip install --upgrade pip
      pip uninstall -y ultralytics # Remove the official ultralytics
      pip install "setuptools>=67.2.0"
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
      pip install -r requirements.txt
      ```
  - On macOS:
    - For GPU: Not available
    - For CPU:
      ```
      python -m pip install --upgrade pip
      pip uninstall -y ultralytics # Remove the official ultralytics
      pip install "setuptools>=67.2.0"
      pip install torch torchvision torchaudio
      pip install -r requirements.txt
      ```

* (Optional) For GPU-Only -> Verify the installed dependencies:
  - Execute the `test_gpu.py`
    - On Windows -> `test_gpu.cmd`
    - On Linux -> `python test_gpu.py`
  - If there is no error, then you are all good and ready to go.
  - For OpenCV, the official `opencv-contrib-python` (No GPU support) is set in the `requirements.txt` file. If you need GPU support, check our [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox) or build one from source by yourself.

* ⚠️ ***Notes:***
  - For CPU-Only, YOLO Ultralytics uses GPU by default, you must set `cpu` as string for the parameter `device` in its configuration.
  - For GPU on Windows, [Tensorflow 2.11+ no long provides native GPU support](https://www.tensorflow.org/install/pip#windows-native). 


## 💽 Setup

You need to install the main package which is `pyppbox` and the data for the modules you need `pyppbox-data-xxx`. If you want to have some fun for the demo on our [GTA_V_DATASET](https://github.com/rathaumons/PoseTReID_DATASET), you also need to install `pyppbox-data-gta5`.

* Install `pyppbox`
  - Use the latest `.whl` from [releases](https://github.com/rathaumons/pyppbox/releases) or install from [PyPI](https://pypi.org/project/pyppbox/):
    ```
    pip install pyppbox
    ``` 
  - Or install directly from GitHub:
    ```
    pip install git+https://github.com/rathaumons/pyppbox.git
    ```
  - Or build from source:
    ```
    pip install "setuptools>=67.2.0"
    pip install wheel build PyYAML
    python -m build --wheel --skip-dependency-check --no-isolation
    cd dist
    pip install pyppbox-xxx.whl
    ```

* Install [`pyppbox-data-xxx`](https://github.com/rathaumons/pyppbox-data/)
  - Download the latest from [releases](https://github.com/rathaumons/pyppbox-data/releases)
  - Or install the ones you need directly:
    ```
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.1.1/pyppbox_data_yolocls-1.1.1-py3-none-any.whl
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.1.1/pyppbox_data_yoloult-1.1.1-py3-none-any.whl
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.1.1/pyppbox_data_deepsort-1.1.1-py3-none-any.whl
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.1.1/pyppbox_data_facenet-1.1.1-py3-none-any.whl
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.1.1/pyppbox_data_torchreid-1.1.1-py3-none-any.whl
    ```

* Install [`pyppbox-data-gta5`](https://github.com/rathaumons/PoseTReID_DATASET#-introducing-pyppbox-data-gta5)
  - Download the latest from [releases](https://github.com/rathaumons/PoseTReID_DATASET/releases)
  - Or install directly:
    ```
    pip install https://github.com/rathaumons/PoseTReID_DATASET/releases/download/v2.0/pyppbox_data_gta5-2.0-py3-none-any.whl
    ```

* (Optional) Quick Test
  - On your terminal or CMD:
    ```
    python
    import pyppbox
    pyppbox.launchGUI()
    ```
  - Now you should see the GUI of pyppbox for easy demo.
    <details><summary><ins>Show GUI example!</ins></summary><img src="https://raw.githubusercontent.com/rathaROG/screenshot/master/pyppbox/pyppbox_gui.jpg"></details>
  - For ***Linux***, if the GUI does not work, you might need to install these:
    ```
    sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
    ```
  - For ***Ubuntu on WSL 2***, you need to install these:
    ```
    sudo apt-get install libgl1-mesa-glx
    sudo apt-get install xdg-utils
    sudo apt-get install libegl1
    ```

## 📢 FYI

### 1️⃣ Customized OpenCV

OpenCV is widely used in many well-known packages, but the majority of the prebuilt WHLs on the Internet including the official one on PyPI do not include GPU support. Thus, we build our custom one which includes NVIDIA [CUDA](https://developer.nvidia.com/cuda-downloads) & [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) supports for the [OpenCV DNN module](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html). In order to well distinguish from the rest, we decided to build and change the package name from `opencv-contrib-python` to [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox) -> [[Repo]](https://github.com/rathaumons/opencv-for-pyppbox) [[WHL]](https://github.com/rathaumons/opencv-for-pyppbox/releases)

### 2️⃣ Customized Torchreid

Similar to [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox), our custom `torchreid` is changed to [`pyppbox-torchreid`](https://github.com/rathaumons/torchreid-for-pyppbox). More than the normal package rename, the module name is also changed from `torchreid` to `pyppbox_torchreid` which means the `import` in the code must be also changed. Find out more why `pyppbox` needs the customized `pyppbox-torchreid` -> [[Repo]](https://github.com/rathaumons/torchreid-for-pyppbox) [[PyPI]](https://pypi.org/project/pyppbox-torchreid/)

### 3️⃣ Customized Ultralytics

Also, similar to `pyppbox_torchreid`, our custom `ultralytics` is changed to [`pyppbox-ultralytics`](https://github.com/rathaumons/ultralytics-for-pyppbox), but this time, the module name is still the same `ultralytics` and it is the main reason why the official `ultralytics` must be removed. Find out more why `pyppbox` needs the customized `pyppbox-ultralytics` -> [[Repo]](https://github.com/rathaumons/ultralytics-for-pyppbox) [[PyPI]](https://pypi.org/project/pyppbox-ultralytics/)
