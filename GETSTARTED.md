# 🚀 Getting Started

Installing `pyppbox` is very easy and straightforward. You can install it from [PyPI](https://pypi.org/project/pyppbox/) directly or use the prebuilt `.whl` files on [GitHub releases](https://github.com/rathaumons/pyppbox-data/releases) or install from GitHub directly or build it from source on your own machine. However, in order to get it work, you need to install all the necessary dependcies or requirements for the modules you needs.

## ⚙️ Requirements

All requirements are not strictly limited. However, some specific modules might need some special dependencies. For example, `YOLO_Classic` (With `.weights` model) relies OpenCV DNN in order to make use of GPU power. In this case, you might need to build OpenCV from source by yourself or use our [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox) instead of the official `opencv-contrib-python` which does not include GPU support.

* Prerequisite: 
  - For NVIDIA GPU: [CUDA Toolkit 11.8.x](https://developer.nvidia.com/cuda-downloads) with default installation path
  - For NVIDIA GPU: [cuDNN 8.9.x](https://developer.nvidia.com/rdp/cudnn-download) with default installation path
  - Python [3.10.x](https://www.python.org/downloads/)
  - Local pyppbox repo: `git clone https://github.com/rathaumons/pyppbox.git`

* Install dependencies/requirments: 
  - If you prefer conda (Python 3.8+): `conda create --name pyppbox_env python=3.10`
  - On Windows, run the `cmd` inside `pyppbox/requirements/`:
    - For GPU + (Python 3.10 & CUDA 11.8.x): `install_req_p310_cuda118.cmd` 
    - For CPU-only + Python 3.8+:  `install_req_p3x_cpu.cmd`
  - On Linux/macOS, under `pyppbox/requirements/`:
    - For GPU + (Python 3.10 & CUDA 11.8.x) + Linux-Only:
      ```
      pip uninstall -y ultralytics # Remove the official ultralytics
      python -m pip install --upgrade pip
      pip install "setuptools>=67.2.0"
      pip install -r pippackages_cuda.txt
      pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
      ```
    - For CPU-only + (Python 3.8+):
      ```
      pip uninstall -y ultralytics # Remove the official ultralytics
      python -m pip install --upgrade pip
      pip install "setuptools>=67.2.0"
      pip install -r pippackages_cpu.txt
      pip install torch torchvision torchaudio
      ```

* (Optional) For GPU-Only -> Verify the installed dependencies:
  - Execute the `test_gup.py`
    - On Windows -> `test_gpu.cmd`
    - On Linux/macOS -> `python test_gup.py`
  - If there is no error, then you are all good and ready to go.
  - For `pyppbox-opencv` on Windows, if `cv2` encounters `ImportError: DLL load failed ...`, please verify the path of your CUDA & cuDNN. Our pre-built `pyppbox-opencv` uses the default path of CUDA & cuDNN (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v1x.x`), and if your CUDA & cuDNN were installed in a different location, simply modify the `YOUR_PYTHON\Lib\site-packages\cv2\config.py` accordingly.

* ⚠️ ***Notes for CPU-Only:***
  - Torchreid does not work for CPU-Only -> It is excluded from `install_req_p3x_cpu.cmd`
  - YOLO Ultralytics uses GPU by default, you must set `cpu` as string for the parameter `device` in its configuration


## 💽 Setup

You need to install the main package which is `pyppbox` and the data for the modules you need `pyppbox-data-xxx`. If you want to have some fun for the demo on our [GTA_V_DATASET](https://github.com/rathaumons/PoseTReID_DATASET), you also need to install `pyppbox-data-gta5`.

* Install `pyppbox`
  - Use PyPI: `pip install pyppbox`
  - Or download the latest `.whl` from [releases](https://github.com/rathaumons/pyppbox/releases)
  - Or install directly from GitHub:
    ```
    pip install git+https://github.com/rathaumons/pyppbox.git
    ```
  - Or build from source:
    ```
    cd pyppbox
    python -m build --wheel --skip-dependency-check
    cd dist
    pip install pyppbox-xxx.whl
    ```

* Install [`pyppbox-data-xxx`](https://github.com/rathaumons/pyppbox-data/)
  - Download the latest from [releases](https://github.com/rathaumons/pyppbox-data/releases)
  - Or install the ones you need directly:
    ```
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.0/pyppbox_data_yolocls-1.0-py3-none-any.whl
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.0/pyppbox_data_yoloult-1.0-py3-none-any.whl
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.0/pyppbox_data_deepsort-1.0-py3-none-any.whl
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.0/pyppbox_data_facenet-1.0-py3-none-any.whl
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.0/pyppbox_data_torchreid-1.0-py3-none-any.whl
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


## 🪧 FYI

🆕 We added support for Linux and macOS from `pyppbox` `v3.0b2`, and instead of using `pyppbox-opencv`, the official `opencv-contrib-python` is used by default. You still can manually remove it and install our `pyppbox-opencv` if you need to.

OpenCV is widely used in many well-known packages, but the majority of the prebuilt WHLs on the Internet including the official one on PyPi do not include GPU support. Thus, we build our custom one which includes CUDA & cuDNN supports for the DNN modules. In order to well distinguish from the rest, we decided to build and change the package name from `opencv-contrib-python` to [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox). [[repo]](https://github.com/rathaumons/opencv-for-pyppbox) [[WHL]](https://github.com/rathaumons/pyppbox-custpkg/tree/main/pyppbox_opencv)

Similar to [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox), our custom `torchreid` is changed to [`pyppbox-torchreid`](https://github.com/rathaumons/torchreid-for-pyppbox). More than the normal package rename, the module name is also changed from `torchreid` to `pyppbox_torchreid` which means the `import` in the code must be also changed. [[WHL]](https://github.com/rathaumons/torchreid-for-pyppbox)

