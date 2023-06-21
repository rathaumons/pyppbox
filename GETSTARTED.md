# 🚀 Getting Started

🆕 `pyppbox` V2+ uses [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox) & [`pyppbox-torchreid`](https://github.com/rathaumons/torchreid-for-pyppbox):

* OpenCV is widely used in many well-known packages, but the majority of the prebuilt WHLs on the Internet including the official one on PyPi do not include GPU support. Thus, we build our custom one which includes CUDA & cuDNN supports for the DNN modules. In order to well distinguish from the rest, we decided to build and change the package name from `opencv-contrib-python` to [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox). [[repo]](https://github.com/rathaumons/opencv-for-pyppbox) [[WHL]](https://github.com/rathaumons/pyppbox-custpkg/tree/main/pyppbox_opencv)

* Similar to [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox), our custom `torchreid` is changed to [`pyppbox-torchreid`](https://github.com/rathaumons/torchreid-for-pyppbox). More than the normal package rename, the module name is also changed from `torchreid` to `pyppbox_torchreid` which means the `import` in the code must be also changed. [[WHL]](https://github.com/rathaumons/torchreid-for-pyppbox)

⚠️ ***Other things you need to know***:

* For the required custom package like OpenCV with DNN, we currently support only ***Windows***; however, our *pyppbox V3+* should also be able to run on other OS platforms as long as you can manage to install all the native requirements.

* All requirements are not strictly limited. You can install only the ones for the modules you need. For example, our `YOLO_Classic` (With `.weights` model) relies OpenCV DNN in order to make use of GPU power, and if it is your case, **DO NOT INSTALL** other OpenCV such as `opencv-python` or `opencv-contrib-python` along side our [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox) unless you known what you are doing. If you wish to run on other version of ***Python/CUDA/cuDNN***, you must rebuild the [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox) or `opencv_contrib_python` with DNN by yourself.


## ⚙️ Requirements

* Prerequisite: 
  - NVIDIA [CUDA Toolkit 11.8.x](https://developer.nvidia.com/cuda-downloads) with default installation path
  - NVIDIA [cuDNN 8.9.x](https://developer.nvidia.com/rdp/cudnn-download) with default installation path
  - Python [3.10.x for Windows](https://www.python.org/downloads/windows/)
  - Local pyppbox: `git clone https://github.com/rathaumons/pyppbox.git`

* (Optional) If you prefer conda:
  - Python 3.10 `conda create --name pyppbox_env python=3.10`

* Run the installer in `pyppbox/requirements/`: 
  - Python 3.10 & CUDA 11.8.x `install_req_p310_cuda118.cmd`

* Verify the requirements -> Simply run the `testme.cmd`
  - If there is no error, then you are all good and ready to go.
  - If `cv2` encounters `ImportError: DLL load failed ...`, please verify the path of your CUDA & CUDNN. Our pre-built `pyppbox-opencv` uses the default path of CUDA & CUDNN (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v1x.x`), and if your CUDA & CUDNN were installed in a different location, you do not need to rebuild the `pyppbox-opencv`, simply modify the `YOUR_PYTHON\Lib\site-packages\cv2\config.py` accordingly.


## 💽 Setup

* Install `pyppbox`
  - Download the latest from [releases](https://github.com/rathaumons/pyppbox/releases)
  - Or install directly, e.g. `v3.0b1`:
    ```
    pip install https://github.com/rathaumons/pyppbox/releases/download/v3.0b1/pyppbox-3.0b1-py3-none-win_amd64.whl
    ```
  - Or build one yourself:
    ```
      cd pyppbox
      create_whl.cmd
      cd dist
      pip install pyppbox-xxx.whl
    ```

* Install [`pyppbox-data`](https://github.com/rathaumons/pyppbox-data/)
  - Download the latest from [releases](https://github.com/rathaumons/pyppbox-data/releases)
  - Or install the ones you neeed directly:
    ```
    # Detectors
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.0/pyppbox_data_yolocls-1.0-py3-none-any.whl
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.0/pyppbox_data_yoloult-1.0-py3-none-any.whl
    # Trackers
    pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.0/pyppbox_data_deepsort-1.0-py3-none-any.whl
    # ReIDers
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

