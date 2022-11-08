# ⚙️ Requirements

## Introduction

* We currently support only **Windows** due to the incompatibility of some modules on other platforms with the latest hardware.

* The `pyppbox` requires `opencv_contrib_python` with `dnn` and customized `torchreid`, and we only provide WHLs for **Python 3.9/3.10** with **CUDA 11.6.x** & **CUDNN 8.6.0** (Defualt path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6`). 

  ***NOTE:** If you wish to build `pyppbox` on other version of Python/CUDA/CUDNN, you must build the corresponding required `opencv_contrib_python` with `dnn` and customized `torchreid` (Check out the [customized torchreid repo here](https://github.com/rathaumons/torchreid-for-pyppbox)).*

## Install requirements
* Optional: If you prefer conda, you can create a virtual conda enviroment of your choice:
  - Python 3.9 `conda create --name pyppbox_env python=3.9`
  - Python 3.10 `conda create --name pyppbox_env python=3.10`
* Run the installer: 
  - Python 3.9 `install_req_p39.cmd`
  - Python 3.10 `install_req_p310.cmd`
## Verify the requirements
* Simply run the `testme.cmd`
  - If there is no error, then you're all good and ready to go.
  - If `cv2` encounters `ImportError: DLL load failed ...`, please verify the path of your CUDA & CUDNN. Our pre-built `opencv_contrib_python` uses the default path of CUDA & CUDNN (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6`), and if your CUDA & CUDNN were installed in a different location, you don't have to rebuild the `opencv_contrib_python`, simply modify the `YOUR_PYTHON\Lib\site-packages\cv2\config.py` accordingly.
