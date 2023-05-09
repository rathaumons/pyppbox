# ⚙️ Requirements

We currently support only **Windows** due to the incompatibility of some modules on other platforms using the latest hardware.

## 🆕 `pyppbox` V2 uses [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox) & [`pyppbox-torchreid`](https://github.com/rathaumons/torchreid-for-pyppbox)
OpenCV is widely used in many well-known packages, but the majority of the prebuilt WHLs on the Internet including the official one on PyPi do not include GPU support. Thus, we build our custom one which includes CUDA & cudnn supports for the DNN modules. In order to well distinguish from the rest, we decided to change the package name from `opencv-contrib-python` to [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox), and this custom [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox) has the same functionality as the `opencv-contrib-python`.
* [**Prebuilt WHLs of `pyppbox-opencv` is available here!**](pyppbox_opencv)
* [Check the repo of `pyppbox-opencv` here](https://github.com/rathaumons/opencv-for-pyppbox)

Similar to [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox), our custom `torchreid` is changed to [`pyppbox-torchreid`](https://github.com/rathaumons/torchreid-for-pyppbox). More than the normal package rename, the module name is also changed from `torchreid` to `pyppbox_torchreid` which means the `import` in the code must be also change.
* [**Prebuilt WHL of `pyppbox-torchreid` is available on its repo here!**](https://github.com/rathaumons/torchreid-for-pyppbox)

## Install requirements
* We dropped supports for Python 3.9 & CUDA 11.6/11.7.
* Optional: If you prefer conda, you can create a virtual conda enviroment of your choice:
  - For Python 3.10 `conda create --name pyppbox_env python=3.10`
* Run the installer: 
  - For Python 3.10 & CUDA 11.8.x `install_req_p310_cuda118.cmd`

## Verify the requirements
* Simply run the `testme.cmd`
  - If there is no error, then you are all good and ready to go.
  - If `cv2` encounters `ImportError: DLL load failed ...`, please verify the path of your CUDA & CUDNN. Our pre-built `pyppbox-opencv` uses the default path of CUDA & CUDNN (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v1x.x`), and if your CUDA & CUDNN were installed in a different location, you do not need to rebuild the `pyppbox-opencv`, simply modify the `YOUR_PYTHON\Lib\site-packages\cv2\config.py` accordingly.

## Notes:
* To avoid unforeseen error, **DO NOT INSTALL** other OpenCV such as `opencv-python` or `opencv-contrib-python` along side our [`pyppbox-opencv`](https://github.com/rathaumons/opencv-for-pyppbox)
* If you wish to build `pyppbox` for other version of Python/CUDA/CUDNN, you must build the corresponding required `pyppbox-opencv` or `opencv_contrib_python` with `dnn`.
* All prebuilt WHLs for V1 are still available on the [`pyppbox-custpkg`](https://github.com/rathaumons/pyppbox-custpkg)
