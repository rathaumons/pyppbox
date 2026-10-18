:: Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


@echo off
setlocal
cd /d "%~dp0"
:: Upgrade & install basic packages
python -m pip install --upgrade pip
pip install "setuptools>=67.8.0"
:: Uninstall conflict packages
pip uninstall -y ultralytics insightface onnxruntime onnxruntime-gpu
:: Install common packages
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
:: Install V4 modules with CUDA 12.1-compatible runtimes
pip install -r rfdetr.txt -r faces_gpu.txt "transformers==5.17.0" "onnxruntime-gpu==1.20.2"
cls
:: Make sure there is no conflict
call verify_packages.cmd
:: Show & save installed pip packages to installed_requirements.txt
pip freeze
pip freeze > installed_requirements.txt
pause
