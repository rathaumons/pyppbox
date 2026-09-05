:: Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


@echo off
setlocal
cd /d %~dp0
:: Upgrade & install basic packages
python -m pip install --upgrade pip
pip install "setuptools>=67.8.0"
:: Uninstall conflict packages
pip uninstall -y ultralytics
:: Install common packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cls
:: Make sure there is no conflict
call verify_packages.cmd
:: Show & save installed pip packages to installed_requirements.txt
pip freeze
pip freeze > installed_requirements.txt
pause
