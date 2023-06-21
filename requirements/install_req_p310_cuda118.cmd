::    pyppbox: Toolbox for people detecting, tracking, and re-identifying.
::    Copyright (C) 2022 UMONS-Numediart
::
::    This program is free software: you can redistribute it and/or modify
::    it under the terms of the GNU General Public License as published by
::    the Free Software Foundation, either version 3 of the License, or
::    (at your option) any later version.
::
::    This program is distributed in the hope that it will be useful,
::    but WITHOUT ANY WARRANTY; without even the implied warranty of
::    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
::    GNU General Public License for more details.
::
::    You should have received a copy of the GNU General Public License
::    along with this program.  If not, see <https://www.gnu.org/licenses/>.


@echo off
setlocal
cd /d %~dp0
:: Set URLs
set "pyppbox-opencv=https://github.com/rathaumons/pyppbox-custpkg/raw/main/pyppbox_opencv/cp310_cu118/pyppbox_opencv-4.7.0-cp310-none-win_amd64.whl"
set "pyppbox-torchreid=https://github.com/rathaumons/torchreid-for-pyppbox/releases/download/v1.4.0/pyppbox_torchreid-1.4.0-py3-none-any.whl"
set "pyppbox-ultralytics=https://github.com/rathaumons/ultralytics-for-pyppbox/releases/download/v8.0.119/pyppbox_ultralytics-8.0.119-py3-none-any.whl"
:: Upgrade & install basic packages
python -m pip install --upgrade pip
pip install "setuptools>=67.2.0"
:: Uinstall conflict packages
pip uninstall -y opencv-python
pip uninstall -y opencv-contrib-python
pip uninstall -y ultralytics
:: Install common packages
pip install "numpy>=1.24.3"
pip install %pyppbox-opencv%
pip install -r pippackages.txt
pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install %pyppbox-torchreid%
pip install %pyppbox-ultralytics%
cls
:: Make sure there is no conflict
:: :: :: :: :: call verify_packages.cmd
:: Show & save installed pip packages to installed_packages.txt
pip freeze
pip freeze > installed_packages.txt
pause