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
python -m pip install --upgrade pip
pip install "setuptools>=67.2.0"
pip install "numpy>=1.24.2"
pip install https://github.com/rathaumons/pyppbox-custpkg/raw/main/py39/opencv_contrib_python-4.7.0+cu120-cp39-cp39-win_amd64.whl
pip install -r pippackages.txt
pip install torch==2.0.0+cu120 torchaudio==2.0.1+cu120 torchvision==0.15.1+cu120 --extra-index-url https://download.pytorch.org/whl/cu120
pip install https://github.com/rathaumons/pyppbox-custpkg/raw/main/py39/torchreid-1.4.0+pyppbox-cp39-cp39-win_amd64.whl
pause