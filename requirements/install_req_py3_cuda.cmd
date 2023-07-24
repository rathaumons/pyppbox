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
:: Upgrade & install basic packages
python -m pip install --upgrade pip
pip install "setuptools>=67.2.0"
:: Uninstall conflict packages
pip uninstall -y ultralytics
:: Install common packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
cls
:: Make sure there is no conflict
call verify_packages.cmd
:: Show & save installed pip packages to installed_requirements.txt
pip freeze
pip freeze > installed_requirements.txt
pause