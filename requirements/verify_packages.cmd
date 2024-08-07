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
echo.
:: Make sure there is no conflict for 'pyppbox-ultralytics'
echo ################################################################
echo #   Make sure there is no conflict for 'pyppbox-ultralytics'   #
echo ################################################################
echo.
pip uninstall -y pyppbox-ultralytics
pip uninstall -y vsensebox-ultralytics
pip uninstall -y ultralytics
pip install --upgrade --no-deps --force-reinstall vsensebox-ultralytics
echo.
echo ################################################################
echo #                            Done!                             #
echo ################################################################
echo.
