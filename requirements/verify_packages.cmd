:: Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


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
pip install --upgrade "pyppbox-ultralytics>=8.4.144"
echo.
echo ################################################################
echo #                            Done!                             #
echo ################################################################
echo.
