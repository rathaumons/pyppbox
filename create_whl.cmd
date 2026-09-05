:: Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


@echo off
setlocal
cd /d %~dp0
set "PYTHONWARNINGS=ignore"
python -m pip install --upgrade pip
pip install "setuptools>=67.8.0"
pip install wheel build PyYAML
python -m build --wheel --skip-dependency-check --no-isolation
pause