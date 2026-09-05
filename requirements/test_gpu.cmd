:: Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


@echo off
setlocal
cd /d %~dp0
nvidia-smi
python test_gpu.py
pause