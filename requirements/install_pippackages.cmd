@echo off
setlocal
cd /d %~dp0
python -m pip install --upgrade pip
pip install numpy==1.22.3
pip install cust\opencv_contrib_python-4.5.5-cp39-cp39-win_amd64.whl
pip install -r pippackages.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install cust\torchreid-1.4.0-cp39-cp39-win_amd64.whl
pause