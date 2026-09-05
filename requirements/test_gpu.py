# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License

import torch
print("########################## PyTorch ##########################")
print(f"GPU with CUDA = {torch.cuda.is_available()}")
print(f"Number of CUDA GPU = {torch.cuda.device_count()}")
import cv2
print("########################### OpenCV ##########################")
print(f"OpenCV = {cv2.__version__}")
print(f"Number of CUDA GPU = {cv2.cuda.getCudaEnabledDeviceCount()}")
import pyppbox_torchreid
print("######################### Torchreid #########################")
print(f"Torchreid = {pyppbox_torchreid.__version__}")
from pyppbox_torchreid.metrics.rank_cylib import rank_cy
print("Cython test finished -> Congrats if you don't see otherwise.")
print("######################## Ultralytics ########################")
import ultralytics
print(f"Ultralytics = {ultralytics.__version__}")
