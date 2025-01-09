# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                           #
#   pyppbox: Toolbox for people detecting, tracking, and re-identifying.    #
#   Copyright (C) 2022 UMONS-Numediart                                      #
#                                                                           #
#   This program is free software: you can redistribute it and/or modify    #
#   it under the terms of the GNU General Public License as published by    #
#   the Free Software Foundation, either version 3 of the License, or       #
#   (at your option) any later version.                                     #
#                                                                           #
#   This program is distributed in the hope that it will be useful,         #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#   GNU General Public License for more details.                            #
#                                                                           #
#   You should have received a copy of the GNU General Public License       #
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.  #
#                                                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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
