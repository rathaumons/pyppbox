:: Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License


@echo off
python align_dataset_mtcnn.py "data/train" "data/train_aligned" --image_size 182 --margin 44 --gpu_memory_fraction 0.5
