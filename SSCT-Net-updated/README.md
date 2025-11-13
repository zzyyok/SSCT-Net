# SSCT-Net (Runnable Skeleton)
This repository contains a runnable, simplified implementation of the SSCT-Net project described by the user.
It is designed for the **CAVE** dataset and supports GPU training with PyTorch.

## Files
- main.py              : command-line entry (train / test)
- train.py             : training script
- test.py              : testing/evaluation script
- models/ssct_net.py   : model high-level wrapper
- models/blocks.py     : building blocks / simplified modules used by SSCT-Net
- datasets/cave_dataset.py : dataset loader (expects .mat files)
- utils/losses.py      : L1 + SSTV loss
- utils/metrics.py     : PSNR, SAM (basic) and helpers
- utils/train_log.py   : simple logging via matplotlib
- utils/utils.py       : utilities (save/load checkpoints, seed)

## Usage (example)
- Install requirements: `pip install torch torchvision numpy scipy matplotlib`
- Prepare CAVE dataset in `datasets/CAVE/` with subfolders `HR_HSI/`, `HR_MSI/`, `LR_HSI/` containing .mat files.
- Train:
    python main.py --mode train --data_root datasets/CAVE --epochs 10 --batch_size 4
- Test:
    python main.py --mode test --data_root datasets/CAVE --model_path checkpoints/best.pth

Notes:
- This is a **minimal, educational** runnable skeleton, not production-optimized.
- Some modules are simplified versions (e.g., Transformers are lightweight).
