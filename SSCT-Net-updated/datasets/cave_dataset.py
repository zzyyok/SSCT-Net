import os, glob
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
import torch
import random

def load_mat(path, key=None):
    data = sio.loadmat(path)
    # try common variable names
    for k in ['hsi','img','data','arr','I']:
        if k in data:
            return np.array(data[k], dtype=np.float32)
    # otherwise return first array-like
    for k,v in data.items():
        if isinstance(v, (np.ndarray,)) and v.ndim>=3:
            return np.array(v, dtype=np.float32)
    raise ValueError('no image array found in .mat')

class CAVEDataset(Dataset):
    def __init__(self, root, mode='train', scale=4, patch_size=32):
        # expects subfolders HR_HSI, HR_MSI, LR_HSI with matching filenames
        self.root = root
        self.mode = mode
        self.scale = scale
        self.patch_size = patch_size
        hsi_dir = os.path.join(root, 'HR_HSI')
        msi_dir = os.path.join(root, 'HR_MSI')
        lr_dir  = os.path.join(root, 'LR_HSI')
        self.files = sorted(os.listdir(hsi_dir))
        self.hsi_paths = [os.path.join(hsi_dir, f) for f in self.files]
        self.msi_paths = [os.path.join(msi_dir, f) for f in self.files]
        self.lr_paths  = [os.path.join(lr_dir, f) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hr_hsi = load_mat(self.hsi_paths[idx])  # H x W x C or C x H x W
        hr_msi = load_mat(self.msi_paths[idx])
        lr_hsi = load_mat(self.lr_paths[idx])
        # normalize to [0,1]
        def to_ch_hw(arr):
            if arr.shape[0] < arr.shape[-1]:
                # assume H W C -> C H W
                arr = np.transpose(arr, (2,0,1))
            return arr
        hr_hsi = to_ch_hw(hr_hsi).astype(np.float32)
        hr_msi = to_ch_hw(hr_msi).astype(np.float32)
        lr_hsi = to_ch_hw(lr_hsi).astype(np.float32)
        # random crop on train
        if self.mode == 'train':
            c, h, w = lr_hsi.shape
            ph = min(self.patch_size, h)
            pw = min(self.patch_size, w)
            i = random.randint(0, h-ph)
            j = random.randint(0, w-pw)
            lr_hsi = lr_hsi[:, i:i+ph, j:j+pw]
            # correspondingly crop HR at scale
            hr_hsi = hr_hsi[:, i*self.scale:(i+ph)*self.scale, j*self.scale:(j+pw)*self.scale]
            hr_msi = hr_msi[:, i*self.scale:(i+ph)*self.scale, j*self.scale:(j+pw)*self.scale]
        # to tensor
        lr_hsi = torch.from_numpy(lr_hsi)
        hr_msi = torch.from_numpy(hr_msi)
        hr_hsi = torch.from_numpy(hr_hsi)
        return lr_hsi, hr_msi, hr_hsi
