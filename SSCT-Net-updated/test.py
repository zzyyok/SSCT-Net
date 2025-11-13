import os
import torch
from torch.utils.data import DataLoader
from models.ssct_net import SSCTNet
from datasets.cave_dataset import CAVEDataset
from utils.metrics import compute_psnr, compute_sam
from utils.utils import set_seed

def test(data_root='datasets/CAVE', model_path='checkpoints/best.pth', device=None):
    set_seed(42)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    test_ds = CAVEDataset(data_root, mode='test')
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    sample = test_ds[0]
    in_hsi = sample[0].shape[0]; in_msi = sample[1].shape[0]; out_ch = sample[2].shape[0]
    model = SSCTNet(in_channels_hsi=in_hsi, in_channels_msi=in_msi, out_channels=out_ch).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    psnr_sum = 0.0; sam_sum = 0.0; cnt=0
    with torch.no_grad():
        for lr_hsi, hr_msi, hr_hsi in loader:
            lr_hsi = lr_hsi.to(device); hr_msi = hr_msi.to(device); hr_hsi = hr_hsi.to(device)
            pred = model(lr_hsi, hr_msi)
            psnr_sum += compute_psnr(pred, hr_hsi)
            sam_sum += compute_sam(pred, hr_hsi)
            cnt += 1
    print(f'Test results -> PSNR: {psnr_sum/max(1,cnt):.4f}, SAM: {sam_sum/max(1,cnt):.4f}')
