import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.ssct_net import SSCTNet
from datasets.cave_dataset import CAVEDataset
from utils.losses import SSTV_Loss
from utils.metrics import compute_psnr, compute_sam
from utils.utils import save_checkpoint, set_seed
from utils.train_log import TrainLogger

def train(data_root='datasets/CAVE', epochs=50, batch_size=4, lr=1e-4, device=None):
    set_seed(42)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # dataset & dataloader
    train_ds = CAVEDataset(data_root, mode='train')
    val_ds = CAVEDataset(data_root, mode='val')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # model
    sample_item = train_ds[0]
    lr_hsi, hr_msi, hr_hsi = sample_item
    in_hsi = lr_hsi.shape[0]
    in_msi = hr_msi.shape[0]
    out_ch = hr_hsi.shape[0]
    model = SSCTNet(in_channels_hsi=in_hsi, in_channels_msi=in_msi, out_channels=out_ch).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = SSTV_Loss(lambda_tv=0.1)
    logger = TrainLogger('logs/train_log.png')

    best_psnr = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for i, (lr_hsi, hr_msi, hr_hsi) in enumerate(train_loader):
            lr_hsi = lr_hsi.to(device)
            hr_msi = hr_msi.to(device)
            hr_hsi = hr_hsi.to(device)
            pred = model(lr_hsi, hr_msi)
            loss = criterion(pred, hr_hsi)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        # validation
        model.eval()
        with torch.no_grad():
            psnr_sum = 0.0
            sam_sum = 0.0
            cnt = 0
            for lr_hsi, hr_msi, hr_hsi in val_loader:
                lr_hsi = lr_hsi.to(device); hr_msi = hr_msi.to(device); hr_hsi = hr_hsi.to(device)
                pred = model(lr_hsi, hr_msi)
                psnr = compute_psnr(pred, hr_hsi)
                sam = compute_sam(pred, hr_hsi)
                psnr_sum += psnr; sam_sum += sam; cnt += 1
            psnr_val = psnr_sum / max(cnt,1)
            sam_val = sam_sum / max(cnt,1)
        print(f'Epoch {epoch}/{epochs} loss={avg_loss:.4f} PSNR={psnr_val:.4f} SAM={sam_val:.4f}')
        logger.append(epoch, avg_loss, psnr_val, sam_val)
        # save best
        if psnr_val > best_psnr:
            best_psnr = psnr_val
            save_checkpoint({'epoch':epoch,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()}, 'checkpoints/best.pth')
    print('Training finished.')
