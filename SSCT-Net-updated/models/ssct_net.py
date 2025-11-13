import torch
from torch import nn
from .blocks import ShallowHSI, ShallowMSI, SpectralSparseTransformer, ResidualCNN, SwinBlock, CrossAttentionFusion, MRCB

class SSCTNet(nn.Module):
    def __init__(self, in_channels_hsi=31, in_channels_msi=3, out_channels=31, base_channels=64, swin_window=8):
        super().__init__()
        # shallow
        self.hsi_shallow = ShallowHSI(in_channels_hsi, base_channels)
        self.msi_shallow = ShallowMSI(in_channels_msi, base_channels)
        # deep enhancers: spectral transformer and swin blocks
        self.hsi_spec = SpectralSparseTransformer(base_channels, embed_dim=base_channels, num_heads=4, group_size=16)
        self.hsi_res = ResidualCNN(base_channels, n=3)
        # Swin blocks: two blocks (non-shifted and shifted)
        self.msi_swin1 = SwinBlock(dim=base_channels, input_resolution=(None,None), num_heads=4, window_size=swin_window, shift_size=0)
        self.msi_swin2 = SwinBlock(dim=base_channels, input_resolution=(None,None), num_heads=4, window_size=swin_window, shift_size=swin_window//2)
        self.msi_res = ResidualCNN(base_channels, n=2)
        # fusion and reconstruction
        self.fusion = CrossAttentionFusion(base_channels)
        self.mrcb = MRCB(base_channels)
        self.up1 = nn.Sequential(nn.Conv2d(base_channels, base_channels*4, 3, padding=1), nn.PixelShuffle(2))
        self.up2 = nn.Sequential(nn.Conv2d(base_channels, base_channels*4, 3, padding=1), nn.PixelShuffle(2))
        self.final = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, lr_hsi, hr_msi):
        # lr_hsi: B, C_h, h, w  ; hr_msi: B, C_m, H, W
        # downsample msi to lr size for fusion processing
        msi_ds = nn.functional.interpolate(hr_msi, size=lr_hsi.shape[2:], mode='bilinear', align_corners=False)
        f_h = self.hsi_shallow(lr_hsi)
        f_m = self.msi_shallow(msi_ds)
        f_h = self.hsi_spec(f_h)
        f_h = self.hsi_res(f_h)
        f_m = self.msi_swin1(f_m)
        f_m = self.msi_swin2(f_m)
        f_m = self.msi_res(f_m)
        fused = self.fusion(f_h, f_m)
        # aggregate and refine with MRCB
        agg = fused + f_h + f_m
        agg = self.mrcb(agg)
        x = self.up1(agg)
        x = self.up2(x)
        out = self.final(x)
        return out
