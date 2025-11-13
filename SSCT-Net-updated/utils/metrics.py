import torch
import numpy as np

def tensor_to_np(x):
    # B,C,H,W -> C,H,W (take first sample)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x[0]

def compute_psnr(pred, gt, max_val=1.0):
    p = tensor_to_np(pred); g = tensor_to_np(gt)
    mse = np.mean((p-g)**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val**2 / mse)

def compute_sam(pred, gt):
    # simplified SAM averaged over pixels
    p = tensor_to_np(pred); g = tensor_to_np(gt)
    # reshape C x H*W
    Cp, Hp, Wp = p.shape
    p = p.reshape(Cp, -1)
    g = g.reshape(Cp, -1)
    # avoid div0
    num = np.sum(p * g, axis=0)
    den = np.linalg.norm(p, axis=0) * np.linalg.norm(g, axis=0) + 1e-8
    cos = np.clip(num/den, -1, 1)
    ang = np.arccos(cos)
    return np.mean(ang)

# wrapper used in training/test (accept torch tensors)
def compute_psnr_torch(pred, gt):
    return compute_psnr(pred, gt)

def compute_sam_torch(pred, gt):
    return compute_sam(pred, gt)
# aliases
compute_sam = compute_sam_torch
compute_psnr = compute_psnr_torch
