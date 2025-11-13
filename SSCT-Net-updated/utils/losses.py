import torch
from torch import nn

def gradient(x, axis):
    # x: B,C,H,W or B,C,L,H,W for 3D variants
    if axis == 'h':
        return torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])
    if axis == 'w':
        return torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
    if axis == 'l':
        # treat channel dimension as spectral
        return torch.abs(x[:,1:,:,:] - x[:,:-1,:,:])

class SSTV_Loss(nn.Module):
    def __init__(self, lambda_tv=0.1):
        super().__init__()
        self.lambda_tv = lambda_tv

    def forward(self, output, target):
        l1 = torch.mean(torch.abs(output - target))
        gh = torch.mean(gradient(output, 'h'))
        gw = torch.mean(gradient(output, 'w'))
        gl = torch.mean(gradient(output, 'l'))
        sstv = gh + gw + gl
        return l1 + self.lambda_tv * sstv
