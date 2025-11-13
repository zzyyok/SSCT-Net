import torch
from torch import nn
import torch.nn.functional as F
import math

# ----------------------
# Utilities
# ----------------------
def window_partition(x, window_size):
    # x: B, C, H, W -> windows: B*num_windows, C, window_size, window_size
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0,2,4,1,3,5).contiguous()  # B, nH, nW, C, ws, ws
    windows = x.view(-1, C, window_size, window_size)
    return windows

def window_reverse(windows, window_size, H, W):
    # windows: num_windows*B, C, ws, ws -> B, C, H, W
    B = int(windows.shape[0] // (H * W / window_size / window_size))
    C = windows.shape[1]
    nH = H // window_size
    nW = W // window_size
    x = windows.view(B, nH, nW, C, window_size, window_size)
    x = x.permute(0,3,1,4,2,5).contiguous()
    x = x.view(B, C, H, W)
    return x

# ----------------------
# Shallow modules (keep as before)
# ----------------------
class ShallowHSI(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.depth = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch)
        self.point = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.reduce(x)
        x = self.depth(x)
        x = self.point(x)
        return self.act(x)

class ShallowMSI(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv5 = nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(out_ch*2, out_ch), nn.ReLU(), nn.Linear(out_ch, out_ch), nn.Sigmoid())
        self.act = nn.GELU()
    def forward(self, x):
        a = self.conv3(x)
        b = self.conv5(x)
        cat = torch.cat([a,b], dim=1)
        w = self.pool(cat).view(cat.size(0), -1)
        w = self.fc(w).view(cat.size(0), -1, 1, 1)
        fused = cat[:, :a.size(1),:,:]*w + cat[:, a.size(1):,:,:]*(1-w)
        return self.act(fused)

# ----------------------
# Spectral sparse transformer: performs self-attention along spectral axis (channel dim)
# Groups channels into groups to keep complexity bounded.
# ----------------------
class SpectralSparseTransformer(nn.Module):
    def __init__(self, channels, embed_dim=None, num_heads=4, group_size=16, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.embed_dim = embed_dim or channels
        self.num_heads = num_heads
        self.group_size = group_size
        self.dropout = nn.Dropout(dropout)
        # linear projections for attention (we'll use nn.MultiheadAttention)
        self.proj_in = nn.Linear(1, self.embed_dim)  # placeholder, we'll reshape input appropriately
        # Use a single-head multihead attention applied per spatial location over spectral bands groups
        self.mha = nn.MultiheadAttention(self.embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.ff = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim*2), nn.GELU(), nn.Linear(self.embed_dim*2, self.embed_dim))
        self.norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        # x: B, C, H, W
        B, C, H, W = x.shape
        # reshape to (B*H*W, C) sequence along spectral axis
        x_perm = x.permute(0,2,3,1).contiguous().view(-1, C)  # N = B*H*W, C
        # pad channels to multiple of group_size
        g = self.group_size
        pad = (g - (C % g)) % g
        if pad > 0:
            x_perm = torch.cat([x_perm, x_perm.new_zeros(x_perm.size(0), pad)], dim=1)
        Cp = x_perm.size(1)
        xg = x_perm.view(-1, Cp // g, g)  # (N, num_groups, group_size)
        out_groups = []
        for gi in range(xg.size(1)):
            grp = xg[:, gi, :]  # (N, g)
            # treat sequence length = g, batch = N, embed_dim = 1 per token -> project to embed_dim
            # prepare tokens: (N, g, 1)
            tokens = grp.unsqueeze(-1)
            tokens = self.proj_in(tokens)  # (N, g, embed_dim)
            # apply MHA over spectral tokens
            att_out, _ = self.mha(tokens, tokens, tokens)  # (N, g, embed_dim)
            att_out = self.dropout(att_out)
            # residual + norm
            tokens2 = self.norm1(tokens + att_out)
            ff = self.ff(tokens2)
            tokens3 = self.norm2(tokens2 + ff)  # (N, g, embed_dim)
            # project back to scalar per band
            proj_back = tokens3.mean(-1)  # simple projection: mean across embed_dim -> (N, g)
            out_groups.append(proj_back)
        out = torch.cat(out_groups, dim=1)[:, :C]  # (N, C)
        out = out.view(B, H, W, C).permute(0,3,1,2).contiguous()
        return out

# ----------------------
# Swin-like components: window attention and basic Swin block (simplified but closer to true Swin)
# ----------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (num_windows*B, ws*ws, dim)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B_, num_heads, N, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2,-1))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B_, N, C)
        out = self.proj(out)
        return out

def get_window_coords(H, W, window_size):
    return (H // window_size, W // window_size)

class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=4, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))

    def forward(self, x):
        # x: B, C, H, W -> Swin expects (B, H, W, C)
        B, C, H, W = x.shape
        x_skip = x
        x = x.permute(0,2,3,1).contiguous()  # B,H,W,C
        # pad H,W to multiple of window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0,0,0,pad_r,0,pad_b))
        Hp, Wp = x.shape[1], x.shape[2]
        # cyclic shift if needed
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
        # partition windows
        x_windows = x.view(B, Hp//self.window_size, self.window_size, Wp//self.window_size, self.window_size, C)
        x_windows = x_windows.permute(0,1,3,2,4,5).contiguous().view(-1, self.window_size*self.window_size, C)
        # attention
        x_windows = self.norm1(x_windows)
        attn_windows = self.attn(x_windows)
        # merge windows
        attn_windows = attn_windows.view(B, Hp//self.window_size, Wp//self.window_size, self.window_size, self.window_size, C)
        x = attn_windows.permute(0,1,3,2,4,5).contiguous().view(B, Hp, Wp, C)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1,2))
        # remove padding
        x = x[:, :H, :W, :].contiguous()
        x = x + x_skip.permute(0,2,3,1)  # residual (note: shape align to H,W,C)
        x = self.mlp(self.norm2(x)).permute(0,3,1,2).contiguous()
        return x

# ----------------------
# Residual CNN (keep)
# ----------------------
class ResidualCNN(nn.Module):
    def __init__(self, ch, n=3):
        super().__init__()
        layers=[]
        for _ in range(n):
            layers += [nn.Conv2d(ch,ch,3,padding=1), nn.GELU(), nn.Conv2d(ch,ch,3,padding=1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return x + self.net(x)

# ----------------------
# CrossAttentionFusion (keep but add softmax temperature handling)
# ----------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.q1 = nn.Conv2d(ch, ch, 1)
        self.k1 = nn.Conv2d(ch, ch, 1)
        self.v1 = nn.Conv2d(ch, ch, 1)
        self.q2 = nn.Conv2d(ch, ch, 1)
        self.k2 = nn.Conv2d(ch, ch, 1)
        self.v2 = nn.Conv2d(ch, ch, 1)
        self.gate = nn.Sequential(nn.Conv2d(ch*2, ch, 1), nn.Sigmoid())
        self.proj = nn.Conv2d(ch, ch, 1)
    def forward(self, hsi, msi):
        Q1 = self.q1(hsi); K1 = self.k1(msi); V1 = self.v1(msi)
        Q2 = self.q2(msi); K2 = self.k2(hsi); V2 = self.v2(hsi)
        B,C,H,W = Q1.shape
        q1 = Q1.view(B,C,-1).permute(0,2,1); k1 = K1.view(B,C,-1)
        att1 = torch.softmax(torch.bmm(q1, k1)/ (C**0.5), dim=-1)
        v1 = V1.view(B,C,-1).permute(0,2,1)
        out1 = torch.bmm(att1, v1).permute(0,2,1).view(B,C,H,W)
        q2 = Q2.view(B,C,-1).permute(0,2,1); k2 = K2.view(B,C,-1)
        att2 = torch.softmax(torch.bmm(q2, k2)/ (C**0.5), dim=-1)
        v2 = V2.view(B,C,-1).permute(0,2,1)
        out2 = torch.bmm(att2, v2).permute(0,2,1).view(B,C,H,W)
        g = self.gate(torch.cat([hsi,msi], dim=1))
        fused = g * out1 + (1-g) * out2
        return self.proj(fused)

# ----------------------
# MRCB: Multi-Resolution Conv Block (approximation)
# - spatial branches: (3x1) and (1x3) separable convs
# - spectral branch: 1D conv over channels (treat channels as sequence)
# ----------------------
class MRCB(nn.Module):
    def __init__(self, ch):
        super().__init__()
        # spatial branches
        self.spatial_h = nn.Sequential(nn.Conv2d(ch, ch, (3,1), padding=(1,0)), nn.GELU(), nn.Conv2d(ch, ch, (1,3), padding=(0,1)))
        self.spatial_v = nn.Sequential(nn.Conv2d(ch, ch, (1,3), padding=(0,1)), nn.GELU(), nn.Conv2d(ch, ch, (3,1), padding=(1,0)))
        # spectral branch: Conv1d over channels
        self.spectral = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1)
        )
        self.fuse = nn.Conv2d(ch*3, ch, 1)
    def forward(self, x):
        # x: B,C,H,W
        sh = self.spatial_h(x)
        sv = self.spatial_v(x)
        B,C,H,W = x.shape
        sp = x.view(B, C, -1)  # B, C, H*W
        sp = self.spectral(sp)  # B, C, H*W
        sp = sp.view(B, C, H, W)
        cat = torch.cat([sh, sv, sp], dim=1)
        out = self.fuse(cat)
        return out + x
