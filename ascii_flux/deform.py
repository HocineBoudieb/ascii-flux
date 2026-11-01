"""Pure PyTorch deformable convolution for the offline segmenter.

The AMD ROCm index has no torchvision build matching the installed torch, and
the CPU torchvision kernels cannot take GPU tensors. This implementation of
``torchvision.ops.deform_conv2d`` (Dai et al. 2017; Zhu et al. 2019 modulation)
uses bilinear ``grid_sample`` and is checked against torchvision on CPU.
"""
import torch
from torch.nn import functional as F


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def deform_conv2d(input, offset, weight, bias=None, stride=1, padding=0, dilation=1, mask=None):
    b, c, h, w = input.shape
    out_channels, c_per_group, kh, kw = weight.shape
    groups = c//c_per_group
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    k = kh*kw
    ho, wo = offset.shape[-2:]
    offset_groups = offset.shape[1]//(2*k)
    dtype = input.dtype
    offset = offset.float().view(b, offset_groups, k, 2, ho, wo)
    ky, kx = torch.meshgrid(torch.arange(kh, device=input.device), torch.arange(kw, device=input.device), indexing="ij")
    oy, ox = torch.meshgrid(torch.arange(ho, device=input.device), torch.arange(wo, device=input.device), indexing="ij")
    base_y = (oy*sh-ph)[None]+(ky.reshape(-1)*dh)[:, None, None]
    base_x = (ox*sw-pw)[None]+(kx.reshape(-1)*dw)[:, None, None]
    y = base_y[None, None]+offset[:, :, :, 0]
    x = base_x[None, None]+offset[:, :, :, 1]
    # align_corners=True maps pixel centers exactly; zero padding outside.
    grid = torch.stack((2*x/max(w-1, 1)-1, 2*y/max(h-1, 1)-1), -1)  # [B,G,K,Ho,Wo,2]
    grid = grid.view(b*offset_groups, k*ho, wo, 2)
    values = input.float().view(b*offset_groups, c//offset_groups, h, w)
    sampled = F.grid_sample(values, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled = sampled.view(b, offset_groups, c//offset_groups, k, ho, wo)
    if mask is not None:
        sampled = sampled*mask.float().view(b, offset_groups, 1, k, ho, wo)
    sampled = sampled.reshape(b, groups, c_per_group*k, ho*wo)
    weights = weight.float().view(groups, out_channels//groups, c_per_group*k)
    out = torch.einsum("bgip,goi->bgop", sampled, weights).reshape(b, out_channels, ho, wo)
    if bias is not None:
        out = out+bias.float()[None, :, None, None]
    return out.to(dtype)


def install():
    """Replace torchvision's operator before loading remote segmentation code."""
    import torchvision.ops
    torchvision.ops.deform_conv2d = deform_conv2d
