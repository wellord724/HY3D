import torch
import torch.nn as nn
from torch.nn import init

from . import pmath


class ToPoincare(nn.Module):
    def __init__(self, c, clip_r=None):
        super().__init__()
        self.c = c
        self.clip_r = clip_r

    def forward(self, x):
        if self.clip_r is not None:
            norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            x = x * torch.min(torch.ones_like(norm), self.clip_r / norm)
        return pmath.project(pmath.expmap0(x, c=self.c), c=self.c)


class MobiusLayer(nn.Module):
    def __init__(self, in_features, out_features, c):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        return pmath.mobius_matvec(self.weight, x, c=self.c)
