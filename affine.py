import torch
from torch import nn
from torch.nn import functional as F

import einops


class ChannelwiseAffineTransform2d(nn.Module):
    def __init__(
            self,
            num_channels,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.grid = nn.Sequential(
            nn.Conv2d(
                num_channels + 2,  # +2 for pos_encoding
                num_channels,
                kernel_size=3,
                padding="same",
            ),
            nn.Mish(),
            nn.Conv2d(
                num_channels,
                num_channels * 2,  # x2 for (x,y) tuples of grid_sample
                kernel_size=3,
                padding="same",
            )
        )
        self.initialize()

    def initialize(self):
        pass

    def create_identity_grid(self, x):
        *_, h, w = x.size()
        identity_matrix = torch.eye(2, 3, device=x.device).unsqueeze(0)
        identity_grid = F.affine_grid(identity_matrix, (1, 1, h, w), align_corners=True)  # shape: (1, h, w, 2)
        return identity_grid

    def create_pos_encoding(self, x):
        b, _, h, w = x.size()
        y_grid = einops.repeat(
            torch.linspace(-1, 1, h, device=x.device),
            "h -> b 1 h w", b=b, w=w
        )
        x_grid = einops.repeat(
            torch.linspace(-1, 1, w, device=x.device),
            "w -> b 1 h w", b=b, h=h
        )
        pos_encoding = torch.cat([y_grid, x_grid], dim=1)  # shape: (b, 2, h, w)
        return pos_encoding

    def forward(self, x):
        b, c, h, w = x.size()
        pos = self.create_pos_encoding(x)
        grid = self.grid(torch.cat([x, pos], 1))
        grid = einops.rearrange(grid, "b (c g) h w -> (b c) h w g", g=2)  # shape: (b * c, h, w, 2)
        grid += self.create_identity_grid(x)
        x = einops.rearrange(x, "b c h w -> (b c) 1 h w ")
        x = F.grid_sample(x, grid, align_corners=True)
        x = einops.rearrange(x, "(b c) 1 h w -> b c h w", b=b, c=c)
        return x
