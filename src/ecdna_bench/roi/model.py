"""
Residual U-Net for ROI prediction (River Summers; unchanged).

Four encoder stages, a bottleneck and a symmetric decoder. Each residual block
applies GroupNorm -> SiLU -> 3x3 conv twice, with a 1x1 projection shortcut when
the channel count changes; groups = the largest of 8, 4, 2, 1 dividing the
channel count. Upsampling is bilinear followed by 3x3 conv, GroupNorm and SiLU.
Inputs are zero-padded to a multiple of 16 px and cropped back afterwards.
``UNet(4)`` has 35,923,337 parameters. Parameter names match the released
checkpoint, which is a plain ``state_dict``.
"""
import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch=1, base=64):
        super().__init__()

        # --- ENCODER ---
        self.enc1 = self.ResBlock(in_ch, base)
        self.enc2 = self.ResBlock(base, base * 2)
        self.enc3 = self.ResBlock(base * 2, base * 4)
        self.enc4 = self.ResBlock(base * 4, base * 8)

        # --- POOL ---
        self.pool = nn.MaxPool2d(2)

        # --- BOTTLENECK ---
        self.bottleneck = self.ResBlock(base * 8, base * 16)

        # --- DECODER ---
        self.up4 = self.up_block(base * 16, base * 8)
        self.up3 = self.up_block(base * 8, base * 4)
        self.up2 = self.up_block(base * 4, base * 2)
        self.up1 = self.up_block(base * 2, base)

        # --- FUSION ---
        self.fuse4 = self.ResBlock(base * 16, base * 8)
        self.fuse3 = self.ResBlock(base * 8,  base * 4)
        self.fuse2 = self.ResBlock(base * 4,  base * 2)
        self.fuse1 = self.ResBlock(base * 2,  base)

        # --- OUT ---
        self.out = nn.Conv2d(base, out_ch, kernel_size=1)

    class ResBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()

            self.conv = nn.Sequential(
                nn.GroupNorm(UNet.get_groups(in_c), in_c),
                nn.SiLU(),
                nn.Conv2d(in_c, out_c, 3, padding=1),

                nn.GroupNorm(UNet.get_groups(out_c), out_c),
                nn.SiLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
            )

            self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

        def forward(self, x):
            return self.conv(x) + self.skip(x)

    def up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(self.get_groups(out_c), out_c),
            nn.SiLU()
        )

    def forward(self, x):
        # --- PADDING ---
        x, pad_h, pad_w = self.pad_to_multiple(x)

        # --- ENCODER ---
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        # --- BOTTLENECK ---
        x5 = self.bottleneck(self.pool(x4))

        # --- DECODER ---
        d4 = self.up4(x5)
        d4 = self.fuse4(torch.cat([d4, x4], dim=1))

        d3 = self.up3(d4)
        d3 = self.fuse3(torch.cat([d3, x3], dim=1))

        d2 = self.up2(d3)
        d2 = self.fuse2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)
        d1 = self.fuse1(torch.cat([d1, x1], dim=1))

        # --- OUT ---
        out = self.out(d1)
        out = self.unpad(out, pad_h, pad_w)

        return out

    @staticmethod
    def get_groups(channels):
        for g in [8, 4, 2, 1]:
            if channels % g == 0:
                return g

    @staticmethod
    def pad_to_multiple(x, multiple=16):
        h, w = x.shape[-2:]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        # Pad (left, right, top, bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, pad_h, pad_w

    @staticmethod
    def unpad(x, pad_h, pad_w):
        if pad_h > 0:
            x = x[:, :, :-pad_h, :]
        if pad_w > 0:
            x = x[:, :, :, :-pad_w]
        return x
