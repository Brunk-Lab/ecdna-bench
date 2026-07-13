"""
ecdna_bench.eccount.model
==========================
Compact U-Net for ecDNA soft localization map prediction.

Frozen architecture (from §2 of REWRITE_PLAN.md)
-------------------------------------------------
* 3 input → 1 output channels
* base_channels = 32  (4 encoder + 4 decoder stages)
* GroupNorm(num_groups=8)
* Bilinear upsampling (not transposed convolutions)
* No dropout (dropout_p = 0.0)
* **Exactly 7,849,601 trainable parameters**

The parameter count is a published result and MUST NOT drift.
A guard in ``build_model`` enforces this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ModelConfig", "build_model", "UNet", "infer_probs", "EXPECTED_PARAM_COUNT"]

NormType = Literal["group", "batch"]

# The exact trainable parameter count for the published ecCount model.
EXPECTED_PARAM_COUNT: int = 7_849_601


@dataclass
class ModelConfig:
    """Configuration for the ecCount U-Net.

    Defaults match the frozen published architecture exactly.
    """
    in_channels:  int      = 3
    out_channels: int      = 1
    base_channels: int     = 32
    bilinear:     bool     = True
    norm_type:    NormType = "group"
    num_groups:   int      = 8
    dropout_p:    float    = 0.0


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------

def _make_norm(num_channels: int, norm_type: NormType, num_groups: int) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "group":
        groups = min(num_groups, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"Unsupported norm_type: {norm_type!r}")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv→Norm→ReLU→Conv→Norm→ReLU (+ optional Dropout2d)."""

    def __init__(
        self,
        in_ch:      int,
        out_ch:     int,
        norm_type:  NormType = "group",
        num_groups: int      = 8,
        dropout_p:  float    = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            _make_norm(out_ch, norm_type, num_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _make_norm(out_ch, norm_type, num_groups),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """MaxPool(2) → ConvBlock."""

    def __init__(
        self, in_ch: int, out_ch: int,
        norm_type: NormType = "group", num_groups: int = 8, dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_ch, out_ch, norm_type=norm_type, num_groups=num_groups, dropout_p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Bilinear upsample (or ConvTranspose) → concat skip → ConvBlock."""

    def __init__(
        self,
        in_ch:      int,
        skip_ch:    int,
        out_ch:     int,
        bilinear:   bool     = True,
        norm_type:  NormType = "group",
        num_groups: int      = 8,
        dropout_p:  float    = 0.0,
    ) -> None:
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            up_out_ch = in_ch
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            up_out_ch = out_ch

        self.conv = ConvBlock(
            up_out_ch + skip_ch, out_ch,
            norm_type=norm_type, num_groups=num_groups, dropout_p=dropout_p,
        )

    @staticmethod
    def _resize_to_match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self._resize_to_match(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class OutConv(nn.Module):
    """1×1 projection to output logits."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """Canonical ecCount U-Net.

    Encoder:  inc → down1 → down2 → down3 → down4
    Decoder:  up1 ← up2 ← up3 ← up4 ← outc
    Skip connections from each encoder level to the corresponding decoder level.
    """

    def __init__(
        self,
        in_channels:  int      = 3,
        out_channels: int      = 1,
        base_channels: int     = 32,
        bilinear:     bool     = True,
        norm_type:    NormType = "group",
        num_groups:   int      = 8,
        dropout_p:    float    = 0.0,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # Encoder
        self.inc   = ConvBlock(in_channels, c1, norm_type, num_groups, dropout_p=0.0)
        self.down1 = DownBlock(c1, c2, norm_type, num_groups, dropout_p=0.0)
        self.down2 = DownBlock(c2, c3, norm_type, num_groups, dropout_p=0.0)
        self.down3 = DownBlock(c3, c4, norm_type, num_groups, dropout_p=0.0)
        self.down4 = DownBlock(c4, c5, norm_type, num_groups, dropout_p=dropout_p)

        # Decoder
        self.up1 = UpBlock(c5, c4, c4, bilinear, norm_type, num_groups, dropout_p=dropout_p)
        self.up2 = UpBlock(c4, c3, c3, bilinear, norm_type, num_groups, dropout_p=0.0)
        self.up3 = UpBlock(c3, c2, c2, bilinear, norm_type, num_groups, dropout_p=0.0)
        self.up4 = UpBlock(c2, c1, c1, bilinear, norm_type, num_groups, dropout_p=0.0)

        self.outc = OutConv(c1, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_model(cfg: ModelConfig) -> nn.Module:
    """Build the canonical ecCount U-Net from *cfg*.

    Raises
    ------
    AssertionError
        If the default config is used and the trainable parameter count
        does not match ``EXPECTED_PARAM_COUNT`` (7,849,601).
        This guard ensures the published architecture is preserved exactly.
    """
    model = UNet(
        in_channels   = cfg.in_channels,
        out_channels  = cfg.out_channels,
        base_channels = cfg.base_channels,
        bilinear      = cfg.bilinear,
        norm_type     = cfg.norm_type,
        num_groups    = cfg.num_groups,
        dropout_p     = cfg.dropout_p,
    )

    # Enforce frozen parameter count only when using the default (published) config
    if (
        cfg.in_channels == 3
        and cfg.out_channels == 1
        and cfg.base_channels == 32
        and cfg.bilinear is True
        and cfg.norm_type == "group"
        and cfg.num_groups == 8
        and cfg.dropout_p == 0.0
    ):
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n == EXPECTED_PARAM_COUNT, (
            f"Trainable parameter count mismatch: expected {EXPECTED_PARAM_COUNT}, got {n}. "
            "The architecture has drifted from the published ecCount model."
        )

    return model


@torch.no_grad()
def infer_probs(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Logits → sigmoid probabilities (inference convenience helper).

    Caller is responsible for ``model.eval()`` and device placement.
    """
    return torch.sigmoid(model(x))
