from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck

BottleneckBlock = lambda x: Bottleneck(x, x // 4)


class ResidualConvUnit(nn.Module):
    """Residual Conv Unit block."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.conv2 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class FeatureFusionBlock(nn.Module):
    """Fuse two features and upsample by factor 2."""
    def __init__(self, channels: int, expand: bool = False,
                 align_corners: bool = True):
        super().__init__()
        self.out_channels = channels // 2 if expand else channels
        self.align_corners = align_corners

        self.res_conv_unit1 = ResidualConvUnit(channels)
        self.res_conv_unit2 = ResidualConvUnit(channels)
        self.projection = nn.Conv2d(channels, self.out_channels, 1, bias=True)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        x = inputs[0]
        if len(inputs) == 2:
            res = inputs[1]
            if x.shape[2:] != res.shape[2:]:
                res = F.interpolate(
                    res, size=x.shape[2:], mode="bilinear",
                    align_corners=self.align_corners,
                )
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x)
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear",
            align_corners=self.align_corners,
        )
        return self.projection(x)


class DPTHead(nn.Module):
    """DPT decoder head supporting 3 feature maps: x4, x8, x16."""
    def __init__(
        self,
        in_channels: list[int],
        post_process_channels: list[int],
        channels: int = 256,
        backbone_type: Literal["convnext", "vit"] = "vit",
        expand_channels: bool = False,
    ):
        super().__init__()
        assert len(in_channels) == 3, \
            "Expected 3 feature maps: x4, x8, x16"

        self.in_channels = in_channels
        self.post_process_channels = [
            c * (2**i) if expand_channels else c
            for i, c in enumerate(post_process_channels)
        ]
        self.channels = channels
        self.backbone_type = backbone_type

        # Reassemble blocks for x4, x8, x16
        self.reassemble_4 = self.make_reassemble_block(0)
        self.reassemble_8 = self.make_reassemble_block(1)
        self.reassemble_16 = self.make_reassemble_block(2)

        # Fusion hierarchy (top-down)
        self.fusion_16 = FeatureFusionBlock(channels)
        self.fusion_8 = FeatureFusionBlock(channels)
        self.fusion_4 = FeatureFusionBlock(channels)

        # Output projection
        self.output_conv = nn.Sequential(
            BottleneckBlock(channels),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def make_reassemble_block(self, idx: int) -> nn.Sequential:
        """Create reassemble block for x4, x8, x16."""
        block = nn.Sequential(
            nn.BatchNorm2d(self.in_channels[idx]),
            nn.Conv2d(
                self.in_channels[idx], self.post_process_channels[idx], 1
            ),
            nn.Identity(),
            nn.Conv2d(
                self.post_process_channels[idx], self.channels, 3, padding=1
            ),
            nn.BatchNorm2d(self.channels),
            nn.GELU(),
        )

        if self.backbone_type == "vit":
            if idx == 0:  # x4
                block[2] = nn.ConvTranspose2d(
                    self.post_process_channels[0],
                    self.post_process_channels[0],
                    kernel_size=4,
                    stride=4,
                )
            elif idx == 1:  # x8
                block[2] = nn.ConvTranspose2d(
                    self.post_process_channels[1],
                    self.post_process_channels[1],
                    kernel_size=2,
                    stride=2,
                )
            elif idx == 2:  # x16
                block[2] = nn.Identity()
        return block

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through DPT head."""
        x4, x8, x16 = inputs

        x4 = self.reassemble_4(x4)
        x8 = self.reassemble_8(x8)
        x16 = self.reassemble_16(x16)

        # Top-down fusion: 16 → 8 → 4
        p16 = self.fusion_16(x16)
        p8 = self.fusion_8(p16, x8)
        p4 = self.fusion_4(p8, x4)

        return self.output_conv(p4)


if __name__ == "__main__":
    model = DPTHead(
        in_channels=[384, 384, 384],
        post_process_channels=[48, 96, 192],
        channels=256,
        backbone_type="vit",
    ).cuda()

    x4 = torch.randn(1, 384, 128, 128).cuda()
    x8 = torch.randn(1, 384, 64, 64).cuda()
    x16 = torch.randn(1, 384, 32, 32).cuda()

    out = model([x4, x8, x16])
    print(out.shape)
