import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_factor: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_factor = upsample_factor

        if self.upsample_factor > 1:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_factor,
                mode="bilinear",
                align_corners=False,
            )
        else:
            self.upsample = nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.conv(x)


class DepthHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        min_depth: float = 0.1,
        max_depth: float = 80.0,
        upsample_factor: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.upsample_factor = upsample_factor
        
        if self.upsample_factor > 1:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_factor,
                mode="bilinear",
                align_corners=False,
            )
        else:
            self.upsample = nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        x = (F.sigmoid(x) * (self.max_depth - self.min_depth)) + self.min_depth 
        return x