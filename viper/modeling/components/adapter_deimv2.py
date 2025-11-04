import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTuningAdapter(nn.Module):
    # Modified from DEIMv2: https://github.com/Intellindust-AI-Lab/DEIMv2/blob/f6c59b7bb3e00861eade2a749f1dd737f9b9a1a8/engine/backbone/dinov3_adapter.py#L4

    def __init__(
        self,
        hidden_dim: int = 768,
        inplanes: int = 16,
    ) -> None:
        super().__init__()

        # Spatial Prior Module (SPM).
        # 1/2
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.GELU(),
            ]
        )
        # 1/4
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * inplanes),
            ]
        )
        # 1/8
        self.conv3 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )
        # 1/16
        self.conv4 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )

        # Linear projections to match the DINOv3 feature dimension.
        self.projs = nn.ModuleList()
        self.projs.append(nn.Sequential(
            nn.Conv2d(hidden_dim + (2 * inplanes), hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
        ))
        self.projs.append(nn.Sequential(
            nn.Conv2d(hidden_dim + (4 * inplanes), hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
        ))
        self.projs.append(nn.Sequential(
            nn.Conv2d(hidden_dim + (4 * inplanes), hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
        ))
    
    def forward(self, image, backbone_outputs):

        # Extract multi-scale features with Spatial Prior Module.
        a1 = self.stem(image)
        a2 = self.conv2(a1)     # 1/4
        a3 = self.conv3(a2)     # 1/8
        a4 = self.conv4(a3)     # 1/16

        # Fuse and project the features.
        outputs = []
        for c, a, proj in zip(backbone_outputs, [a2, a3, a4], self.projs):
            scale_factor = a.shape[-1] / c.shape[-1]
            c_reshaped = F.interpolate(
                c, scale_factor=scale_factor, mode="bilinear", align_corners=False
            )
            ca = torch.cat([c_reshaped, a], dim=1)
            c_proj = proj(ca)
            outputs.append(c_proj)

        return outputs

    