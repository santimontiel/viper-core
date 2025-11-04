from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from viper.modeling.components.timm_backbone import DinoBackbone
from viper.modeling.components.adapter_deimv2 import SpatialTuningAdapter
from viper.modeling.components.dpt import DPTHead
from viper.modeling.components.heads import SegmentationHead, DepthHead


class DinoV3FromTimm(nn.Module):

    def __init__(
        self,
        model_name: str = "small_plus",
        pretrained: bool = True,
        inplanes: int = 16,
    ) -> None:
        """Initialize the DinoV3FromTimm module.
        """
        super().__init__()
        self.backbone = DinoBackbone(
            in_channels=3,
            out_indices=(-3, -2, -1),
            model_name=model_name,
            pretrained=pretrained,
            freeze=True,
        )

        self.seg_adapter = SpatialTuningAdapter(
            hidden_dim=self.backbone.embed_dim, inplanes=inplanes
        )
        self.depth_adapter = SpatialTuningAdapter(
            hidden_dim=self.backbone.embed_dim, inplanes=inplanes
        )

        self.seg_decoder = DPTHead(
            in_channels=[self.backbone.embed_dim] * 3,
            post_process_channels=[256, 256, 256],
            channels=256,
            backbone_type="convnext",
        )
        self.depth_decoder = DPTHead(
            in_channels=[self.backbone.embed_dim] * 3,
            post_process_channels=[256, 256, 256],
            channels=256,
            backbone_type="convnext",
        )

        self.seg_head = SegmentationHead(
            in_channels=256,
            out_channels=19,
            upsample_factor=4,
        )
        self.depth_head = DepthHead(
            in_channels=256,
            min_depth=0.1,
            max_depth=80.0,
            upsample_factor=4,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        backbone_outputs = self.backbone(x)
        s2, s3, s4 = self.seg_adapter(x, backbone_outputs)
        d2, d3, d4 = self.depth_adapter(x, backbone_outputs)
        s0 = self.seg_decoder([s2, s3, s4])
        d0 = self.depth_decoder([d2, d3, d4])
        seg = self.seg_head(s0)
        depth = self.depth_head(d0)
        return {
            "seg": seg,
            "depth": depth,
            "feats": backbone_outputs[-1],
        }
    

if __name__ == "__main__":
    model = DinoV3FromTimm(
        model_name="small_plus",
        pretrained=True,
        inplanes=32,
    ).to("cuda")
    x = torch.randn(1, 3, 512, 1024).cuda()
    out = model(x)

    for k, v in out.items():
        print(f"{k}: {v.shape}")