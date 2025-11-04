from typing import Any

import timm
import torch.nn as nn


class DinoBackbone(nn.Module):
    """A wrapper class for the DINOv3 family of backbones hosted on timm.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_indices: tuple = (-3, -2, -1),
        model_name: str = "small_plus",
        pretrained: bool = True,
        freeze: bool = True,
    ):
        """Initialize the DINO backbone.

        Args:
            model_name (str): The name of the DINO model to load from timm.
            pretrained (bool): Whether to load pretrained weights.
        """
        super().__init__()
        self.model_name = model_name
        self.out_indices = out_indices
        self.pretrained = pretrained

        match model_name:
            case "small":
                model_name = "timm/vit_small_patch16_dinov3.lvd1689m"
            case "small_plus":
                model_name = "timm/vit_small_plus_patch16_dinov3.lvd1689m"
            case "base":
                model_name = "timm/vit_base_patch16_dinov3.lvd1689m"
            case "large":
                model_name = "timm/vit_large_patch16_dinov3.lvd1689m"
            case "convnext_tiny":
                model_name = "timm/convnext_tiny.dinov3_lvd1689m"
            case "convnext_small":
                model_name = "timm/convnext_small.dinov3_lvd1689m"
            case "convnext_base":
                model_name = "timm/convnext_base.dinov3_lvd1689m"
            case "convnext_large":
                model_name = "timm/convnext_large.dinov3_lvd1689m"
            case _:
                raise ValueError(
                    f"Unknown model_name '{self.model_name}'. "
                    "Supported values are: 'small', 'small_plus', 'base', 'large', "
                    "'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'."
                )

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_channels
        )
        self.embed_dim = self.model.feature_info.channels()[-1]

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: Any) -> Any:
        """Forward pass through the DINO backbone.

        Args:
            x (Any): Input tensor.

        Returns:
            Any: Output features from the backbone.
        """
        return self.model(x)
    

if __name__ == "__main__":

    import torch

    # Test single backbone forward pass.
    for model in [
        "small",
        "small_plus",
        "base",
        "large",
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_large",
    ]:
        model = DinoBackbone(model_name=model, pretrained=True)
        dummy_input = torch.randn(1, 3, 512, 1024)
        output = model(dummy_input)
        print(f"Model: {model.model_name}, Output feature shapes: {[o.shape for o in output]}")

 