import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from jaxtyping import Float



class MaskedMSELoss(nn.Module):
    """Masked Mean Squared Error Loss."""

    def forward(
        self,
        preds: Float[torch.Tensor, "b 1 h w"],
        targets: Float[torch.Tensor, "b 1 h w"],
        mask: Float[torch.Tensor, "b 1 h w"],
    ) -> Float[torch.Tensor, ""]:
        """Compute the masked MSE loss.

        Args:
            preds (torch.Tensor): The predicted values.
            targets (torch.Tensor): The ground truth values.
            mask (torch.Tensor): The mask to apply.

        Returns:
            torch.Tensor: The computed loss.
        """
        loss = F.mse_loss(preds * mask, targets * mask, reduction="none")
        loss = loss.sum() / (mask.sum() + 1e-8)
        return loss


class ComboLoss(nn.Module):
    """Combination of Dice Loss and Cross-Entropy Loss for segmentation tasks."""

    def __init__(self, weight_dice: float = 0.5, weight_ce: float = 0.5, num_classes: int = 19) -> None:
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", ignore_index=255)


    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined CE + Dice loss."""
        ce = self.ce_loss(preds, targets)
        dice = self.dice_loss(preds, targets)

        return self.weight_ce * ce + self.weight_dice * dice
