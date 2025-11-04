from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
import rootutils
from transformers import get_cosine_schedule_with_warmup
from torchmetrics.classification import MulticlassJaccardIndex

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from viper.losses import ComboLoss, MaskedMSELoss
from viper.metrics import MaskedMAE, MaskedRMSE, DiceScore


class ViperModule(L.LightningModule):
    def __init__(
        self,
        cfg: Any = None,
        model: torch.nn.Module = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.save_hyperparameters(ignore=["model"])

        # Define the losses.
        self.depth_loss_fn = MaskedMSELoss()
        self.segmentation_loss_fn = ComboLoss()

        # Define the metrics.
        self.depth_mae_metric = MaskedMAE()
        self.depth_rmse_metric = MaskedRMSE()
        self.seg_dice_metric = DiceScore(num_classes=19, ignore_index=255, average="macro",)
        self.train_miou = MulticlassJaccardIndex(num_classes=19, ignore_index=255)
        self.val_miou = MulticlassJaccardIndex(num_classes=19, ignore_index=255)


    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x)


    def common_step(self, batch, batch_idx, stage: str):
        # Extract inputs and targets from the batch.
        inputs = batch["rgb"]
        depth_targets = batch["depth"]
        seg_targets = batch["seg"]

        # Forward pass through the model.
        outputs = self(inputs)
        seg_preds_soft = F.softmax(outputs["seg"], dim=1)
        seg_preds = torch.argmax(seg_preds_soft, dim=1)

        # Calculate the loss.
        depth_preds = outputs["depth"] / 80.0
        depth_targets = depth_targets / 80.0
        depth_mask = depth_targets > 0
        depth_loss = self.depth_loss_fn(depth_preds, depth_targets, depth_mask)
        seg_loss = self.segmentation_loss_fn(outputs["seg"], seg_targets)
        combined_loss = depth_loss + seg_loss

        # Log losses.
        self.log(f"{stage}/depth_loss", depth_loss.item(), sync_dist=True, prog_bar=True)
        self.log(f"{stage}/seg_loss", seg_loss.item(), sync_dist=True, prog_bar=True)
        self.log(f"{stage}/total_loss", combined_loss.item(), sync_dist=True, prog_bar=True)

        # Update and log metrics.
        self.depth_mae_metric.update(outputs["depth"], depth_targets, depth_mask)
        self.log(f"{stage}/depth_mae", self.depth_mae_metric.compute(), sync_dist=True, prog_bar=True)
        self.depth_rmse_metric.update(outputs["depth"], depth_targets, depth_mask)
        self.log(f"{stage}/depth_rmse", self.depth_rmse_metric.compute(), sync_dist=True, prog_bar=True)
        self.seg_dice_metric.update(seg_preds, seg_targets.unsqueeze(1))
        self.log(f"{stage}/seg_dice", self.seg_dice_metric.compute(), sync_dist=True, prog_bar=True)
        if stage == "train":
            self.train_miou.update(seg_preds, seg_targets)
            self.log(f"{stage}/mIoU", self.train_miou.compute(), sync_dist=True, prog_bar=True)
        else:
            self.val_miou.update(seg_preds, seg_targets)
            self.log(f"{stage}/mIoU", self.val_miou.compute(), sync_dist=True, prog_bar=True)

        return combined_loss


    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, "train")


    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, "val")


    def configure_optimizers(self):

        c = self.cfg

        # Optimizer.
        if c.trainer.optimizer.name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=c.trainer.optimizer.lr,
                weight_decay=c.trainer.optimizer.weight_decay,
                betas=c.trainer.optimizer.betas,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.optimizer.name}")
        
        # Learning rate scheduler.
        if c.trainer.scheduler.name == "get_cosine_schedule_with_warmup":
            num_training_steps = (
                int(c.trainer.max_epochs  # Number of epochs.
                * int((2975 + 7539) // c.trainer.batch_size)  # Steps per epoch.
                / (c.trainer.effective_batch_size // (c.trainer.batch_size * c.trainer.devices))  # Effective batch size.
                / c.trainer.devices)
                + 1000  # Safety margin.
            )
            num_warmup_steps = c.trainer.scheduler.num_warmup_epochs * (num_training_steps // c.trainer.max_epochs)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.cfg.scheduler.name}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }