import logging

import hydra
import lightning as L
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import rootutils
import torch
from torchmetrics.classification import MulticlassJaccardIndex
from omegaconf import DictConfig
from tqdm.auto import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)


def load_from_checkpoint(
    module: L.LightningModule,
    checkpoint_path: str,
    device: str = "cpu",
) -> L.LightningModule:
    """Load model weights from a checkpoint file.

    Args:
        module (L.LightningModule): The LightningModule instance to load weights into.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        L.LightningModule: The LightningModule instance with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    module.load_state_dict(checkpoint["state_dict"])
    module.to(device)
    return module


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Main function to run an evaluation over the validation set.
    """
    log.info(f"Loading datamodule <{cfg.data._target_}>...")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")

    log.info(f"Loading module <{cfg.module._target_}>...")
    module: L.LightningModule = hydra.utils.instantiate(cfg.module, cfg=cfg)
    if cfg.checkpoint_path:
        module = load_from_checkpoint(module, cfg.checkpoint_path, cfg.device)
        module.eval()

    miou = MulticlassJaccardIndex(num_classes=19, ignore_index=255).to(cfg.device)

    pbar = tqdm(datamodule.val_dataset, desc="Evaluating", unit="sample")
    for sample in pbar:

        # Send tensors to device.
        sample = {k: v.to(cfg.device).unsqueeze(0) if torch.is_tensor(v) else v for k, v in sample.items()}
        
        # Run inference.
        with torch.inference_mode():
            outputs = module(sample["rgb"])

        # Convert segmentation to preds.
        probs = torch.softmax(outputs["seg"], dim=1)
        preds = torch.argmax(probs, dim=1)

        # Compute IoU.
        miou.update(preds, sample["seg"])
        pbar.set_postfix({"Mean IoU": f"{miou.compute().item():.4f}"})

    log.info(f"Mean IoU: {miou.compute().item():.4f}")
    log.info("👌 Evaluation completed.")


if __name__ == "__main__":
    main()