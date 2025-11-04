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

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)

from viper.data.cityscapes import labels_to_colors


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
    """Main function to run an inference over a sample.
    """
    log.info(f"Loading datamodule <{cfg.data._target_}>...")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    sample = datamodule.val_dataset[3]

    log.info(f"Loading module <{cfg.module._target_}>...")
    module: L.LightningModule = hydra.utils.instantiate(cfg.module, cfg=cfg)
    if cfg.checkpoint_path:
        module = load_from_checkpoint(module, cfg.checkpoint_path, cfg.device)
        module.eval().to(torch.bfloat16)

    # torch.save(module.model.state_dict(), "model_weights.pth")
    # import pdb; pdb.set_trace()

    # Send tensors to device.
    sample = {
        k: v.to(cfg.device).unsqueeze(0).to(torch.bfloat16) if torch.is_tensor(v) else v
        for k, v in sample.items()
    }
    
    # Run inference.
    with torch.inference_mode():
        for _ in range(10):  # Warm-up
            _ = module(sample["rgb"])

    t1 = torch.cuda.Event(enable_timing=True)
    t2 = torch.cuda.Event(enable_timing=True)
    t1.record()

    with torch.inference_mode():
        for _ in range(10):
            outputs = module(sample["rgb"])
    t2.record()

    t1.synchronize()
    t2.synchronize()
    log.info(f"Inference time: {t1.elapsed_time(t2) / 10} ms")

    # Convert segmentation to preds.
    probs = torch.softmax(outputs["seg"], dim=1)
    segmentation_preds = torch.argmax(probs, dim=1, keepdim=True)

    # Convert results to float32 for visualization.
    sample = {k: v.to(torch.float32) for k, v in sample.items()}
    outputs = {k: v.to(torch.float32) for k, v in outputs.items()}

    # Visualize results.
    plt.figure(figsize=(16, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(sample["rgb"][0].cpu().permute(1, 2, 0))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(outputs["depth"][0].cpu().squeeze(), cmap="plasma")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(labels_to_colors(segmentation_preds[0].cpu().squeeze().numpy()))
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("inference_results.png")

    log.info("Inference completed. ✅")


if __name__ == "__main__":
    main()