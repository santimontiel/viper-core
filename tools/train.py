import logging

import hydra
import lightning as L
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)


def instantiate_trainer(cfg: DictConfig) -> L.Trainer:
    """Instantiate the Lightning Trainer."""

    # Instantiate the loggers.
    loggers = []
    if cfg.get("loggers", None):
        for logger_cfg in cfg.loggers:
            loggers.append(hydra.utils.instantiate(logger_cfg))
    
    # Instantiate the callbacks.
    callbacks = []
    if cfg.get("callbacks", None):
        for callback_cfg in cfg.callbacks:
            callbacks.append(hydra.utils.instantiate(callback_cfg))

    return L.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=cfg.strategy,
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        callbacks=callbacks,
        logger=loggers,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        accumulate_grad_batches=cfg.effective_batch_size // (cfg.batch_size * cfg.devices),
        gradient_clip_val=cfg.gradient_clip_val,
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Main function to run the training process.
    """
    log.info(f"Loading datamodule <{cfg.data._target_}>...")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")

    log.info(f"Loading module <{cfg.module._target_}>...")
    module: L.LightningModule = hydra.utils.instantiate(cfg.module, cfg=cfg)

    log.info("Instantiating the trainer...")
    trainer: L.Trainer = instantiate_trainer(cfg.trainer)

    log.info("Starting training...")
    trainer.fit(module, datamodule)

    log.info("Training completed. ✅")


if __name__ == "__main__":
    main()