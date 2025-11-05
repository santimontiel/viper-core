from typing import Any

import lightning.pytorch as L
import rootutils
from torch.utils.data import random_split, DataLoader, ConcatDataset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from viper.data.urbansyn import UrbanSynDataset
from viper.data.cityscapes import CityScapesDataset


class ViperDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        cityscapes_data_dir: str,
        urbansyn_data_dir: str,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int,
        debug_mode: bool = False,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.cityscapes_data_dir = cityscapes_data_dir
        self.urbansyn_data_dir = urbansyn_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.debug_mode = debug_mode

        if self.debug_mode:
            self.batch_size = 1
            self.num_workers = 0
            self.prefetch_factor = None

    def setup(self, stage: str | None = None):
        train_dataset_1 = CityScapesDataset(data_dir=self.cityscapes_data_dir, split="train")
        train_dataset_2 = UrbanSynDataset(data_dir=self.urbansyn_data_dir, split="train")
        train_dataset = ConcatDataset([train_dataset_1, train_dataset_2])
        val_dataset = CityScapesDataset(data_dir=self.cityscapes_data_dir, split="val")

        if stage == "fit":
            self.train_dataset = train_dataset
        if stage in ["fit", "validate"]:
            self.val_dataset = val_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=True if not self.debug_mode else False,
            pin_memory=False,
            persistent_workers=False,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
        )
