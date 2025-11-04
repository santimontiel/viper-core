import os
from pathlib import Path
from typing import Dict, Literal

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import rootutils
from torch.utils.data import Dataset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from viper.data.cityscapes import labels_to_colors
from viper.data.transforms import Normalize, RandomFlip

class UrbanSynDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: Literal["train", "val"] = "train",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split

        self.rgb_files = sorted(list((self.data_dir / "rgb").rglob("*.png")))
        self.depth_files = sorted(list((self.data_dir / "depth").rglob("*.exr")))
        self.seg_files = sorted(list((self.data_dir / "ss").rglob("*.png")))

        self.transforms = [
            Normalize(),
            RandomFlip(p=0.5 if split == "train" else 0.0),
        ]

    def __len__(self) -> int:
        return len(self.rgb_files)

    def read_exr_depth_cv2(self, path: str) -> np.ndarray:
        """Read 32-bit EXR depth map using OpenCV (units: 1e5 meters)."""
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) * 1e5
        return depth

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        # Collect files.
        rgb_path = self.rgb_files[idx]
        depth_path = self.depth_files[idx]
        seg_path = self.seg_files[idx]
        assert (
            rgb_path.stem.split("_")[1]
            == depth_path.stem.split("_")[1]
            == seg_path.stem.split("_")[1]
        ), f"File names do not match: {rgb_path}, {depth_path}, {seg_path}"

        # Load data.
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        depth = self.read_exr_depth_cv2(str(depth_path))
        depth = np.clip(depth, 0, 80)  # Clip to 80 meters.
        seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)[..., 0]
        seg[seg >= 19] = 255  # Ignore labels >= 19.

        # Resize to 512x1024.
        rgb = cv2.resize(rgb, (1024, 512), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (1024, 512), interpolation=cv2.INTER_NEAREST)
        seg = cv2.resize(seg, (1024, 512), interpolation=cv2.INTER_NEAREST)

        # Convert to tensors.
        rgb = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        depth = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)
        seg = torch.from_numpy(seg.astype(np.int64))

        # Apply transforms.
        sample = {"rgb": rgb, "depth": depth, "seg": seg}
        for transform in self.transforms:
            sample = transform(sample)

        return sample


if __name__ == "__main__":
    dataset = UrbanSynDataset(data_dir="/data/UrbanSyn", split="train")
    sample = dataset[909]
    print(f"Dataset length: {len(dataset)}")
    print(sample["rgb"].shape, sample["depth"].shape, sample["seg"].shape)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(sample["rgb"].permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(np.log(sample["depth"].squeeze(0).numpy()), cmap="plasma")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(labels_to_colors(sample["seg"].numpy()))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("sample_urbansyn.png")