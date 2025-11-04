import os
from pathlib import Path
from typing import Dict, Literal

import cv2
import numpy as np
import rootutils
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from viper.data.transforms import Normalize, RandomFlip


class CityScapesDataset(Dataset):
    
    baseline: float = 0.209313
    fx: float = 2262.52
    fy: float = 2265.30179059858554

    mapping_20 = {
        0: 255,
        1: 255,
        2: 255,
        3: 255,
        4: 255,
        5: 255,
        6: 255,
        7: 0,
        8: 1,
        9: 255,
        10: 255,
        11: 2,
        12: 3,
        13: 4,
        14: 255,
        15: 255,
        16: 255,
        17: 5,
        18: 255,
        19: 6,
        20: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        29: 255,
        30: 255,
        31: 16,
        32: 17,
        33: 18,
        -1: 255,
    }

    def __init__(
        self,
        data_dir: str,
        split: Literal["train", "val"] = "train",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split

        self.rgb_files = sorted(list((self.data_dir / "leftImg8bit" / split).rglob("*.png")))
        self.depth_files = sorted(list((self.data_dir / "disparity" / split).rglob("*.png")))
        self.seg_files = sorted(list((self.data_dir / "gtFine" / split).rglob("*_labelIds.png")))
        
        self.transforms = [
            Normalize(),
            RandomFlip(p=0.5 if split == "train" else 0.0),
        ]

    def __len__(self) -> int:
        return len(self.rgb_files)
    
    def read_disparity_cv2(self, path: str) -> np.ndarray:
        """Read 16-bit PNG disparity map using OpenCV and convert to depth (units: meters)."""
        disp = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0  # Convert to float disparity
        with np.errstate(divide='ignore'):
            depth = (self.fx * self.baseline) / disp  # Depth = (f * B) / disparity
        depth[disp == 0] = 0  # Set depth to 0 where disparity is 0 (invalid)
        return depth
    
    def read_and_encode_labels(self, path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        label_mask = np.zeros_like(mask)
        for k in self.mapping_20:
            label_mask[mask == k] = self.mapping_20[k]
        return label_mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        # Collect files.
        rgb_path = self.rgb_files[idx]
        depth_path = self.depth_files[idx]
        seg_path = self.seg_files[idx]
        assert (
            rgb_path.stem.split("_")[0]
            == depth_path.stem.split("_")[0]
            == seg_path.stem.split("_")[0]
        ), f"File names do not match: {rgb_path}, {depth_path}, {seg_path}"

        # Load data.
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        depth = self.read_disparity_cv2(str(depth_path))
        depth = np.clip(depth, 0, 80)  # Clip to 80 meters.
        seg = self.read_and_encode_labels(str(seg_path))

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
    
def labels_to_colors(label_mask: np.ndarray) -> np.ndarray:
    """Convert label mask to color image using Cityscapes palette."""
    cityscapes_palette = [
        (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156),
        (190,153,153), (153,153,153), (250,170, 30), (220,220,  0),
        (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60),
        (255,  0,  0), (  0,  0,142), (  0,  0, 70), (  0, 60,100),
        (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,  0),
    ]
    h, w = label_mask.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, color in enumerate(cityscapes_palette):
        color_image[label_mask == label_id] = color
    return color_image

if __name__ == "__main__":
    dataset = CityScapesDataset(data_dir="/data/cityscapes", split="train")
    sample = dataset[0]
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
    plt.savefig("sample_cityscapes.png")