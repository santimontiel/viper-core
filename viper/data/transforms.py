from typing import Tuple

import torch
import torchvision.transforms as transforms


class Normalize:
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.transform = transforms.Normalize(mean, std)

    def __call__(self, sample):
        images = sample['rgb']
        sample["orig_rgb"] = images
        sample["rgb"] = self.transform(images)
        return sample
    

class Denormalize:
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.transform = transforms.Normalize(
            mean=[-m/s for m, s in zip(mean, std)],
            std=[1/s for s in std]
        )

    def __call__(self, sample):
        sample["rgb"] = self.transform(sample['rgb'])
        return sample
    

class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        rgb, depth, seg = sample["rgb"], sample["depth"], sample["seg"]
        if torch.rand(1) < self.p:
            rgb = torch.flip(rgb, [-1])
            depth = torch.flip(depth, [-1])
            seg = torch.flip(seg, [-1])
        sample["rgb"] = rgb
        sample["depth"] = depth
        sample["seg"] = seg
        return sample