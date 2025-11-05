# viper-core

**Fine-Tuning DINOv3 Backbones for Robot Perception**

This repository provides a modular framework to fine-tune DINOv3 backbones for downstream robot perception tasks.  
Currently, the framework supports **semantic segmentation** and **depth estimation** using a shared DINOv3 encoder.

## 🚀 Overview

**Supported tasks:**

- 🧩 **Semantic Segmentation** — dense scene parsing for robot navigation
- 🌊 **Depth Estimation** — monocular or stereo-based scene geometry prediction

Both tasks share the same DINOv3 encoder (S+/B, more versions incoming...), enabling efficient multi-task or single-task adaptation.

## 🧠 Architecture

```
          ┌──────────────────────┐
          │ DINOv3 ViT + Adapter │  ← fine-tune pretrained backbone
          └──────────┬───────────┘
                     │
       ┌─────────────┴─────────────┐
       │                           │
┌──────▼──────┐           ┌────────▼────────┐
│   Seg Dec.  │           │   Depth Dec.    │
└─────────────┘           └─────────────────┘
       │                           │
┌──────▼──────┐           ┌────────▼────────┐
│   Seg Head  │           │   Depth Head    │
└─────────────┘           └─────────────────┘

```

## ⚙️ Installation

We provide a Dockerfile and Makefile for easy setup. Please ensure you have Docker and nvidia-docker pre-installed. Current setup is tested with CUDA 12.4
and Ubuntu 22.04.

```bash
git clone https://github.com/santimontiel/viper-core.git
cd viper-core

# Setup the Docker environment.
make build

# Set environments variables for dataset paths.
export PATH_TO_CITYSCAPES=/path/to/Cityscapes
export PATH_TO_URBANSYN=/path/to/UrbanSyn

# Launch a container and start playing!
make run

```

## 🧩 Usage
Inside the Docker container, uv manages Python environments and dependencies.
All hyperparameters are managed via OmegaConf/Hydra YAML configs in `configs/`.

1. To train:
```bash
uv run tools/train.py
```

2. To run inference over the CityScapes validation set:
```bash
uv run tools/eval.py
```
3. To make inference with a pretrained checkpoint:
```bash
uv run tools/eval.py checkpoint_path=/path/to/ckpt.ckpt
```

## 🫂 Acknowledgements

This work is supported by project PID2024-161576OB-I00, funded by MCIN/AEI/10.13039/501100011033 and co-funded by the European Regional Development Fund (ERDF, “A way of making Europe”).