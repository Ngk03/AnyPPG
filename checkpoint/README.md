# 🧩 AnyPPG Checkpoint

This directory contains the pretrained **AnyPPG encoder checkpoint**.

## 📘 File Description
- **`anyppg_ckpt.pth`** - pretrained weights of the AnyPPG encoder.

The checkpoint was trained on **over 100,000 hours** of synchronized **PPG** and **ECG** recordings from **58,796 subjects**.  
It serves as the initialization for downstream tasks such as linear probing or fine-tuning.

## ⚙️ Training Overview
- **Objective:** Cross-modal contrastive alignment between PPG and ECG signals (CLIP-style)
- **Input modality:** PPG (125 Hz, z-score normalized along the time axis)
- **Architecture:** 1D convolutional encoder (`Net1D`)
- **Output:** 1024-dimensional physiological embedding

## 🧠 Usage
To load the pretrained encoder:

```python
import torch
from model.net1d import Net1D

# Model configuration
anyppg_cfg = {
    "in_channels": 1,
    "base_filters": 64,
    "ratio": 1.0,
    "filter_list": [64, 160, 160, 400, 400, 1024],
    "m_blocks_list": [2, 2, 2, 3, 3, 1],
    "kernel_size": 3,
    "stride": 2,
    "groups_width": 16,
    "verbose": False,
}

anyppg = Net1D(**anyppg_cfg)
state_dict = torch.load("./checkpoint/anyppg_ckpt.pth", map_location="cpu")
anyppg.load_state_dict(state_dict)
```

## Note

This checkpoint contains only the encoder parameters and does not include any downstream linear heads or task-specific adapters.
