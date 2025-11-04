<div align="left">
  <h1>AnyPPG: An ECG-Guided PPG Foundation Model Trained on Over 100,000 Hours of Recordings for Holistic Health Profiling</h1>
  <p>
    <a href="https://arxiv.org/abs/2511.01747">
      <img src="https://img.shields.io/badge/arXiv-2511.01747-b31b1b.svg" alt="arXiv">
    </a>
  </p>
  <p><em>🚧 This work is under active development - ongoing improvements and new releases will follow.</em></p>
</div>


---

## 🩺 Overview

**AnyPPG** is a **photoplethysmography (PPG) foundation model** pretrained on **over 100,000 hours** of synchronized **PPG–ECG recordings** from **58,796 subjects**, using a CLIP-style contrastive alignment framework to learn physiologically meaningful representations.

AnyPPG demonstrates strong and versatile performance across a wide range of downstream tasks, including:
- **Conventional physiological analyses** on six public datasets (e.g., heart rate estimation, atrial fibrillation detection). 
- **Large-scale ICD-10 disease diagnosis** (Chapters I-XV) on the MC-MED dataset, achieving an **AUROC above 0.70** in 137 diseases - many of which are non-cardiovascular conditions such as Parkinson's disease and chronic kidney disease.

---

## ⚙️ Getting Started

### 🧩 Installation

Clone the repository:
```bash
git clone https://github.com/Ngks03/AnyPPG.git
cd AnyPPG
```

### 🧠 Using the AnyPPG Encoder

The pretrained checkpoint is available at (./checkpoint/anyppg_ckpt.pth).

**Input requirements**:

- Sampling rate: 125 Hz
- Normalization: z-score normalization along the time axis

```python
import torch
from model.net1d import Net1D

# AnyPPG encoder configuration
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

# Initialize encoder
anyppg = Net1D(**anyppg_cfg)

# Load pretrained model weights
ckpt_path = "./checkpoint/anyppg_ckpt.pth"
state_dict = torch.load(ckpt_path, map_location="cpu")
anyppg.load_state_dict(state_dict)
```

### 📊 Downstream Usage

To evaluate on a downstream task (e.g., heart rate regression or disease classification), freeze the encoder and attach a simple linear head:
```python
import torch.nn as nn

# Freeze encoder
for param in anyppg.parameters():
    param.requires_grad = False

# Linear probing model
linear_head = nn.Linear(1024, num_classes)  # num_classes depends on the task
model = nn.Sequential(anyppg, linear_head)
```

For full fine-tuning, simply unfreeze the encoder:
```python
for param in anyppg.parameters():
    param.requires_grad = True
```

### 📘 Citation
If you find AnyPPG useful in your research, please consider citing:
```bibtex
@article{nie2025anyppg,
  title   = {AnyPPG: An ECG-Guided PPG Foundation Model Trained on Over 100,000 Hours of Recordings for Holistic Health Profiling},
  author  = {Nie, Guangkun and Tang, Gongzheng and Xiao, Yujie and Li, Jun and Huang, Shun and Zhang, Deyun and Zhao, Qinghao and Hong, Shenda},
  journal = {arXiv preprint arXiv:2511.01747},
  year    = {2025}
}
```
