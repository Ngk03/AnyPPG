<div align="left">
  <h1>AnyPPG: An ECG-Guided PPG Foundation Model Trained on Over 100,000 Hours of Recordings for Holistic Health Profiling</h1>
  <p>
    <a href="https://arxiv.org/pdf/2511.01747">
      <img src="https://img.shields.io/badge/arXiv-2502.12345-b31b1b.svg" alt="arXiv">
    </a>
  </p>
</div>


AnyPPG is a **photoplethysmography (PPG) foundation model** pretrained on **over 100,000 hours** of synchronized PPG-ECG recordings from **58,796 subjects**.  

We evaluate it across a diverse set of downstream tasks, including conventional physiological analysis on six public datasets (e.g., heart rate estimation, atrial fibrillation detection) and broader ICD-10 disease diagnosis (chapter I-XV) on the MC-MED dataset, where it achieves an AUROC above 0.70 in 137 diseases.


---

## 🧩 Installation and Usage

The following script demonstrates the **complete workflow** — from installation to loading the pretrained encoder, extracting embeddings, and applying the model to downstream tasks via linear probing or fine-tuning.

```python
# ===============================
# 1. Environment Setup
# ===============================
# Clone the repository and install dependencies
# !git clone https://github.com/Ngks03/AnyPPG.git
# %cd AnyPPG
# !pip install -r requirements.txt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from model.net1d import Net1D
import numpy as np


# ===============================
# 2. Load the AnyPPG Encoder
# ===============================
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

# Load pretrained model weights
ckpt_path = "./checkpoint/anyppg_ckpt.pth"
state_dict = torch.load(ckpt_path, map_location="cpu")
anyppg.load_state_dict(state_dict)
anyppg.eval()
print("✅ AnyPPG encoder loaded successfully.")


# ===============================
# 3. Linear Probing (Feature Extraction)
# ===============================
# Freeze encoder parameters
for p in anyppg.parameters():
    p.requires_grad = False

# Example input (batch_size=8, 1 channel, 2048 samples)
x = torch.randn(8, 1, 2048)
with torch.no_grad():
    features = anyppg(x)          # [batch, feature_dim, T]
    features = features.mean(-1)  # temporal pooling → [batch, feature_dim]

# Example dummy labels
labels = torch.randint(0, 2, (8,))

# Train a simple linear classifier
clf = LogisticRegression(max_iter=200)
clf.fit(features.numpy(), labels.numpy())
print("✅ Linear probing complete.")


# ===============================
# 4. End-to-End Fine-tuning
# ===============================
class LinearProbe(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        x = x.mean(dim=-1)
        return self.fc(x)

# Combine encoder and classification head
model = nn.Sequential(anyppg, LinearProbe(in_dim=1024, num_classes=5))
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

# Example training step
x = torch.randn(8, 1, 2048)
y = torch.randint(0, 5, (8,))
logits = model(x)
loss = criterion(logits, y)
loss.backward()
optimizer.step()
print("✅ Fine-tuning step complete.")


# ===============================
# 5. Feature Extraction for Downstream Analysis
# ===============================
anyppg.eval()
embeddings, targets = [], []
for i in range(10):  # simulated batches
    batch_x = torch.randn(16, 1, 2048)
    batch_y = torch.randint(0, 2, (16,))
    with torch.no_grad():
        feats = anyppg(batch_x).mean(dim=-1)
    embeddings.append(feats.cpu().numpy())
    targets.append(batch_y.numpy())

embeddings = np.concatenate(embeddings)
targets = np.concatenate(targets)
np.save("features.npy", embeddings)
np.save("labels.npy", targets)
print("✅ Features saved for downstream analysis.")
