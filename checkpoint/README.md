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

## Note

This checkpoint contains only the encoder parameters and does not include any downstream linear heads or task-specific adapters.
