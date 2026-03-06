# LL-SDR: Low-Latency Speech Enhancement via Discrete Representations

<p align="center">
  <a href="https://anonymous.4open.science/w/demo_SE-29F5/"><strong>🌐 Demo Page</strong></a>
</p>

## 📌 Overview

This repository contains the official implementation of **LL-SDR: Low-Latency Speech Enhancement via Discrete Representations**.

Speech enhancement systems often face a trade-off between **latency**, **computational efficiency**, and **speech quality**. In this work, we propose **LL-SDR**, a novel speech enhancement framework that leverages **discrete audio representations** to enable **low-latency and efficient denoising** while maintaining high perceptual quality.

Instead of operating directly in the waveform or spectrogram domain, LL-SDR performs enhancement in a **discrete latent space**, which significantly reduces computation and allows fast inference suitable for **real-time applications**.

## 🎧 Demo

Audio examples can be found on our demo page:

👉 **Demo Page:**  
https://anonymous.4open.science/w/demo_SE-29F5/

The demo includes comparisons between:

- Noisy speech
- Baseline speech enhancement methods
- Our proposed **LL-SDR** model

## ✨ Highlights

- 🚀 **Low-latency speech enhancement**
- 🧠 **Discrete representation based denoising**
- ⚡ **Efficient inference**
- 🎧 **High perceptual speech quality**

## 📄 Paper

**LL-SDR: Low-Latency Speech Enhancement via Discrete Representations**

*(Paper link will be added after publication.)*

## 🧩 Method

LL-SDR operates in three main stages:

1. **Audio Encoding**  
   The noisy waveform is encoded into a **discrete representation** using a neural audio codec.

2. **Discrete-domain Enhancement**  
   A lightweight enhancement network operates directly on the discrete tokens to remove noise.

3. **Audio Reconstruction**  
   The enhanced discrete tokens are decoded back into the waveform domain.

This design significantly reduces **latency and computational overhead** compared to conventional waveform-based enhancement systems.

## 📁 Repository Structure
LL-SDR/
│
├── models/ # Model architectures
├── dataset/ # Data loading scripts
├── training/ # Training scripts
├── inference/ # Inference scripts
├── evaluation/ # Evaluation tools
└── README.md


## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/your_repo/LL-SDR.git
cd LL-SDR
pip install -r requirements.txt


### Inference
```bash
python inference/enhance.py \
  --input noisy.wav \
  --output enhanced.wav

## 🏋️ Training

### 1. Dataset Preparation

Prepare paired **noisy-clean speech data** for training.

Example dataset structure:
dataset/
│
├── train/
│ ├── noisy/
│ │ ├── sample1.wav
│ │ ├── sample2.wav
│ │ └── ...
│ └── clean/
│ ├── sample1.wav
│ ├── sample2.wav
│ └── ...
│
└── val/
├── noisy/
└── clean/


Make sure that noisy and clean files share the **same filenames**.

---
### 2. Edit dataset.py

### 3. Train the Model

Run the training script:

```bash
python train.py \
    --train_dir dataset/train \
    --val_dir dataset/val \
    --batch_size 16 \
    --num_epochs 200 \
    --learning_rate 3e-4

### 4. Resume Training
To resume training from a checkpoint:
python train.py \
    --resume checkpoints/model_last.pt

### 5. Logging

Training logs and checkpoints will be saved in:
experiments/
├── checkpoints/
├── logs/
└── configs/

### 6. Checkpoints
The result in paper is obtaiend from pre-trained checkpoint:
