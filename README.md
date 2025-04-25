# MLND-IU: Multi-stage Lung Nodule Detection with Improved U-Net++

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx/xxxxx-green)]()

Official PyTorch implementation of **MLND-IU**, a multi-stage detection framework for subcentimeter lung nodules in CT scans, as proposed in the paper ["MLND-IU: A Multi-stage Detection Model of Subcentimeter Lung Nodule with Improved U-Net++"](#).

---

## 📌 Table of Contents
- [Key Features](#-key-features)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Dataset Preparation](#-dataset-preparation)
- [Training & Inference](#-training--inference)
- [Results](#-results)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Key Features
- **Three-Stage Cascade Architecture**:  
  - **Stage 1**: Enhanced RetinaNet with Dynamic Focal Loss for high-recall candidate generation
  - **Stage 2**: Attention-Guided U-Net++ (AG-UNet++) with Dense Attention Bridging Module (DABM) for precise segmentation
  - **Stage 3**: 3D Contextual Pyramid Module (3D-CPM) for false-positive suppression
- **Dynamic Focal Loss**: Adaptive gradient modulation for extreme class imbalance (nodule vs. background)
- **Multi-Scale Feature Fusion**: Cross-scale attention in Feature Pyramid Network (FPN)
- **Clinical Efficiency**: Real-time processing (2.3s per case) with high sensitivity (93.4%) and low FP/Scan (1.4)

---

## 🧠 Model Architecture
![MLND-IU Architecture](docs/architecture.png)  
*(Replace with actual diagram from the paper)*

---

## 🛠 Installation
### Prerequisites
- Python ≥ 3.8
- NVIDIA GPU with CUDA ≥ 11.3
- PyTorch ≥ 2.0

### Step-by-Step Setup
```bash
# Clone repository
git clone https://github.com/Yks151/MLND-IU.git
cd MLND-IU

# Create conda environment
conda create -n mlndiu python=3.8
conda activate mlndiu

# Install dependencies
pip install -r requirements.txt

# Install MONAI for medical imaging
pip install monai

# Install TorchIO for data augmentation
pip install torchio
