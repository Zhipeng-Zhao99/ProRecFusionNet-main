# ProRecFusionNet: An Adaptive Progressive Recursive Network for Pansharpening

This repository contains the official PyTorch implementation of the paper  
**"An Adaptive Progressive Recursive Network for Pansharpening"**, accepted as a *Regular Paper* in **IEEE JSTARS (2025)**.

---

## Overview

This project provides training, testing, and evaluation code for **ProRecFusionNet**,  
a progressive recursive fusion framework for pansharpening and related image fusion tasks.

Main modules:
- **Dense Adaptive Upsample Module (DAUM):** enhances spatial detail reconstruction  
- **Gradual Fusion Module (GFM)** with **DRAM-iTransformer:** performs spatial–spectral alignment  
- **Recursive Fusion Module (RFM):** refines and cleans the fused result progressively

---

## Directory Structure

```
├── model/
│ ├── PRFN.py # Network architecture (ProRecFusionNet)
│
├── data/
│ ├── data.py # Dataset loading utilities
│ ├── dataset.py # Dataset classes and normalization/denormalization
│
├── train.py # Training script
├── test.py # Evaluation / inference script
├── args.py # Global configuration (paths, hyperparameters, CUDA setup)
│
├── checkpoint/ # Saved model weights
├── log/ # Training & validation logs
├── result/ # Output fusion results
└── README.md # Project documentation
```

---

##  Requirements

```bash
Python >= 3.8
PyTorch >= 2.0 
Torchvision >= 0.15 
pytorch-msssim
numpy >= 1.23
tqdm
GDAL (e.g., 3.6.x)
thop (optional, for FLOPs/Params profiling)
einops (optional, if your transformer blocks rely on it)
```



## Training

1. **Set dataset paths** in :`args.py`

   ```
   train_data_dir = Path("D:/QB/Dataset/train")
   eval_data_dir  = Path("D:/QB/Dataset/eval")
   ```

2. **Train the model**:

   ```
   python train.py
   ```

Training logs and checkpoints will be saved under:

```
./log/
./checkpoint/
```



## Testing

1. Specify the pretrained model path in :`args.py`

   ```
   pretrained_model_path = Path("./checkpoint/best.pth")
   ```

2. Run inference and metric evaluation:

   ```
   python test.py
   ```

Results  and metrics will be saved in:

```
./result/
./log/test_metrics.txt
```

