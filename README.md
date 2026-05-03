# RL-Driven Adaptive Data Augmentation for Traffic Sign Recognition

A reinforcement learning framework that learns optimal data augmentation policies for traffic sign classification using Deep Q-Networks (DQN) and YOLO-based preprocessing.

## Overview

This project proposes a novel approach to data augmentation for traffic sign recognition by replacing fixed augmentation schedules with an adaptive policy learned via reinforcement learning. Instead of applying the same transformations throughout training, a DQN agent observes validation metrics and selects from six augmentation operations per epoch, optimizing directly for validation accuracy.

**Key Results:**
- **98.34%** test accuracy with RL-driven augmentation
- **97.74%** with fixed augmentation (baseline)
- **96.33%** with no augmentation

Tested on the [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/).

## Features

- **Adaptive Augmentation Policy**: DQN agent learns which augmentations help most during training
- **Real-World Data Pipeline**: YOLOv8-based preprocessing extracts traffic signs from raw dashcam footage
- **Modular Design**: Three comparable systems (baseline, fixed augmentation, RL-driven) for clear ablation
- **Deployment-Ready Insights**: Addresses the train/test gap between benchmark datasets and real driving scenarios

## Architecture

### CNN Classifier
A shallow two-block CNN trained at 32×32 resolution (GTSRB standard):
```
Conv2D(32, 3×3) → ReLU → MaxPool(2×2)
Conv2D(64, 3×3) → ReLU → MaxPool(2×2)
Flatten → Dense(128) → ReLU → Dropout(0.3) → Dense(43, softmax)
```

### DQN Agent
- **State**: Current validation accuracy, validation loss, previous action (8-dim vector)
- **Action Space**: Rotation, Brightness, Gaussian Noise, Translation, Zoom, Identity
- **Reward**: Per-epoch change in validation accuracy
- **Q-Network**: 3-layer MLP (8 → 64 → 64 → 6)

### YOLO Preprocessing
YOLOv8 extracts traffic sign crops from raw dashcam video, producing training samples that better reflect real deployment conditions.

## Getting Started

### Requirements
```bash
python >= 3.9
tensorflow >= 2.13
torch >= 2.0
ultralytics >= 8.0  # YOLOv8
scikit-learn
numpy
opencv-python
```

### Installation
```bash
git clone https://github.com/yourusername/Traffic-Sign-Recognition-System-RL-and-Data-Aug.git
cd Traffic-Sign-Recognition-System-RL-and-Data-Aug
pip install -r requirements.txt
```

### Quick Start

#### 1. Download GTSRB Dataset
```bash
# Download from http://benchmark.ini.rub.de/
# Extract to ./data/gtsrb/
```

#### 2. Train Baseline CNN
```python
python Baseline_CNN_&_RL_Augmentation.ipynb
# Or run the notebook directly in Jupyter
```

#### 3. Train with Fixed Augmentation
```python
python Fixed_Augmentation_CNN.ipynb
```

#### 4. Train with RL-Driven Augmentation
```python
python Baseline_CNN_&_RL_Augmentation.ipynb
# Uncomment the RL training section
```

#### 5. Extract Traffic Signs from Dashcam Video
```python
from yolo_crop_extractor import extract_signs_from_video

extract_signs_from_video(
    video_path='dashcam_footage.mp4',
    output_dir='dashcam_crops/',
    confidence_threshold=0.5
)
```

## Experimental Setup

| Component | Baseline | Fixed Aug | RL-Driven |
|-----------|----------|-----------|-----------|
| Epochs    | 20       | 30        | 50        |
| Augmentations | None | 5 ops (fixed) | 6 ops (adaptive) |
| Test Accuracy | 96.33% | 97.74% | 98.34% |

### Augmentation Operations
- **Rotation**: ±15°
- **Brightness**: ±0.3
- **Gaussian Noise**: σ=0.05
- **Translation**: ±10% of image dimensions
- **Zoom**: [-15%, +15%]
- **Identity**: No transformation

## Key Findings

1. **DQN Learning Behaviour**: The agent learns to prioritize identity (11/50 epochs) and noise injection (10/50 epochs), recognizing when augmentation plateaus.

2. **Modest but Meaningful Gain**: 0.60 percentage points over fixed augmentation (≈75 fewer misclassifications on 12,630 test samples).

3. **Speed Limit Confusion**: All models struggle most with visually similar speed limit signs (classes 0–8), where the distinguishing feature is a small numeral.

4. **Deployment Gap**: Real dashcam footage differs significantly from pre-cropped GTSRB images in motion blur, crop tightness, and background clutter.

## Reproduction Notes

- **Training Time**: ~15–20 minutes per configuration on GPU (NVIDIA A100)
- **Early Stopping**: Applied only to fixed augmentation (patience=7) to avoid conflating augmentation benefit with model selection
- **Hyperparameters**: Fixed across conditions; differences reflect augmentation strategy only

## Paper & Citation

This work is part of an Intelligent Autonomous Systems (IAI) course project at Manipal Institute of Technology. For details, see `IAI_Project_Final_Draft.pdf`.

If you use this code, please cite:
```bibtex
@inproceedings{verma2025rl_augmentation,
  title={RL-Driven Adaptive Data Augmentation for Traffic Sign Recognition Using CNNs and YOLO-Based Cropping},
  author={Verma, Shlok and Telikicherla, Isha Sarvani and Ragannagari, Praket Reddy and Biswal, Aatman and Kedia, Kushagra},
  year={2025},
  institution={Manipal Institute of Technology}
}
```

## Future Work

- **Faster ε Decay**: Tighter exploration schedule to allow earlier policy exploitation
- **Pseudo-Labelling Pipeline**: Fold YOLO-extracted dashcam crops into training with self-supervision
- **Extended Action Space**: Additional augmentation operations (elastic deformation, color jitter)
- **Transfer Learning**: Pre-train on synthetic traffic sign data, fine-tune with RL

## Troubleshooting

**Out of Memory**: Reduce batch size in `Baseline_CNN_&_RL_Augmentation.ipynb` (default: 32)

**GTSRB Download Issues**: Use the direct link at [http://benchmark.ini.rub.de/](http://benchmark.ini.rub.de/)

**YOLOv8 Model Size**: First inference auto-downloads YOLOv8 weights (~6.3 MB); ensure internet connection

## Project Structure
```
.
├── Baseline_CNN_&_RL_Augmentation.ipynb  # Main DQN training code
├── Fixed_Augmentation_CNN.ipynb          # Fixed augmentation baseline
├── yolo_crop_extractor.py               # YOLO preprocessing pipeline
├── data/
│   └── gtsrb/                           # GTSRB dataset (download separately)
├── results/
│   ├── confusion_matrices/
│   ├── training_curves/
│   └── models/
├── IAI_Project_Final_Draft.pdf          # Full paper
├── RL_Traffic_Sign_Presentation.pdf     # Presentation slides
└── README.md                             # This file
```

## Authors

- **Shlok Verma** — Lead, DQN implementation & experiments
- **Isha Sarvani Telikicherla** — Data processing & validation
- **Praket Reddy Ragannagari** — YOLO pipeline & preprocessing
- **Kushagra Kedia** — Analysis & results interpretation
- **Aatman Biswal** — Documentation & reproducibility

All authors: School of Computer Science and Engineering (Data Science), Manipal Institute of Technology, Manipal Academy of Higher Education, India.

## License

This project is shared for educational and research purposes. Please refer to your institution's academic integrity policies before use.

## Questions or Feedback?

For questions or issues, please open a GitHub issue or contact the authors directly via email or through your institution.

---

**Last Updated**: May 2, 2026  
**Status**: Complete (Course Submission)
