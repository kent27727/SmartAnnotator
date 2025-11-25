# 🎯 Auto-Annotation & Training Tool

A comprehensive semi-supervised learning pipeline for object segmentation, featuring automatic annotation with trained models, manual annotation GUI, and multi-project support.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v11%20%7C%20v8-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Workflow Overview](#-workflow-overview)
- [Project Structure](#-project-structure)
- [Usage Guide](#-usage-guide)
- [Configuration](#-configuration)
- [Server Training](#-server-training)
- [Contributing](#-contributing)

## ✨ Features

### 🔹 Multi-Project Support
- Create and manage multiple datasets/projects
- Each project has its own configuration, classes, and models
- Easy switching between projects

### 🔹 Model Flexibility
- **YOLOv11** - Latest YOLO architecture
- **YOLOv8** - Stable and well-tested
- **ResNet** - For classification tasks
- Multiple sizes: Nano, Small, Medium, Large, XLarge

### 🔹 Task Types
- **Segmentation** - Pixel-level object masks
- **Detection** - Bounding box localization
- **Classification** - Image-level categorization

### 🔹 Semi-Supervised Learning Pipeline
1. Manual annotation (small dataset)
2. Initial model training
3. Auto-annotation (large dataset)
4. Human review & correction
5. Final model training

### 🔹 Smart Validation
- Minimum detection threshold (e.g., 2 eyes for dark circles)
- Confidence-based filtering
- Invalid images saved separately for review

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/auto-annotation-tool.git
cd auto-annotation-tool

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.21.0
gradio>=4.0.0
tqdm>=4.60.0
pyyaml>=6.0
```

## 🏃 Quick Start

```bash
# Run the main application
python main.py
```

This will launch an interactive menu:

```
📋 MAIN MENU
============================================================

🎯 Active Project: None

    ━━━━━━━━━━ PROJECT MANAGEMENT ━━━━━━━━━━
    [1] 📁 Project Management (Create/Load/Edit)
    
    ━━━━━━━━━━ ANNOTATION ━━━━━━━━━━
    [2] 📥 Import Data (Raw images / Annotations)
    [3] ✏️ Manual Annotation GUI (Gradio)
    [4] 🤖 Start Auto Annotation
    [5] 📦 Prepare Final Dataset
    
    ━━━━━━━━━━ MODEL TRAINING ━━━━━━━━━━
    [6] 🚀 Train Initial Model (with manual annotations)
    [7] 🎯 Train Final Model
    [8] 🖥️ Server Export (ZIP)
    
    ━━━━━━━━━━ INFO ━━━━━━━━━━
    [9] 📊 Show Project Status
    [10] ⚙️ Project Settings

    [0] 🚪 Exit
```

## 📊 Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEMI-SUPERVISED PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  1. CREATE   │───▶│  2. MANUAL   │───▶│  3. INITIAL  │       │
│  │   PROJECT    │    │  ANNOTATION  │    │   TRAINING   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  • Select model      • Use Gradio GUI    • Train with           │
│  • Choose task       • Draw polygons       small dataset        │
│  • Define classes    • 200-300 samples   • Get initial model    │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   6. FINAL   │◀───│  5. PREPARE  │◀───│   4. AUTO    │       │
│  │   TRAINING   │    │   DATASET    │    │  ANNOTATION  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  • Train with        • Merge manual      • Use trained model    │
│    full dataset        + auto            • Annotate thousands   │
│  • Export model      • Split train/val   • Filter invalid       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
project_root/
├── main.py                 # Main entry point
├── config.py               # Global configuration
├── project_manager.py      # Multi-project management
├── annotation_tool.py      # Gradio-based manual annotation GUI
├── auto_annotate.py        # Automatic annotation engine
├── train_model.py          # Model training logic
├── prepare_final_dataset.py# Dataset preparation utilities
├── utils.py                # Helper functions
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
└── projects/               # All projects stored here
    └── my_project/
        ├── project_config.json  # Project configuration
        ├── classes.txt          # Class definitions
        ├── raw_images/          # Unlabeled images
        ├── manual_annotations/  # Manual labels
        │   ├── images/
        │   └── labels/
        ├── auto_annotations/    # Auto-generated labels
        │   ├── images/
        │   ├── labels/
        │   ├── visualizations/  # Annotation previews
        │   └── unvalid/         # Rejected images
        ├── final_dataset/       # Ready for training
        │   ├── train/
        │   ├── val/
        │   ├── test/
        │   └── dataset.yaml
        └── models/              # Trained models
            └── latest_model.pt
```

## 📖 Usage Guide

### Step 1: Create a Project

```bash
python main.py
# Select [1] Project Management
# Select [1] Create New Project
```

You'll be prompted to configure:
- **Project name**: e.g., `dark_circle_detection`
- **Model family**: YOLOv11, YOLOv8, or ResNet
- **Task type**: Segmentation, Detection, or Classification
- **Model size**: Nano to XLarge
- **Train/Val/Test split**: Automatic or manual ratios
- **Classes**: e.g., `dark_circle, wrinkle, eyebag`

### Step 2: Import Raw Images

```bash
# Select [2] Import Data
# Select [1] Import raw images
# Enter path to your image folder
```

### Step 3: Manual Annotation (Gradio GUI)

```bash
# Select [3] Manual Annotation GUI
# Opens browser at http://localhost:7861
```

**GUI Features:**
- Load project and images
- Click to draw polygon points
- Click on start point (white ring) to complete
- Zoom slider for detail work
- Add new classes dynamically
- Undo/Clear functionality
- Save & Next workflow

### Step 4: Train Initial Model

```bash
# Select [6] Train Initial Model
# Confirm training parameters
# Training starts automatically
```

### Step 5: Auto Annotation

```bash
# Select [4] Start Auto Annotation
# Model annotates all raw images
# Valid images: saved to labels/
# Invalid images: saved to unvalid/
```

**Validation Rules:**
- Minimum detections required (default: 2)
- Confidence threshold filtering
- Low confidence items flagged for review

### Step 6: Prepare Final Dataset

```bash
# Select [5] Prepare Final Dataset
# Merges manual + auto annotations
# Splits into train/val/test
```

### Step 7: Train Final Model

```bash
# Select [7] Train Final Model
# Enter number of epochs
# Training with full dataset
```

## ⚙️ Configuration

### Project Configuration (project_config.json)

```json
{
  "project_name": "dark_circle_v1",
  "model": {
    "family": "yolov11",
    "task": "segmentation",
    "size": "m",
    "weights": "yolo11m-seg.pt"
  },
  "classes": {
    "0": "dark_circle"
  },
  "split": "auto",
  "training": {
    "epochs": 100,
    "batch_size": 16,
    "image_size": 640
  },
  "annotation": {
    "confidence_threshold": 0.5,
    "min_detections": 2
  }
}
```

### Model Sizes

| Size | Name | Speed | Accuracy | Use Case |
|------|------|-------|----------|----------|
| n | Nano | ⚡⚡⚡⚡⚡ | ⭐ | Edge devices |
| s | Small | ⚡⚡⚡⚡ | ⭐⭐ | Mobile |
| m | Medium | ⚡⚡⚡ | ⭐⭐⭐ | Balanced |
| l | Large | ⚡⚡ | ⭐⭐⭐⭐ | High accuracy |
| x | XLarge | ⚡ | ⭐⭐⭐⭐⭐ | Maximum accuracy |

### Split Ratios (Automatic)

| Dataset Size | Train | Val | Test |
|--------------|-------|-----|------|
| < 1000 images | 70% | 20% | 10% |
| 1000-5000 images | 80% | 10% | 10% |
| > 5000 images | 85% | 10% | 5% |

## 🖥️ Server Training

For training on a GPU server:

```bash
# Select [8] Server Export (ZIP)
# Enter export name
# ZIP file created with all necessary files
```

**On Server:**
```bash
unzip project_server.zip
cd project_server
pip install -r requirements.txt
python main.py
```

After training, the best model is saved at:
```
models/dark_circle_seg/train_YYYYMMDD_HHMMSS/weights/best.pt
```

## 📝 Label Format

### YOLO Segmentation Format
```
# class_id x1 y1 x2 y2 x3 y3 ... (normalized 0-1)
0 0.234 0.456 0.345 0.567 0.456 0.678 ...
```

### YOLO Detection Format
```
# class_id x_center y_center width height (normalized 0-1)
0 0.5 0.5 0.2 0.3
```

## 🔧 Advanced Usage

### Custom Confidence Threshold

```python
from auto_annotate import AutoAnnotator

annotator = AutoAnnotator(model_path, min_detections=2)
annotator.annotate_batch(
    images_dir=Path("./images"),
    confidence_threshold=0.7,  # Higher = more strict
    save_visualizations=True
)
```

### Programmatic Training

```python
from train_model import DarkCircleTrainer

trainer = DarkCircleTrainer()
trainer.prepare_training_data(roboflow_export_dir=Path("./data"))
trainer.train(epochs=150, batch_size=32)
trainer.evaluate()
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
- [Gradio](https://gradio.app/) for the annotation GUI framework
- [OpenCV](https://opencv.org/) for image processing

---

**Made with ❤️ for the computer vision community**
