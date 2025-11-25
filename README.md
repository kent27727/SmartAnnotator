# 🎯 SmartAnnotator - Auto-Annotation & Training Tool

A comprehensive semi-supervised learning pipeline for object segmentation, detection, and classification. Annotate thousands of images automatically with just 200-300 manual labels!

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v11%20%7C%20v8-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🚀 Quick Start (30 seconds)

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/SmartAnnotator.git
cd SmartAnnotator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

---

## 📋 Table of Contents

- [What Does This Tool Do?](#-what-does-this-tool-do)
- [Installation](#-installation)
- [Step-by-Step Guide](#-step-by-step-guide)
- [How Auto-Annotation Works](#-how-auto-annotation-works)
- [FAQ](#-faq)
- [Project Structure](#-project-structure)

---

## 🤔 What Does This Tool Do?

**Problem:** Manually labeling 10,000 images takes days.

**Solution:** 
1. Manually label only **200-300 images**
2. Train an **initial model** with these labels
3. Let the model **automatically annotate** the rest
4. Train the final model with all data

**Result:** Days of work reduced to hours! ⚡

---

## 🔧 Installation

### Requirements
- Python 3.8+
- GPU (recommended, but not required)

### Step 1: Create Environment (Recommended)

```bash
# With Conda
conda create -n annotation python=3.10
conda activate annotation

# Or with venv
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run

```bash
python main.py
```

---

## 📖 Step-by-Step Guide

### 🔵 STEP 1: Create a Project

```
python main.py
→ Select [1] Project Management
→ Select [1] Create New Project
```

You'll be asked:
| Question | Example Answer | Description |
|----------|----------------|-------------|
| Project name | `car_detection` | Name of your project |
| Model | `[1] YOLOv11` | Model to use |
| Task | `[1] Detection` or `[3] Segmentation` | Task type |
| Size | `[3] Medium` | Model size |
| Split | `[1] Automatic` | Train/Val/Test ratios |
| Classes | `car, person` | Classes to detect |
| Min Detections | `1` | Minimum detections per image |

### 🔵 STEP 2: Import Images

```
→ Select [2] Import Data
→ Select [1] Import raw images
→ Enter the path to your image folder
```

Example: `C:\Users\john\Desktop\my_images`

### 🔵 STEP 3: Manual Annotation (200-300 images)

```
→ Select [3] Manual Annotation GUI
→ Browser opens at http://localhost:7861
```

**How to Use the GUI:**
1. Select your project from the left panel and click **"Load Project"**
2. Click **"Project Images"** to load images
3. **Click on the image** to draw polygon points
4. To complete the polygon, **click on the start point** (white ring)
5. Click **"Save"** or **"Save →"** to save and go to next

> 💡 **Tip:** Label at least 200-300 images. More = better model!

### 🔵 STEP 4: Train Initial Model

```
→ Select [6] Train Initial Model
→ Confirm the number of epochs (default: 100)
→ Training starts...
```

⏱️ **Duration:** 30 minutes - 2 hours depending on GPU

### 🔵 STEP 5: Auto-Annotation ⭐

```
→ Select [4] Start Auto Annotation
→ Confirm
→ Model automatically labels all images
```

**What happens:**
- ✅ Valid images → `auto_annotations/images/` and `labels/`
- ❌ Invalid images → `auto_annotations/unvalid/`
- 📊 Statistics are displayed

### 🔵 STEP 6: Prepare Final Dataset

```
→ Select [5] Prepare Final Dataset
→ Manual + Auto annotations are merged
→ Split into Train/Val/Test
```

### 🔵 STEP 7: Train Final Model

```
→ Select [7] Train Final Model
→ Enter number of epochs (e.g., 150)
→ Training starts...
```

🎉 **Done!** Best model saved at: `projects/PROJECT_NAME/models/`

---

## 🤖 How Auto-Annotation Works

### Workflow

```
┌──────────────────┐
│  Trained Model   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────┐
│   Raw Images     │────▶│  Model Analysis │
│   (1000+ images) │     │                 │
└──────────────────┘     └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
           ┌───────────────┐          ┌───────────────┐
           │   VALID ✅     │          │  INVALID ❌    │
           │               │          │               │
           │ • Min detection│          │ • Few detects │
           │   met          │          │ • Low confid. │
           │ • High confid. │          │               │
           └───────┬───────┘          └───────┬───────┘
                   │                          │
                   ▼                          ▼
           auto_annotations/           auto_annotations/
           ├── images/                 └── unvalid/
           └── labels/
```

### Minimum Detection Setting

This setting determines **how many objects must be detected** in an image for it to be valid:

| Min Detection | Use Case |
|---------------|----------|
| `1` | Single object detection (car, dog, etc.) |
| `2` | Paired objects (two eyes, etc.) |
| `3+` | Multiple objects required |

**To Change This Setting:**
```
→ Select [10] Project Settings
→ Select [3] Change annotation settings
→ Enter Min Detections value
```

### Confidence Threshold

- **0.5** (default): Medium confidence, more detections
- **0.7**: High confidence, fewer but more accurate detections
- **0.3**: Low confidence, many detections (may be noisy)

---

## ❓ FAQ

### "Select a project first!" error
➡️ First create a project: **[1] Project Management** → **[1] Create New Project**

### How many images should I label?
➡️ Minimum **200-300 images** recommended. More = better model.

### Does it work without GPU?
➡️ Yes, but training takes much longer. 100 epochs on CPU = 5-10 hours.

### What does Min Detection do?
➡️ Sets minimum objects required per image. E.g., `min=2` means images with only 1 detection go to `unvalid/` folder.

### What if auto-labels are wrong?
➡️ Check `auto_annotations/visualizations/` folder for visual review. Delete incorrect ones.

### Where is the model saved?
➡️ `projects/PROJECT_NAME/models/latest_model.pt`

### I want to train on a server
➡️ Use **[8] Server Export (ZIP)** to export all files as a ZIP.

---

## 📁 Project Structure

```
SmartAnnotator/
├── main.py                 # Main menu
├── config.py               # Global settings
├── project_manager.py      # Project management
├── annotation_tool.py      # Manual annotation GUI
├── auto_annotate.py        # Auto annotation engine
├── train_model.py          # Model training
├── prepare_final_dataset.py# Dataset preparation
├── utils.py                # Helper functions
├── requirements.txt        # Python dependencies
│
└── projects/               # All projects stored here
    └── my_project/
        ├── project_config.json  # Project settings
        ├── classes.txt          # Class definitions
        ├── raw_images/          # Raw unlabeled images
        ├── manual_annotations/  # Manual labels
        │   ├── images/
        │   └── labels/
        ├── auto_annotations/    # Auto-generated labels
        │   ├── images/          # Valid images
        │   ├── labels/          # Label files
        │   ├── visualizations/  # Visual previews
        │   └── unvalid/         # Invalid images
        ├── final_dataset/       # Ready for training
        │   ├── train/
        │   ├── val/
        │   ├── test/
        │   └── dataset.yaml
        └── models/              # Trained models
            └── latest_model.pt
```

---

## ⚙️ Project Configuration (project_config.json)

```json
{
  "project_name": "car_detection",
  "model": {
    "family": "yolov11",
    "task": "detection",
    "size": "m",
    "weights": "yolo11m.pt"
  },
  "classes": {
    "0": "car",
    "1": "person"
  },
  "split": "auto",
  "training": {
    "epochs": 100,
    "batch_size": 16,
    "image_size": 640
  },
  "annotation": {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "min_detections": 1
  }
}
```

---

## 🏷️ Label Formats

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

---

## 📊 Model Sizes

| Size | Name | Speed | Accuracy | Use Case |
|------|------|-------|----------|----------|
| `n` | Nano | ⚡⚡⚡⚡⚡ | ⭐ | Mobile/Edge |
| `s` | Small | ⚡⚡⚡⚡ | ⭐⭐ | Fast inference |
| `m` | Medium | ⚡⚡⚡ | ⭐⭐⭐ | Balanced (recommended) |
| `l` | Large | ⚡⚡ | ⭐⭐⭐⭐ | High accuracy |
| `x` | XLarge | ⚡ | ⭐⭐⭐⭐⭐ | Maximum accuracy |

---

## 🤝 Contributing

1. Fork the repo
2. Create feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push (`git push origin feature/NewFeature`)
5. Open Pull Request

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO implementation
- [Gradio](https://gradio.app/) - GUI framework
- [OpenCV](https://opencv.org/) - Image processing

---

**Made with ❤️ for the computer vision community**
