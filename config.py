"""
Auto-Annotation Tool - Configuration
=====================================
All project settings are configured in this file.
For project-based settings, use project_manager.py.
"""

import os
import json
from pathlib import Path

# ============================================
# PATHS - File Paths (Default)
# ============================================

# Main project directory
PROJECT_DIR = Path(__file__).parent

# Projects directory (new system)
PROJECTS_DIR = PROJECT_DIR / "projects"

# Raw image dataset (unlabeled) - default
RAW_IMAGES_DIR = Path(r"C:\Users\enes1\OneDrive\Desktop\darkcircledataset")

# Manual labeled data (legacy compatibility)
MANUAL_ANNOTATIONS_DIR = PROJECT_DIR / "manual_annotations"

# Auto labeled data (legacy)
AUTO_ANNOTATIONS_DIR = PROJECT_DIR / "auto_annotations"

# Trained models (legacy)
MODELS_DIR = PROJECT_DIR / "models"

# Final dataset (legacy)
FINAL_DATASET_DIR = PROJECT_DIR / "final_dataset"

# Temporary files
TEMP_DIR = PROJECT_DIR / "temp"

# ============================================
# ACTIVE PROJECT MANAGEMENT
# ============================================

_active_project = None
_active_project_config = None

def set_active_project(project_name: str):
    """Set the active project"""
    global _active_project, _active_project_config
    global RAW_IMAGES_DIR, MANUAL_ANNOTATIONS_DIR, AUTO_ANNOTATIONS_DIR
    global MODELS_DIR, FINAL_DATASET_DIR, YOLO_MODEL_SIZE, YOLO_SEG_MODEL
    global CLASSES, NUM_CLASSES, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
    global TRAIN_RATIO, VAL_RATIO, TEST_RATIO, BATCH_SIZE, IMAGE_SIZE
    global INITIAL_TRAINING_EPOCHS
    
    project_path = PROJECTS_DIR / project_name
    config_file = project_path / "project_config.json"
    
    if not config_file.exists():
        print(f"❌ Project not found: {project_name}")
        return False
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    _active_project = project_name
    _active_project_config = config
    
    # Update directories
    MANUAL_ANNOTATIONS_DIR = project_path / "manual_annotations"
    AUTO_ANNOTATIONS_DIR = project_path / "auto_annotations"
    MODELS_DIR = project_path / "models"
    FINAL_DATASET_DIR = project_path / "final_dataset"
    
    # Update model settings
    model_config = config.get("model", {})
    YOLO_MODEL_SIZE = model_config.get("size", "m")
    YOLO_SEG_MODEL = model_config.get("weights", f"yolo11{YOLO_MODEL_SIZE}-seg.pt")
    
    # Update class settings
    CLASSES = {int(k): v for k, v in config.get("classes", {0: "dark_circle"}).items()}
    NUM_CLASSES = len(CLASSES)
    
    # Annotation settings
    ann_config = config.get("annotation", {})
    CONFIDENCE_THRESHOLD = ann_config.get("confidence_threshold", 0.5)
    IOU_THRESHOLD = ann_config.get("iou_threshold", 0.45)
    
    # Training settings
    train_config = config.get("training", {})
    BATCH_SIZE = train_config.get("batch_size", 16)
    IMAGE_SIZE = train_config.get("image_size", 640)
    INITIAL_TRAINING_EPOCHS = train_config.get("epochs", 100)
    
    # Split settings
    split = config.get("split", "auto")
    if split != "auto":
        TRAIN_RATIO = split.get("train", 0.8)
        VAL_RATIO = split.get("val", 0.1)
        TEST_RATIO = split.get("test", 0.1)
    
    print(f"✅ Active project: {project_name}")
    return True

def get_active_project():
    """Return active project name"""
    return _active_project

def get_active_project_config():
    """Return active project configuration"""
    return _active_project_config

def get_active_project_path():
    """Return active project directory"""
    if _active_project:
        return PROJECTS_DIR / _active_project
    return None

# ============================================
# MODEL SETTINGS
# ============================================

# YOLOv11 model size: 'n', 's', 'm', 'l', 'x'
# n = nano (fastest, least accurate)
# x = extra large (slowest, most accurate)
YOLO_MODEL_SIZE = "m"  # Medium - balanced

# Segmentation model name
YOLO_SEG_MODEL = f"yolo11{YOLO_MODEL_SIZE}-seg.pt"

# ============================================
# TRAINING SETTINGS
# ============================================

# Minimum samples for initial training
MIN_SAMPLES_FOR_TRAINING = 50

# Recommended number of samples
RECOMMENDED_SAMPLES = 200

# Training epochs
INITIAL_TRAINING_EPOCHS = 100
FINE_TUNING_EPOCHS = 50

# Batch size (adjust based on GPU memory)
BATCH_SIZE = 16

# Image size
IMAGE_SIZE = 640

# Learning rate
LEARNING_RATE = 0.01

# ============================================
# AUTO-ANNOTATION SETTINGS
# ============================================

# Minimum confidence threshold for auto annotation
CONFIDENCE_THRESHOLD = 0.5

# IoU threshold for NMS
IOU_THRESHOLD = 0.45

# Low confidence threshold for review flag
LOW_CONFIDENCE_THRESHOLD = 0.7

# ============================================
# CLASS DEFINITIONS
# ============================================

# Classes to be labeled
CLASSES = {
    0: "dark_circle"
}

# Number of classes
NUM_CLASSES = len(CLASSES)

# ============================================
# DATA AUGMENTATION
# ============================================

AUGMENTATION_ENABLED = True

AUGMENTATION_CONFIG = {
    "horizontal_flip": True,
    "vertical_flip": False,
    "rotation_range": 15,
    "brightness_range": (0.8, 1.2),
    "contrast_range": (0.8, 1.2),
    "scale_range": (0.8, 1.2),
}

# ============================================
# EXPORT SETTINGS
# ============================================

# Train/Val/Test split ratios
TRAIN_RATIO = 0.8   # 80% training
VAL_RATIO = 0.1     # 10% validation
TEST_RATIO = 0.1    # 10% test

# ============================================
# UTILITY FUNCTIONS
# ============================================

def create_directories():
    """Create required directories"""
    dirs = [
        MANUAL_ANNOTATIONS_DIR,
        AUTO_ANNOTATIONS_DIR,
        MODELS_DIR,
        FINAL_DATASET_DIR,
        TEMP_DIR,
        FINAL_DATASET_DIR / "train" / "images",
        FINAL_DATASET_DIR / "train" / "labels",
        FINAL_DATASET_DIR / "val" / "images",
        FINAL_DATASET_DIR / "val" / "labels",
        FINAL_DATASET_DIR / "test" / "images",
        FINAL_DATASET_DIR / "test" / "labels",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        
    print("✅ All directories created!")
    return True

def get_dataset_yaml_path():
    """Return dataset YAML file path"""
    return FINAL_DATASET_DIR / "dataset.yaml"

def print_config():
    """Print current configuration"""
    print("=" * 50)
    print("AUTO-ANNOTATION CONFIG")
    print("=" * 50)
    print(f"📁 Raw Images: {RAW_IMAGES_DIR}")
    print(f"📁 Project Dir: {PROJECT_DIR}")
    print(f"🤖 YOLO Model: {YOLO_SEG_MODEL}")
    print(f"📊 Image Size: {IMAGE_SIZE}")
    print(f"🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"📈 Training Epochs: {INITIAL_TRAINING_EPOCHS}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
    create_directories()
