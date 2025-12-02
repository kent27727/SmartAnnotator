"""
Server Training Configuration
==============================
Project: eyeline
Task: SEGMENTATION
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).parent
PROJECTS_DIR = PROJECT_DIR
FINAL_DATASET_DIR = PROJECT_DIR / "final_dataset"
MODELS_DIR = PROJECT_DIR / "models"
TEMP_DIR = PROJECT_DIR / "temp"
MANUAL_ANNOTATIONS_DIR = PROJECT_DIR / "final_dataset"

# Model Configuration
MODEL_TASK = "segmentation"  # detection, segmentation, classification
MODEL_FAMILY = "yolov11"  # yolov11, yolov8, resnet
YOLO_MODEL_SIZE = "m"
YOLO_SEG_MODEL = "yolo11m-seg.pt"  # Used by train_model.py

# Training
INITIAL_TRAINING_EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
LEARNING_RATE = 0.01

# Minimum samples
MIN_SAMPLES_FOR_TRAINING = 20
RECOMMENDED_SAMPLES = 100

# Classes
CLASSES = {'0': 'eyefineline'}
NUM_CLASSES = 1

# Augmentation
AUGMENTATION_ENABLED = True
AUGMENTATION_CONFIG = {
    "horizontal_flip": True,
    "vertical_flip": False,
    "rotation_range": 15,
}

# Split (already applied)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def create_directories():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return True

def get_dataset_yaml_path():
    return FINAL_DATASET_DIR / "dataset.yaml"

def print_config():
    print("Task:", MODEL_TASK)
    print("Model:", YOLO_SEG_MODEL)
    print("Epochs:", INITIAL_TRAINING_EPOCHS)
    print("Batch Size:", BATCH_SIZE)
