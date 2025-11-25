"""
Auto-Annotation Tool - Model Training
======================================
Train YOLOv11 segmentation model with manually labeled data
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import shutil

# Ultralytics import
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics not installed!")
    print("Install with: pip install ultralytics")
    sys.exit(1)

import config
from utils import (
    create_dataset_yaml, 
    split_dataset, 
    count_images,
    roboflow_to_yolo
)


class ModelTrainer:
    """YOLO model trainer for detection/segmentation tasks"""
    
    def __init__(self):
        self.model = None
        self.trained_model_path = None
        
    def prepare_training_data(self, 
                              roboflow_export_dir: Path = None,
                              images_dir: Path = None,
                              labels_dir: Path = None):
        """
        Prepare training data
        
        Args:
            roboflow_export_dir: Directory exported from Roboflow
            images_dir: Manually specified images directory
            labels_dir: Manually specified labels directory
        """
        print("\n" + "="*50)
        print("📦 PREPARING TRAINING DATA")
        print("="*50)
        
        # Create directories
        config.create_directories()
        
        # Convert Roboflow export if exists
        if roboflow_export_dir and roboflow_export_dir.exists():
            print(f"\n📥 Processing Roboflow export: {roboflow_export_dir}")
            
            # Find and copy images
            temp_images = config.TEMP_DIR / "images"
            temp_labels = config.TEMP_DIR / "labels"
            temp_images.mkdir(parents=True, exist_ok=True)
            temp_labels.mkdir(parents=True, exist_ok=True)
            
            # Analyze Roboflow structure
            self._process_roboflow_export(roboflow_export_dir, temp_images, temp_labels)
            
            images_dir = temp_images
            labels_dir = temp_labels
        
        if images_dir is None or labels_dir is None:
            raise ValueError("images_dir and labels_dir must be specified!")
        
        # Check image count
        n_images = count_images(images_dir)
        n_labels = len(list(labels_dir.glob("*.txt")))
        
        print(f"\n📊 Found data:")
        print(f"   Images: {n_images}")
        print(f"   Label files: {n_labels}")
        
        if n_labels < config.MIN_SAMPLES_FOR_TRAINING:
            print(f"\n⚠️ WARNING: At least {config.MIN_SAMPLES_FOR_TRAINING} labeled samples required!")
            print(f"   Current: {n_labels} samples.")
            print(f"   Recommended: {config.RECOMMENDED_SAMPLES} samples")
            
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Operation cancelled.")
                return False
        
        # Split dataset
        print("\n📂 Splitting dataset (train/val/test)...")
        split_dataset(
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=config.FINAL_DATASET_DIR,
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
            test_ratio=config.TEST_RATIO
        )
        
        # Create dataset YAML
        class_names = list(config.CLASSES.values())
        create_dataset_yaml(
            dataset_dir=config.FINAL_DATASET_DIR,
            class_names=class_names
        )
        
        print("\n✅ Training data ready!")
        return True
    
    def _process_roboflow_export(self, 
                                  roboflow_dir: Path,
                                  images_out: Path,
                                  labels_out: Path):
        """Process Roboflow export structure"""
        
        # Check different Roboflow export structures
        possible_structures = [
            # YOLOv8 format
            {"images": "train/images", "labels": "train/labels"},
            {"images": "valid/images", "labels": "valid/labels"},
            {"images": "test/images", "labels": "test/labels"},
            # Flat structure
            {"images": "images", "labels": "labels"},
            {"images": "", "labels": ""},  # Root level
        ]
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Find all images and labels
        all_images = []
        all_labels = []
        
        for ext in image_extensions:
            all_images.extend(roboflow_dir.rglob(f"*{ext}"))
        all_labels.extend(roboflow_dir.rglob("*.txt"))
        
        # Filter classes.txt and data.yaml
        all_labels = [l for l in all_labels 
                     if l.name not in ['classes.txt', 'data.yaml', 'README.txt', 'README.roboflow.txt']]
        
        print(f"   Found images: {len(all_images)}")
        print(f"   Found label files: {len(all_labels)}")
        
        # Copy
        for img in all_images:
            shutil.copy(img, images_out / img.name)
        
        for lbl in all_labels:
            shutil.copy(lbl, labels_out / lbl.name)
        
        # COCO format check
        coco_files = list(roboflow_dir.rglob("*annotations*.json"))
        if coco_files and len(all_labels) == 0:
            print("   COCO format detected, converting...")
            roboflow_to_yolo(roboflow_dir, labels_out)
    
    def train(self, 
              epochs: int = None,
              batch_size: int = None,
              image_size: int = None,
              resume: bool = False):
        """
        Train YOLOv11 segmentation model
        
        Args:
            epochs: Number of epochs
            batch_size: Batch size
            image_size: Image size
            resume: Resume from previous training
        """
        epochs = epochs or config.INITIAL_TRAINING_EPOCHS
        batch_size = batch_size or config.BATCH_SIZE
        image_size = image_size or config.IMAGE_SIZE
        
        print("\n" + "="*50)
        print("🚀 MODEL TRAINING STARTING")
        print("="*50)
        
        # Dataset YAML path
        dataset_yaml = config.get_dataset_yaml_path()
        
        if not dataset_yaml.exists():
            raise FileNotFoundError(
                f"Dataset YAML not found: {dataset_yaml}\n"
                "Run prepare_training_data() first!"
            )
        
        # Load model
        print(f"\n📥 Loading model: {config.YOLO_SEG_MODEL}")
        self.model = YOLO(config.YOLO_SEG_MODEL)
        
        # Training parameters
        print(f"\n⚙️ Training parameters:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Image Size: {image_size}")
        print(f"   Dataset: {dataset_yaml}")
        
        # Start training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = config.MODELS_DIR / "yolo_training"
        run_name = f"train_{timestamp}"
        
        print(f"\n🏃 Training starting...")
        print("-" * 50)
        
        results = self.model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            project=str(project_name),
            name=run_name,
            exist_ok=True,
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=42,
            deterministic=True,
            resume=resume,
            # Augmentation
            augment=config.AUGMENTATION_ENABLED,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=config.AUGMENTATION_CONFIG.get('rotation_range', 0),
            flipud=0.0 if not config.AUGMENTATION_CONFIG.get('vertical_flip', False) else 0.5,
            fliplr=0.5 if config.AUGMENTATION_CONFIG.get('horizontal_flip', True) else 0.0,
            scale=0.5,
            # Early stopping
            patience=20,
        )
        
        # Save best model
        best_model_path = project_name / run_name / "weights" / "best.pt"
        last_model_path = project_name / run_name / "weights" / "last.pt"
        
        if best_model_path.exists():
            self.trained_model_path = best_model_path
            
            # Create shortcut
            latest_model = config.MODELS_DIR / "latest_model.pt"
            shutil.copy(best_model_path, latest_model)
            
            # Print results prominently
            print("\n" + "="*60)
            print("✅ TRAINING COMPLETE!")
            print("="*60)
            print(f"\n📁 MODEL FILES:")
            print(f"   ┌─────────────────────────────────────────────────────────")
            print(f"   │ 🏆 BEST MODEL:")
            print(f"   │    {best_model_path.absolute()}")
            print(f"   │")
            print(f"   │ 📄 LAST MODEL (last epoch):")
            print(f"   │    {last_model_path.absolute()}")
            print(f"   │")
            print(f"   │ 🔗 SHORTCUT:")
            print(f"   │    {latest_model.absolute()}")
            print(f"   └─────────────────────────────────────────────────────────")
            print(f"\n💡 You can download and use best.pt file!")
            print("="*60)
        else:
            print("\n⚠️ best.pt not found!")
        
        return results
    
    def evaluate(self, model_path: Path = None):
        """Evaluate model performance"""
        
        if model_path is None:
            model_path = config.MODELS_DIR / "latest_model.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"\n📊 Evaluating model: {model_path}")
        
        model = YOLO(str(model_path))
        dataset_yaml = config.get_dataset_yaml_path()
        
        results = model.val(
            data=str(dataset_yaml),
            split='test',
            verbose=True
        )
        
        print("\n" + "="*50)
        print("📈 EVALUATION RESULTS")
        print("="*50)
        
        # Print metrics
        if hasattr(results, 'seg'):
            print(f"Segmentation mAP50: {results.seg.map50:.4f}")
            print(f"Segmentation mAP50-95: {results.seg.map:.4f}")
        
        if hasattr(results, 'box'):
            print(f"Box mAP50: {results.box.map50:.4f}")
            print(f"Box mAP50-95: {results.box.map:.4f}")
        
        return results


def main():
    """Main training pipeline"""
    
    print("\n" + "="*60)
    print("🎯 AUTO-ANNOTATION TOOL - MODEL TRAINING")
    print("="*60)
    
    trainer = ModelTrainer()
    
    # Get Roboflow export directory from user
    print("\n📂 Specify your Roboflow export directory.")
    print("   (YOLO format or COCO format)")
    
    roboflow_dir = input("\nRoboflow export directory (or Enter for default): ").strip()
    
    if not roboflow_dir:
        roboflow_dir = config.MANUAL_ANNOTATIONS_DIR
        print(f"   Using default: {roboflow_dir}")
    
    roboflow_dir = Path(roboflow_dir)
    
    if not roboflow_dir.exists():
        print(f"\n❌ Directory not found: {roboflow_dir}")
        print("\n💡 Export data from Roboflow in this format:")
        print("   - Format: YOLOv8 or COCO")
        print("   - Segmentation masks must be included")
        print(f"\n   Place export at: {config.MANUAL_ANNOTATIONS_DIR}")
        return
    
    # Prepare data
    success = trainer.prepare_training_data(roboflow_export_dir=roboflow_dir)
    
    if not success:
        return
    
    # Start training
    print("\n" + "-"*50)
    response = input("Start training? (y/n): ")
    
    if response.lower() == 'y':
        trainer.train()
        
        # Evaluation
        print("\n" + "-"*50)
        response = input("Run model evaluation? (y/n): ")
        
        if response.lower() == 'y':
            trainer.evaluate()
    
    print("\n✅ Process complete!")
    print(f"\n📌 Next step: Run auto_annotate.py for automatic labeling.")


if __name__ == "__main__":
    main()
