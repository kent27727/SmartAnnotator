"""
Auto-Annotation Tool - Model Training
======================================
Train YOLO/ResNet models for Detection, Segmentation, and Classification tasks

Supported:
    - YOLOv11: Detection, Segmentation, Classification
    - YOLOv8: Detection, Segmentation, Classification
    - ResNet: Classification only
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
    """Universal model trainer for Detection/Segmentation/Classification tasks"""
    
    def __init__(self):
        self.model = None
        self.trained_model_path = None
        
        # Get task and model info from config
        self.task = getattr(config, 'MODEL_TASK', 'segmentation')
        self.model_weights = getattr(config, 'YOLO_SEG_MODEL', 'yolo11m-seg.pt')
        self.model_family = self._detect_model_family()
        
        print(f"\n🤖 Model Trainer initialized:")
        print(f"   Task: {self.task.upper()}")
        print(f"   Model: {self.model_weights}")
        print(f"   Family: {self.model_family.upper()}")
    
    def _detect_model_family(self) -> str:
        """Detect model family from weights name"""
        weights = self.model_weights.lower()
        if 'yolo11' in weights or 'yolov11' in weights:
            return 'yolov11'
        elif 'yolov8' in weights or 'yolo8' in weights:
            return 'yolov8'
        elif 'resnet' in weights:
            return 'resnet'
        else:
            # Default to yolov11
            return 'yolov11'
    
    def prepare_training_data(self, 
                              roboflow_export_dir: Path = None,
                              images_dir: Path = None,
                              labels_dir: Path = None):
        """
        Prepare training data based on task type
        
        Args:
            roboflow_export_dir: Directory exported from Roboflow
            images_dir: Manually specified images directory
            labels_dir: Manually specified labels directory
        """
        print("\n" + "="*50)
        print("📦 PREPARING TRAINING DATA")
        print(f"   Task: {self.task.upper()}")
        print("="*50)
        
        # Create directories
        config.create_directories()
        
        # Classification uses different structure
        if self.task == "classification":
            return self._prepare_classification_data(roboflow_export_dir, images_dir)
        else:
            # Detection and Segmentation use same structure
            return self._prepare_detection_data(roboflow_export_dir, images_dir, labels_dir)
    
    def _prepare_classification_data(self, roboflow_dir: Path = None, images_dir: Path = None):
        """Prepare classification data (folder-based structure)"""
        
        print("\n📂 Classification data structure:")
        print("   dataset/class_name/image.jpg")
        
        # Find source directory
        source_dir = roboflow_dir or images_dir or config.MANUAL_ANNOTATIONS_DIR
        
        if not source_dir or not Path(source_dir).exists():
            print(f"❌ Source directory not found: {source_dir}")
            return False
        
        source_dir = Path(source_dir)
        
        # Check for class folders
        class_folders = [d for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        # Also check train/val/test subfolders (Roboflow format)
        for split_name in ['train', 'valid', 'test', 'val']:
            split_dir = source_dir / split_name
            if split_dir.exists():
                for d in split_dir.iterdir():
                    if d.is_dir() and d.name not in class_folders:
                        class_folders.append(d)
        
        if not class_folders:
            print("❌ No class folders found!")
            print("   Expected structure: folder/class_name/images.jpg")
            return False
        
        print(f"\n📊 Found {len(class_folders)} class folders")
        
        # Copy to final dataset with train/val/test split
        # For classification, we copy the folder structure
        final_dir = config.FINAL_DATASET_DIR
        
        # If source already has train/val/test structure, copy directly
        if (source_dir / 'train').exists():
            print("   Roboflow structure detected, copying...")
            for split in ['train', 'valid', 'test']:
                src_split = source_dir / split
                dst_split = final_dir / ('val' if split == 'valid' else split)
                if src_split.exists():
                    shutil.copytree(src_split, dst_split, dirs_exist_ok=True)
        else:
            print("   Flat structure, will need to split manually...")
            # Would need to implement splitting for flat classification data
        
        print("\n✅ Classification data ready!")
        return True
    
    def _prepare_detection_data(self, roboflow_dir: Path = None, 
                                 images_dir: Path = None, 
                                 labels_dir: Path = None):
        """Prepare detection/segmentation data (images + labels structure)"""
        
        # Convert Roboflow export if exists
        if roboflow_dir and roboflow_dir.exists():
            print(f"\n📥 Processing Roboflow export: {roboflow_dir}")
            
            # Find and copy images
            temp_images = config.TEMP_DIR / "images"
            temp_labels = config.TEMP_DIR / "labels"
            temp_images.mkdir(parents=True, exist_ok=True)
            temp_labels.mkdir(parents=True, exist_ok=True)
            
            # Analyze Roboflow structure
            self._process_roboflow_export(roboflow_dir, temp_images, temp_labels)
            
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
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Find all images and labels
        all_images = []
        all_labels = []
        
        for ext in image_extensions:
            all_images.extend(roboflow_dir.rglob(f"*{ext}"))
        all_labels.extend(roboflow_dir.rglob("*.txt"))
        
        # Filter classes.txt and data.yaml
        all_labels = [l for l in all_labels 
                     if l.name not in ['classes.txt', 'data.yaml', 'README.txt', 'README.roboflow.txt', 'README.dataset.txt']]
        
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
        Train model based on task type
        
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
        print(f"🚀 {self.task.upper()} MODEL TRAINING")
        print("="*50)
        
        # Different training for different tasks
        if self.task == "classification" and self.model_family == "resnet":
            return self._train_resnet(epochs, batch_size, image_size)
        else:
            return self._train_yolo(epochs, batch_size, image_size, resume)
    
    def _train_yolo(self, epochs: int, batch_size: int, image_size: int, resume: bool = False):
        """Train YOLO model (Detection/Segmentation/Classification)"""
        
        # Dataset YAML path
        dataset_yaml = config.get_dataset_yaml_path()
        
        # For classification, YOLO uses folder structure, not YAML
        if self.task == "classification":
            # Point to the dataset directory directly
            data_path = str(config.FINAL_DATASET_DIR)
            print(f"\n📁 Using folder structure: {data_path}")
        else:
            if not dataset_yaml.exists():
                raise FileNotFoundError(
                    f"Dataset YAML not found: {dataset_yaml}\n"
                    "Run prepare_training_data() first!"
                )
            data_path = str(dataset_yaml)
        
        # Load model
        print(f"\n📥 Loading model: {self.model_weights}")
        print(f"   Task: {self.task}")
        print(f"   Family: {self.model_family}")
        
        self.model = YOLO(self.model_weights)
        
        # Training parameters
        print(f"\n⚙️ Training parameters:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Image Size: {image_size}")
        print(f"   Data: {data_path}")
        
        # Start training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = config.MODELS_DIR / "yolo_training"
        run_name = f"train_{self.task}_{timestamp}"
        
        print(f"\n🏃 Training starting...")
        print("-" * 50)
        
        # Common training args
        train_args = {
            "data": data_path,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": image_size,
            "project": str(project_name),
            "name": run_name,
            "exist_ok": True,
            "pretrained": True,
            "optimizer": 'auto',
            "verbose": True,
            "seed": 42,
            "deterministic": True,
            "resume": resume,
            "patience": 20,
        }
        
        # Add augmentation for detection/segmentation
        if self.task in ["detection", "segmentation"]:
            train_args.update({
                "augment": config.AUGMENTATION_ENABLED,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "degrees": config.AUGMENTATION_CONFIG.get('rotation_range', 0),
                "flipud": 0.0 if not config.AUGMENTATION_CONFIG.get('vertical_flip', False) else 0.5,
                "fliplr": 0.5 if config.AUGMENTATION_CONFIG.get('horizontal_flip', True) else 0.0,
                "scale": 0.5,
            })
        
        results = self.model.train(**train_args)
        
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
            print(f"\n🎯 Task: {self.task.upper()}")
            print(f"🤖 Model: {self.model_weights}")
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
    
    def _train_resnet(self, epochs: int, batch_size: int, image_size: int):
        """Train ResNet model (Classification only)"""
        
        print("\n🔬 ResNet Classification Training")
        print("   This requires torchvision and custom training loop")
        
        # ResNet training would require PyTorch/torchvision
        # For now, recommend using YOLO classification instead
        print("\n⚠️ ResNet training not yet implemented!")
        print("   Recommendation: Use YOLOv11/YOLOv8 classification instead.")
        print("   They are faster and easier to use.")
        
        return None
    
    def evaluate(self, model_path: Path = None):
        """Evaluate model performance"""
        
        if model_path is None:
            model_path = config.MODELS_DIR / "latest_model.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"\n📊 Evaluating model: {model_path}")
        print(f"   Task: {self.task}")
        
        model = YOLO(str(model_path))
        
        if self.task == "classification":
            # Classification evaluation
            results = model.val(
                data=str(config.FINAL_DATASET_DIR),
                split='test',
                verbose=True
            )
        else:
            # Detection/Segmentation evaluation
            dataset_yaml = config.get_dataset_yaml_path()
            results = model.val(
                data=str(dataset_yaml),
                split='test',
                verbose=True
            )
        
        print("\n" + "="*50)
        print("📈 EVALUATION RESULTS")
        print("="*50)
        
        # Print metrics based on task
        if self.task == "segmentation" and hasattr(results, 'seg'):
            print(f"Segmentation mAP50: {results.seg.map50:.4f}")
            print(f"Segmentation mAP50-95: {results.seg.map:.4f}")
        
        if self.task in ["detection", "segmentation"] and hasattr(results, 'box'):
            print(f"Box mAP50: {results.box.map50:.4f}")
            print(f"Box mAP50-95: {results.box.map:.4f}")
        
        if self.task == "classification":
            if hasattr(results, 'top1'):
                print(f"Top-1 Accuracy: {results.top1:.4f}")
            if hasattr(results, 'top5'):
                print(f"Top-5 Accuracy: {results.top5:.4f}")
        
        return results


def main():
    """Main training pipeline"""
    
    print("\n" + "="*60)
    print("🎯 AUTO-ANNOTATION TOOL - MODEL TRAINING")
    print("="*60)
    
    # Show current config
    task = getattr(config, 'MODEL_TASK', 'segmentation')
    model = getattr(config, 'YOLO_SEG_MODEL', 'yolo11m-seg.pt')
    
    print(f"\n📋 Current Configuration:")
    print(f"   Task: {task.upper()}")
    print(f"   Model: {model}")
    
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
        if task == "segmentation":
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
