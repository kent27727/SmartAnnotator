"""
Auto-Annotation & Training Tool
===============================
Main control script - Manage all operations from here

Usage:
    python main.py

Features:
    - Multi-project support
    - YOLOv11, YOLOv8, ResNet model selection
    - Classification, Detection, Segmentation task selection
    - Automatic or manual train/val/test ratios
"""

import os
import sys
from pathlib import Path
import shutil
import zipfile

# Fix Windows console encoding for Unicode/emoji characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, TypeError):
        # Python < 3.7 or reconfigure not available
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

# Add project directory to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

import config
from project_manager import ProjectManager, SUPPORTED_MODELS, TASK_DESCRIPTIONS


# Global project manager
project_manager = ProjectManager()


def print_banner():
    """Print banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     🎯 AUTO-ANNOTATION & TRAINING TOOL                       ║
    ║     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                        ║
    ║     YOLOv11 / YOLOv8 / ResNet - Multi-Model Support          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_active_project():
    """Show active project info"""
    if project_manager.current_project:
        cfg = project_manager.project_config
        model = cfg["model"]
        print(f"\n🎯 Active Project: {project_manager.current_project}")
        print(f"   Model: {model['family'].upper()} {model['task']} ({model['size']})")
        print(f"   Classes: {list(cfg['classes'].values())}")
    else:
        print("\n⚠️ No active project! Create or select a project first.")


def print_project_status():
    """Show active project status"""
    if not project_manager.current_project:
        print("\n❌ Select a project first!")
        return
    
    project_path = project_manager.get_project_path()
    cfg = project_manager.project_config
    
    print("\n" + "="*50)
    print("📊 PROJECT STATUS")
    print("="*50)
    
    print(f"\n📁 Project: {project_manager.current_project}")
    print(f"   Directory: {project_path}")
    
    # Model info
    model = cfg["model"]
    print(f"\n🤖 Model: {SUPPORTED_MODELS[model['family']]['name']}")
    print(f"   Task: {model['task'].upper()}")
    print(f"   Size: {model['size'].upper()}")
    print(f"   Weights: {model['weights']}")
    
    # Classes
    print(f"\n🏷️ Classes: {list(cfg['classes'].values())}")
    
    # Raw images
    raw_dir = project_path / "raw_images"
    raw_count = 0
    if raw_dir.exists():
        for ext in ['.jpg', '.jpeg', '.png']:
            raw_count += len(list(raw_dir.glob(f"*{ext}")))
    print(f"\n📷 Raw Images: {raw_count}")
    
    # Manual annotations
    manual_dir = project_path / "manual_annotations"
    if cfg["model"]["task"] == "classification":
        manual_count = sum(len(list((manual_dir / c).glob("*.jpg"))) 
                         for c in cfg["classes"].values() 
                         if (manual_dir / c).exists())
    else:
        labels_dir = manual_dir / "labels"
        manual_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
    print(f"📝 Manual Annotations: {manual_count}")
    
    # Auto annotations
    auto_dir = project_path / "auto_annotations"
    if cfg["model"]["task"] == "classification":
        auto_count = sum(len(list((auto_dir / c).glob("*.jpg"))) 
                        for c in cfg["classes"].values() 
                        if (auto_dir / c).exists())
    else:
        labels_dir = auto_dir / "labels"
        auto_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
    print(f"🤖 Auto Annotations: {auto_count}")
    
    # Unvalid
    unvalid_dir = auto_dir / "unvalid"
    unvalid_count = len(list(unvalid_dir.glob("*.jpg"))) if unvalid_dir.exists() else 0
    print(f"❌ Invalid (unvalid): {unvalid_count}")
    
    # Trained model
    models_dir = project_path / "models"
    model_exists = (models_dir / "latest_model.pt").exists()
    print(f"\n🧠 Trained Model: {'✅ Available' if model_exists else '❌ Not found'}")
    
    # Final dataset
    final_dir = project_path / "final_dataset"
    train_dir = final_dir / "train" / "images"
    train_count = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
    val_dir = final_dir / "val" / "images"
    val_count = len(list(val_dir.glob("*.jpg"))) if val_dir.exists() else 0
    test_dir = final_dir / "test" / "images"
    test_count = len(list(test_dir.glob("*.jpg"))) if test_dir.exists() else 0
    
    print(f"\n📦 Final Dataset:")
    print(f"   Train: {train_count}")
    print(f"   Val: {val_count}")
    print(f"   Test: {test_count}")
    
    # Split settings
    split = cfg.get("split", "auto")
    if split == "auto":
        print(f"\n📊 Split: Automatic")
    else:
        print(f"\n📊 Split: Train={split['train']}, Val={split['val']}, Test={split['test']}")
    
    print("="*50)


def project_menu():
    """Project management menu"""
    print("\n" + "="*50)
    print("📁 PROJECT MANAGEMENT")
    print("="*50)
    print("""
    [1] 🆕 Create New Project
    [2] 📂 Load Existing Project
    [3] 📋 List Projects
    [4] ⚙️ Edit Project Settings
    [0] ↩️ Back to Main Menu
    """)
    
    choice = input("Choice: ").strip()
    
    if choice == '1':
        cfg = project_manager.create_project()
        if cfg:
            config.set_active_project(project_manager.current_project)
    
    elif choice == '2':
        cfg = project_manager.load_project()
        if cfg:
            config.set_active_project(project_manager.current_project)
    
    elif choice == '3':
        projects = project_manager.list_projects()
        if projects:
            print("\n📂 Available Projects:")
            for p in projects:
                marker = "→ " if p == project_manager.current_project else "  "
                print(f"   {marker}{p}")
        else:
            print("\n❌ No projects yet!")
    
    elif choice == '4':
        if project_manager.current_project:
            project_manager.edit_project_settings()
        else:
            print("\n❌ Select a project first!")


def import_data():
    """Import data"""
    if not project_manager.current_project:
        print("\n❌ Select a project first!")
        return
    
    project_path = project_manager.get_project_path()
    cfg = project_manager.project_config
    
    print("\n" + "="*50)
    print("📥 DATA IMPORT")
    print("="*50)
    print("""
    [1] Import raw images (unlabeled)
    [2] Import Roboflow/Manual annotations
    [0] Back
    """)
    
    choice = input("Choice: ").strip()
    
    if choice == '1':
        print("\nEnter the directory containing raw images:")
        src_dir = input("Directory path: ").strip()
        
        if not src_dir or not Path(src_dir).exists():
            print("❌ Invalid directory!")
            return
        
        dest_dir = project_path / "raw_images"
        
        count = 0
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            for img in Path(src_dir).glob(f"*{ext}"):
                shutil.copy(img, dest_dir / img.name)
                count += 1
        
        print(f"\n✅ {count} images imported to: {dest_dir}")
    
    elif choice == '2':
        print("\nEnter Roboflow export or manual annotation directory:")
        src_dir = input("Directory path: ").strip()
        
        if not src_dir or not Path(src_dir).exists():
            print("❌ Invalid directory!")
            return
        
        src_path = Path(src_dir)
        dest_dir = project_path / "manual_annotations"
        
        # Copy all contents
        if (src_path / "images").exists():
            # YOLO format
            shutil.copytree(src_path / "images", dest_dir / "images", dirs_exist_ok=True)
            if (src_path / "labels").exists():
                shutil.copytree(src_path / "labels", dest_dir / "labels", dirs_exist_ok=True)
        else:
            # Other formats - copy all
            for item in src_path.iterdir():
                dest = dest_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
        
        print(f"\n✅ Annotations imported to: {dest_dir}")


def train_initial_model():
    """Train initial model (with manual annotations)"""
    if not project_manager.current_project:
        print("\n❌ Select a project first!")
        return
    
    project_path = project_manager.get_project_path()
    cfg = project_manager.project_config
    
    print("\n" + "="*50)
    print("🚀 INITIAL MODEL TRAINING")
    print("="*50)
    
    # Manual annotation check - support both flat and Roboflow structure
    manual_dir = project_path / "manual_annotations"
    
    if cfg["model"]["task"] != "classification":
        labels_found = False
        total_labels = 0
        
        # Check 1: Flat structure (manual_annotations/labels/)
        labels_dir = manual_dir / "labels"
        if labels_dir.exists():
            total_labels += len(list(labels_dir.glob("*.txt")))
        
        # Check 2: Roboflow structure (train/labels, valid/labels, test/labels)
        for split_name in ["train", "valid", "test", "val"]:
            split_labels = manual_dir / split_name / "labels"
            if split_labels.exists():
                total_labels += len(list(split_labels.glob("*.txt")))
        
        if total_labels == 0:
            print("❌ Manual annotations not found!")
            print(f"   Import data first: {manual_dir}")
            print(f"   Expected structure:")
            print(f"     - {manual_dir}/labels/*.txt (flat)")
            print(f"     - {manual_dir}/train/labels/*.txt (Roboflow)")
            return
        
        print(f"✅ Found {total_labels} label files")
    
    from train_model import ModelTrainer
    
    # Update config
    config.set_active_project(project_manager.current_project)
    
    trainer = ModelTrainer()
    
    # Prepare data
    success = trainer.prepare_training_data(roboflow_export_dir=manual_dir)
    
    if success:
        # Show training parameters
        train_cfg = cfg.get("training", {})
        print(f"\n⚙️ Training Parameters:")
        print(f"   Epochs: {train_cfg.get('epochs', 100)}")
        print(f"   Batch Size: {train_cfg.get('batch_size', 16)}")
        print(f"   Image Size: {train_cfg.get('image_size', 640)}")
        print(f"   Model: {cfg['model']['weights']}")
        
        response = input("\nStart training? (y/n): ")
        if response.lower() == 'y':
            trainer.train(epochs=train_cfg.get('epochs', 100))


def run_manual_annotation_gui():
    """Launch manual annotation GUI"""
    print("\n" + "="*50)
    print("✏️ MANUAL ANNOTATION GUI")
    print("="*50)
    
    print("\n🌐 Starting Gradio interface...")
    print("   Will open in browser: http://localhost:7861")
    print("\n💡 Tips:")
    print("   - Select project from left panel")
    print("   - Load images")
    print("   - Click to draw polygons")
    print("   - Click on start point to complete")
    print("   - Press Ctrl+C to close GUI")
    
    try:
        from annotation_tool import create_annotation_gui, main as annotation_main
        annotation_main()
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("   annotation_tool.py not found!")
    except KeyboardInterrupt:
        print("\n\n✅ GUI closed.")


def run_auto_annotation():
    """Run auto annotation"""
    if not project_manager.current_project:
        print("\n❌ Select a project first!")
        return
    
    project_path = project_manager.get_project_path()
    cfg = project_manager.project_config
    
    # Model check
    model_path = project_path / "models" / "latest_model.pt"
    
    if not model_path.exists():
        print("❌ Trained model not found!")
        print("   Train a model first.")
        return
    
    print("\n" + "="*50)
    print("🤖 AUTO ANNOTATION")
    print("="*50)
    
    # Update config
    config.set_active_project(project_manager.current_project)
    
    # Raw images directory
    raw_dir = project_path / "raw_images"
    
    if not raw_dir.exists() or len(list(raw_dir.glob("*.jpg"))) == 0:
        print("❌ Raw images not found!")
        print(f"   Place images in: {raw_dir}")
        return
    
    # Annotation settings - ASK USER
    ann_cfg = cfg.get("annotation", {})
    default_conf = ann_cfg.get('confidence_threshold', 0.5)
    default_min_det = ann_cfg.get('min_detections', 1)
    
    print(f"\n⚙️ Annotation Settings:")
    print(f"   Source: {raw_dir}")
    print(f"   Target: {project_path / 'auto_annotations'}")
    
    # Ask for confidence threshold
    print(f"\n📊 Confidence Threshold (0.001 - 1.0)")
    print(f"   Higher = fewer but more accurate detections")
    print(f"   Lower = more detections but may include false positives")
    conf_input = input(f"   Confidence threshold [{default_conf}]: ").strip()
    if conf_input:
        try:
            confidence_threshold = float(conf_input)
            confidence_threshold = max(0.001, min(1.0, confidence_threshold))
        except ValueError:
            confidence_threshold = default_conf
    else:
        confidence_threshold = default_conf
    
    # Ask for minimum detections
    print(f"\n🔢 Minimum Detections per Image")
    print(f"   How many objects must be detected for an image to be valid?")
    print(f"   Example: 1 = at least 1 detection required")
    print(f"   Example: 2 = at least 2 detections required")
    min_det_input = input(f"   Minimum detections [{default_min_det}]: ").strip()
    if min_det_input:
        try:
            min_detections = int(min_det_input)
            min_detections = max(1, min_detections)  # At least 1
        except ValueError:
            min_detections = default_min_det
    else:
        min_detections = default_min_det
    
    # Show final settings
    print(f"\n✅ Final Settings:")
    print(f"   Confidence Threshold: {confidence_threshold}")
    print(f"   Min Detections: {min_detections}")
    
    response = input("\nStart auto-annotation? (y/n): ")
    
    if response.lower() == 'y':
        from auto_annotate import AutoAnnotator
        
        annotator = AutoAnnotator(model_path, min_detections=min_detections)
        
        annotator.annotate_batch(
            images_dir=raw_dir,
            output_dir=project_path / "auto_annotations",
            confidence_threshold=confidence_threshold,
            save_visualizations=True
        )
        
        # Update project config with new values
        ann_cfg['confidence_threshold'] = confidence_threshold
        ann_cfg['min_detections'] = min_detections
        project_manager.update_project_config({"annotation": ann_cfg})


def prepare_final_dataset():
    """Prepare final dataset"""
    if not project_manager.current_project:
        print("\n❌ Select a project first!")
        return
    
    project_path = project_manager.get_project_path()
    cfg = project_manager.project_config
    
    print("\n" + "="*50)
    print("📦 FINAL DATASET PREPARATION")
    print("="*50)
    
    # Update config
    config.set_active_project(project_manager.current_project)
    
    from prepare_final_dataset import FinalDatasetPreparer
    
    preparer = FinalDatasetPreparer()
    
    # Merge
    merged_images, merged_labels = preparer.merge_annotations(
        manual_dir=project_path / "manual_annotations",
        auto_dir=project_path / "auto_annotations",
        output_dir=project_path / "temp" / "merged"
    )
    
    # Get split ratios based on image count
    num_images = len(list(merged_images.glob("*.jpg")))
    split_ratios = project_manager.get_split_ratios(num_images)
    
    # Update config
    config.TRAIN_RATIO = split_ratios["train"]
    config.VAL_RATIO = split_ratios["val"]
    config.TEST_RATIO = split_ratios["test"]
    
    # Prepare dataset
    preparer.prepare_yolov11_dataset(
        merged_images, 
        merged_labels,
        output_dir=project_path / "final_dataset"
    )
    
    preparer.print_summary()


def train_final_model():
    """Train final model"""
    if not project_manager.current_project:
        print("\n❌ Select a project first!")
        return
    
    project_path = project_manager.get_project_path()
    cfg = project_manager.project_config
    
    # Dataset check
    dataset_yaml = project_path / "final_dataset" / "dataset.yaml"
    
    if not dataset_yaml.exists():
        print("❌ Dataset YAML not found!")
        print("   Prepare final dataset first.")
        return
    
    print("\n" + "="*50)
    print("🎯 FINAL MODEL TRAINING")
    print("="*50)
    
    # Update config
    config.set_active_project(project_manager.current_project)
    
    from train_model import ModelTrainer
    
    trainer = ModelTrainer()
    
    # Training parameters
    train_cfg = cfg.get("training", {})
    default_epochs = train_cfg.get('epochs', 100)
    
    epochs = input(f"\nNumber of epochs (default {default_epochs}): ").strip()
    epochs = int(epochs) if epochs else default_epochs
    
    trainer.train(epochs=epochs)
    
    response = input("\nRun model evaluation? (y/n): ")
    if response.lower() == 'y':
        trainer.evaluate()


def prepare_for_server():
    """Prepare for server training"""
    if not project_manager.current_project:
        print("\n❌ Select a project first!")
        return
    
    project_path = project_manager.get_project_path()
    cfg = project_manager.project_config
    
    # Check what data we have
    final_dataset = project_path / "final_dataset"
    manual_dir = project_path / "manual_annotations"
    raw_images_dir = project_path / "raw_images"
    
    # Check if final_dataset has actual data (not just empty folders)
    has_final_dataset = False
    if final_dataset.exists() and (final_dataset / "dataset.yaml").exists():
        # Check if train folder has actual images
        train_images = final_dataset / "train" / "images"
        if train_images.exists():
            image_count = len(list(train_images.glob("*.jpg"))) + len(list(train_images.glob("*.png")))
            if image_count > 0:
                has_final_dataset = True
                print(f"✅ Final dataset found with {image_count} training images")
    
    has_manual_annotations = False
    manual_image_count = 0
    has_raw_images = raw_images_dir.exists() and (any(raw_images_dir.glob("*.jpg")) or any(raw_images_dir.glob("*.png")))
    
    # Check manual annotations (both flat and Roboflow structure)
    if manual_dir.exists():
        # Flat structure
        if (manual_dir / "labels").exists() and list((manual_dir / "labels").glob("*.txt")):
            has_manual_annotations = True
        # Roboflow structure
        for split_name in ["train", "valid", "test", "val"]:
            if (manual_dir / split_name / "images").exists():
                img_count = len(list((manual_dir / split_name / "images").glob("*.jpg")))
                img_count += len(list((manual_dir / split_name / "images").glob("*.png")))
                manual_image_count += img_count
            if (manual_dir / split_name / "labels").exists():
                if list((manual_dir / split_name / "labels").glob("*.txt")):
                    has_manual_annotations = True
    
    if has_manual_annotations:
        print(f"✅ Manual annotations found with {manual_image_count} images")
    
    # Decide what to include
    if not has_final_dataset and not has_manual_annotations:
        print("❌ No dataset found!")
        print("   Either prepare final dataset first, or import Roboflow annotations.")
        return
    
    # If final_dataset is empty/missing but we have manual_annotations, use them directly
    use_manual_as_dataset = False
    if not has_final_dataset and has_manual_annotations:
        print("\n⚠️ Final dataset empty or not found, but manual annotations exist.")
        print("   Will prepare dataset from manual annotations.")
        use_manual_as_dataset = True
    
    print("\n" + "="*50)
    print("🖥️ SERVER TRAINING PREPARATION")
    print("="*50)
    
    # Export name
    default_name = f"{project_manager.current_project}_server"
    export_name = input(f"\nExport name (default: {default_name}): ").strip()
    export_name = export_name or default_name
    
    export_dir = config.PROJECT_DIR / export_name
    
    if export_dir.exists():
        response = input(f"⚠️ '{export_name}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            return
        shutil.rmtree(export_dir)
    
    export_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📦 Preparing files...")
    
    # 1. Copy dataset (final_dataset or prepare from manual_annotations)
    print("   📂 Preparing dataset...")
    
    if use_manual_as_dataset:
        # Prepare dataset from manual_annotations (Roboflow format)
        dest_dataset = export_dir / "final_dataset"
        dest_dataset.mkdir(parents=True, exist_ok=True)
        
        print(f"   📁 Source: {manual_dir}")
        print(f"   📁 Destination: {dest_dataset}")
        
        # Copy Roboflow structure directly, renaming 'valid' to 'val' if needed
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        total_images_copied = 0
        total_labels_copied = 0
        
        for split_name in ["train", "valid", "test", "val"]:
            src_split = manual_dir / split_name
            if src_split.exists():
                # Normalize to 'val' instead of 'valid' for YOLO
                dest_split_name = "val" if split_name == "valid" else split_name
                dest_split = dest_dataset / dest_split_name
                
                images_copied = 0
                labels_copied = 0
                
                # Copy images
                src_images = src_split / "images"
                if src_images.exists():
                    dest_images = dest_split / "images"
                    dest_images.mkdir(parents=True, exist_ok=True)
                    for ext in image_extensions:
                        for img in src_images.glob(f"*{ext}"):
                            shutil.copy(img, dest_images / img.name)
                            images_copied += 1
                            total_images_copied += 1
                
                # Copy labels
                src_labels = src_split / "labels"
                if src_labels.exists():
                    dest_labels = dest_split / "labels"
                    dest_labels.mkdir(parents=True, exist_ok=True)
                    for lbl in src_labels.glob("*.txt"):
                        if lbl.name not in ['classes.txt']:
                            shutil.copy(lbl, dest_labels / lbl.name)
                            labels_copied += 1
                            total_labels_copied += 1
                
                print(f"      {split_name}: {images_copied} images, {labels_copied} labels")
        
        # Create dataset.yaml
        class_names = list(cfg['classes'].values())
        yaml_content = f"""# Dataset YAML for {project_manager.current_project}
path: .
train: train/images
val: val/images
test: test/images

names:
"""
        for i, name in enumerate(class_names):
            yaml_content += f"  {i}: {name}\n"
        
        with open(dest_dataset / "dataset.yaml", 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f"   ✅ Dataset prepared: {total_images_copied} images, {total_labels_copied} labels")
    else:
        # Copy existing final_dataset
        shutil.copytree(final_dataset, export_dir / "final_dataset")
    
    # 2. Required files
    print("   📄 Copying Python files...")
    for f in ['train_model.py', 'utils.py', 'requirements.txt']:
        src = config.PROJECT_DIR / f
        if src.exists():
            shutil.copy(src, export_dir / f)
    
    # 3. Copy project config
    shutil.copy(project_path / "project_config.json", export_dir / "project_config.json")
    
    # 4. Create server config.py
    model_cfg = cfg["model"]
    train_cfg = cfg.get("training", {})
    task_type = model_cfg.get('task', 'segmentation')
    
    server_config = f'''"""
Server Training Configuration
==============================
Project: {project_manager.current_project}
Task: {task_type.upper()}
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).parent
PROJECTS_DIR = PROJECT_DIR
FINAL_DATASET_DIR = PROJECT_DIR / "final_dataset"
MODELS_DIR = PROJECT_DIR / "models"
TEMP_DIR = PROJECT_DIR / "temp"
MANUAL_ANNOTATIONS_DIR = PROJECT_DIR / "final_dataset"

# Model Configuration
MODEL_TASK = "{task_type}"  # detection, segmentation, classification
MODEL_FAMILY = "{model_cfg['family']}"  # yolov11, yolov8, resnet
YOLO_MODEL_SIZE = "{model_cfg['size']}"
YOLO_SEG_MODEL = "{model_cfg['weights']}"  # Used by train_model.py

# Training
INITIAL_TRAINING_EPOCHS = {train_cfg.get('epochs', 100)}
BATCH_SIZE = {train_cfg.get('batch_size', 16)}
IMAGE_SIZE = {train_cfg.get('image_size', 640)}
LEARNING_RATE = {train_cfg.get('learning_rate', 0.01)}

# Minimum samples
MIN_SAMPLES_FOR_TRAINING = 20
RECOMMENDED_SAMPLES = 100

# Classes
CLASSES = {cfg['classes']}
NUM_CLASSES = {cfg['num_classes']}

# Augmentation
AUGMENTATION_ENABLED = True
AUGMENTATION_CONFIG = {{
    "horizontal_flip": True,
    "vertical_flip": False,
    "rotation_range": 15,
}}

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
'''
    
    with open(export_dir / "config.py", 'w', encoding='utf-8') as f:
        f.write(server_config)
    
    # 5. Server main.py
    server_main = f'''"""
Server Training Script
======================
Project: {project_manager.current_project}
Task: {task_type.upper()}
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import config
from train_model import ModelTrainer

def main():
    print("=" * 60)
    print("🎯 {project_manager.current_project.upper()} - SERVER TRAINING")
    print("=" * 60)
    print(f"Task: {{config.MODEL_TASK.upper()}}")
    print(f"Model: {{config.YOLO_SEG_MODEL}}")
    
    config.create_directories()
    
    # Dataset info
    train_dir = config.FINAL_DATASET_DIR / "train" / "images"
    val_dir = config.FINAL_DATASET_DIR / "val" / "images"
    
    train_count = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
    val_count = len(list(val_dir.glob("*.jpg"))) if val_dir.exists() else 0
    
    print(f"\\nDataset: Train={{train_count}}, Val={{val_count}}")
    print(f"Model: {{config.YOLO_SEG_MODEL}}")
    print(f"Epochs: {{config.INITIAL_TRAINING_EPOCHS}}")
    
    response = input("\\nStart training? (y/n): ")
    if response.lower() == 'y':
        trainer = ModelTrainer()
        trainer.train(epochs=config.INITIAL_TRAINING_EPOCHS)
        
        response = input("\\nRun evaluation? (y/n): ")
        if response.lower() == 'y':
            trainer.evaluate()

if __name__ == "__main__":
    main()
'''
    
    with open(export_dir / "main.py", 'w', encoding='utf-8') as f:
        f.write(server_main)
    
    # 6. README
    readme = f'''# {project_manager.current_project} - Server Training

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python main.py
```

## Model
- Family: {model_cfg['family']}
- Task: {model_cfg['task']}
- Size: {model_cfg['size']}
- Weights: {model_cfg['weights']}

## After Training
Best model: `models/yolo_training/train_*/weights/best.pt`
'''
    
    with open(export_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme)
    
    # 7. Create ZIP
    print("   🗜️ Creating ZIP...")
    zip_path = config.PROJECT_DIR / f"{export_name}.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(export_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(export_dir.parent)
                zipf.write(file_path, arcname)
    
    zip_size = zip_path.stat().st_size / (1024 * 1024)
    
    print("\n" + "="*50)
    print("✅ SERVER EXPORT COMPLETE!")
    print("="*50)
    print(f"\n📁 Folder: {export_dir}")
    print(f"🗜️ ZIP: {zip_path}")
    print(f"📊 Size: {zip_size:.1f} MB")
    
    print(f"""
📋 On Server:
   1. unzip {export_name}.zip
   2. cd {export_name}
   3. pip install -r requirements.txt
   4. python main.py
    """)


def main_menu():
    """Main menu"""
    while True:
        print("\n" + "="*60)
        print("📋 MAIN MENU")
        print("="*60)
        
        print_active_project()
        
        print("""
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
        """)
        
        choice = input("Your choice (0-10): ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye!")
            break
        
        elif choice == '1':
            project_menu()
        
        elif choice == '2':
            import_data()
        
        elif choice == '3':
            run_manual_annotation_gui()
        
        elif choice == '4':
            run_auto_annotation()
        
        elif choice == '5':
            prepare_final_dataset()
        
        elif choice == '6':
            train_initial_model()
        
        elif choice == '7':
            train_final_model()
        
        elif choice == '8':
            prepare_for_server()
        
        elif choice == '9':
            print_project_status()
        
        elif choice == '10':
            if project_manager.current_project:
                project_manager.edit_project_settings()
            else:
                print("\n❌ Select a project first!")
        
        else:
            print("❌ Invalid choice!")


def main():
    """Main function"""
    print_banner()
    
    # Create projects directory
    config.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # List existing projects
    projects = project_manager.list_projects()
    
    if projects:
        print(f"\n📂 Existing projects: {', '.join(projects)}")
        print("   Select [1] from menu to load a project.")
    else:
        print("\n💡 No projects yet. Select [1] from menu to create a new project.")
    
    # Main menu
    main_menu()


if __name__ == "__main__":
    main()
