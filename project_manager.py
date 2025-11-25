"""
Auto-Annotation Tool - Project Manager
======================================
Project and dataset management, model selection, configuration
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import shutil

# ============================================
# MODEL AND TASK DEFINITIONS
# ============================================

# Supported models and task types
SUPPORTED_MODELS = {
    "yolov11": {
        "name": "YOLOv11",
        "tasks": ["detection", "segmentation", "classification"],
        "sizes": {
            "n": "Nano (fastest, lightest)",
            "s": "Small (fast, light)",
            "m": "Medium (balanced)",
            "l": "Large (powerful)",
            "x": "XLarge (most powerful)"
        },
        "weights": {
            "detection": "yolo11{size}.pt",
            "segmentation": "yolo11{size}-seg.pt",
            "classification": "yolo11{size}-cls.pt"
        }
    },
    "yolov8": {
        "name": "YOLOv8",
        "tasks": ["detection", "segmentation", "classification"],
        "sizes": {
            "n": "Nano (fastest, lightest)",
            "s": "Small (fast, light)",
            "m": "Medium (balanced)",
            "l": "Large (powerful)",
            "x": "XLarge (most powerful)"
        },
        "weights": {
            "detection": "yolov8{size}.pt",
            "segmentation": "yolov8{size}-seg.pt",
            "classification": "yolov8{size}-cls.pt"
        }
    },
    "resnet": {
        "name": "ResNet (Classification Only)",
        "tasks": ["classification"],
        "sizes": {
            "18": "ResNet18 (light)",
            "34": "ResNet34 (medium)",
            "50": "ResNet50 (powerful)",
            "101": "ResNet101 (very powerful)",
            "152": "ResNet152 (most powerful)"
        },
        "weights": {
            "classification": "resnet{size}"
        }
    }
}

# Task type descriptions
TASK_DESCRIPTIONS = {
    "classification": "Classification - Which class does the image belong to?",
    "detection": "Detection (Bounding Box) - Object location with rectangular box",
    "segmentation": "Segmentation - Pixel-level mask of the object"
}

# Label format information
LABEL_FORMATS = {
    "classification": {
        "description": "Folder-based (separate folder for each class)",
        "structure": "dataset/class_name/image.jpg"
    },
    "detection": {
        "description": "YOLO format (txt file: class x_center y_center width height)",
        "structure": "dataset/images/, dataset/labels/"
    },
    "segmentation": {
        "description": "YOLO Segment format (txt file: class x1 y1 x2 y2 ... xn yn)",
        "structure": "dataset/images/, dataset/labels/"
    }
}

# Recommended train/val/test ratios
RECOMMENDED_SPLITS = {
    "small": {"train": 0.7, "val": 0.2, "test": 0.1, "description": "< 1000 images"},
    "medium": {"train": 0.8, "val": 0.1, "test": 0.1, "description": "1000-5000 images"},
    "large": {"train": 0.85, "val": 0.1, "test": 0.05, "description": "> 5000 images"}
}


class ProjectManager:
    """Project management class"""
    
    def __init__(self, base_dir: Path = None):
        """
        Args:
            base_dir: Main directory where projects will be stored
        """
        self.base_dir = base_dir or Path(__file__).parent
        self.projects_dir = self.base_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_project = None
        self.project_config = None
    
    def list_projects(self) -> List[str]:
        """List existing projects"""
        projects = []
        for item in self.projects_dir.iterdir():
            if item.is_dir() and (item / "project_config.json").exists():
                projects.append(item.name)
        return sorted(projects)
    
    def create_project(self) -> Optional[Dict]:
        """Create new project - interactive"""
        
        print("\n" + "="*60)
        print("📁 CREATE NEW PROJECT")
        print("="*60)
        
        # 1. Project name
        print("\n1️⃣ PROJECT NAME")
        print("   Example: car_detection, product_catalog, my_project")
        
        while True:
            project_name = input("\nProject name: ").strip().lower().replace(" ", "_")
            
            if not project_name:
                print("❌ Project name cannot be empty!")
                continue
            
            if not project_name[0].isalpha():
                print("❌ Project name must start with a letter!")
                continue
            
            project_path = self.projects_dir / project_name
            if project_path.exists():
                response = input(f"⚠️ '{project_name}' already exists. Overwrite? (y/n): ")
                if response.lower() != 'y':
                    continue
                shutil.rmtree(project_path)
            
            break
        
        # 2. Model selection
        print("\n2️⃣ MODEL SELECTION")
        print("-" * 40)
        
        model_options = list(SUPPORTED_MODELS.keys())
        for i, model_key in enumerate(model_options, 1):
            model_info = SUPPORTED_MODELS[model_key]
            tasks = ", ".join(model_info["tasks"])
            print(f"   [{i}] {model_info['name']}")
            print(f"       Supported tasks: {tasks}")
        
        while True:
            choice = input(f"\nSelect model (1-{len(model_options)}): ").strip()
            try:
                model_idx = int(choice) - 1
                if 0 <= model_idx < len(model_options):
                    selected_model = model_options[model_idx]
                    break
            except ValueError:
                pass
            print("❌ Invalid selection!")
        
        # 3. Task type selection
        print("\n3️⃣ TASK TYPE SELECTION")
        print("-" * 40)
        
        available_tasks = SUPPORTED_MODELS[selected_model]["tasks"]
        for i, task in enumerate(available_tasks, 1):
            print(f"   [{i}] {task.upper()}")
            print(f"       {TASK_DESCRIPTIONS[task]}")
        
        while True:
            choice = input(f"\nSelect task type (1-{len(available_tasks)}): ").strip()
            try:
                task_idx = int(choice) - 1
                if 0 <= task_idx < len(available_tasks):
                    selected_task = available_tasks[task_idx]
                    break
            except ValueError:
                pass
            print("❌ Invalid selection!")
        
        # 4. Model size selection
        print("\n4️⃣ MODEL SIZE SELECTION")
        print("-" * 40)
        
        sizes = SUPPORTED_MODELS[selected_model]["sizes"]
        size_options = list(sizes.keys())
        for i, size_key in enumerate(size_options, 1):
            print(f"   [{i}] {size_key.upper()} - {sizes[size_key]}")
        
        while True:
            choice = input(f"\nSelect model size (1-{len(size_options)}): ").strip()
            try:
                size_idx = int(choice) - 1
                if 0 <= size_idx < len(size_options):
                    selected_size = size_options[size_idx]
                    break
            except ValueError:
                pass
            print("❌ Invalid selection!")
        
        # 5. Train/Val/Test ratios
        print("\n5️⃣ TRAIN / VAL / TEST RATIOS")
        print("-" * 40)
        print("   [1] Automatic (system selects based on data size)")
        print("   [2] Manual (you specify)")
        
        split_config = None
        while True:
            choice = input("\nChoice (1-2): ").strip()
            if choice == '1':
                split_config = "auto"
                break
            elif choice == '2':
                print("\nEnter ratios (must sum to 1.0):")
                try:
                    train = float(input("   Train ratio (e.g., 0.8): ").strip())
                    val = float(input("   Val ratio (e.g., 0.1): ").strip())
                    test = float(input("   Test ratio (e.g., 0.1): ").strip())
                    
                    if abs(train + val + test - 1.0) > 0.01:
                        print("❌ Ratios must sum to 1.0!")
                        continue
                    
                    split_config = {"train": train, "val": val, "test": test}
                    break
                except ValueError:
                    print("❌ Invalid value!")
            else:
                print("❌ Invalid selection!")
        
        # 6. Class definitions
        print("\n6️⃣ CLASS DEFINITIONS")
        print("-" * 40)
        print("   Enter classes to be labeled.")
        print("   Separate multiple classes with comma.")
        print("   Example: car")
        print("   Example: car, person, bicycle")
        
        classes_input = input("\nClasses: ").strip()
        classes = [c.strip() for c in classes_input.split(",") if c.strip()]
        
        if not classes:
            classes = ["object"]
            print(f"   Using default: {classes}")
        
        # 7. Minimum detections for auto annotation
        print("\n7️⃣ MINIMUM DETECTIONS (Auto Annotation)")
        print("-" * 40)
        print("   How many objects must be detected in an image for it to be valid?")
        print("   Example: 1 = At least 1 detection required")
        print("   Example: 2 = At least 2 detections required (e.g., two eyes)")
        print("   Example: 3 = At least 3 detections required")
        
        while True:
            min_det_input = input("\nMinimum detections (default: 1): ").strip()
            if not min_det_input:
                min_detections = 1
                break
            try:
                min_detections = int(min_det_input)
                if min_detections < 1:
                    print("❌ Minimum detections must be at least 1!")
                    continue
                break
            except ValueError:
                print("❌ Please enter a valid number!")
        
        print(f"   ✅ Min detections set to: {min_detections}")
        
        # Create project configuration
        config = {
            "project_name": project_name,
            "created_at": datetime.now().isoformat(),
            "model": {
                "family": selected_model,
                "task": selected_task,
                "size": selected_size,
                "weights": SUPPORTED_MODELS[selected_model]["weights"][selected_task].format(size=selected_size)
            },
            "classes": {i: name for i, name in enumerate(classes)},
            "num_classes": len(classes),
            "split": split_config,
            "training": {
                "epochs": 100,
                "batch_size": 16,
                "image_size": 640,
                "learning_rate": 0.01
            },
            "annotation": {
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "min_detections": min_detections
            }
        }
        
        # Create project directories
        project_path = self.projects_dir / project_name
        self._create_project_structure(project_path, config)
        
        # Save configuration
        config_file = project_path / "project_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Summary
        print("\n" + "="*60)
        print("✅ PROJECT CREATED!")
        print("="*60)
        self._print_project_summary(config)
        print(f"\n📁 Project directory: {project_path}")
        
        self.current_project = project_name
        self.project_config = config
        
        return config
    
    def _create_project_structure(self, project_path: Path, config: Dict):
        """Create project directory structure"""
        
        task = config["model"]["task"]
        
        # Main directories
        dirs = [
            project_path / "raw_images",           # Unlabeled images
            project_path / "manual_annotations",   # Manual labels
            project_path / "auto_annotations",     # Auto labels
            project_path / "final_dataset",        # Final dataset
            project_path / "models",               # Trained models
            project_path / "exports",              # Export files
        ]
        
        # Task-specific subdirectories
        if task == "classification":
            # Classification class folders
            for class_name in config["classes"].values():
                dirs.append(project_path / "manual_annotations" / class_name)
                dirs.append(project_path / "auto_annotations" / class_name)
                dirs.append(project_path / "final_dataset" / "train" / class_name)
                dirs.append(project_path / "final_dataset" / "val" / class_name)
                dirs.append(project_path / "final_dataset" / "test" / class_name)
        else:
            # Detection/Segmentation images/labels structure
            for split in ["train", "val", "test"]:
                dirs.append(project_path / "final_dataset" / split / "images")
                dirs.append(project_path / "final_dataset" / split / "labels")
            
            dirs.append(project_path / "manual_annotations" / "images")
            dirs.append(project_path / "manual_annotations" / "labels")
            dirs.append(project_path / "auto_annotations" / "images")
            dirs.append(project_path / "auto_annotations" / "labels")
            dirs.append(project_path / "auto_annotations" / "visualizations")
            dirs.append(project_path / "auto_annotations" / "unvalid")
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        # Create dataset YAML (for detection/segmentation)
        if task != "classification":
            self._create_dataset_yaml(project_path, config)
    
    def _create_dataset_yaml(self, project_path: Path, config: Dict):
        """Create dataset YAML file"""
        
        yaml_content = f"""# {config['project_name']} Dataset Configuration
# Created: {config['created_at']}
# Task: {config['model']['task']}

path: {project_path / 'final_dataset'}
train: train/images
val: val/images
test: test/images

nc: {config['num_classes']}
names: {list(config['classes'].values())}
"""
        
        yaml_file = project_path / "final_dataset" / "dataset.yaml"
        with open(yaml_file, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
    
    def _print_project_summary(self, config: Dict):
        """Print project summary"""
        
        model_info = SUPPORTED_MODELS[config["model"]["family"]]
        
        print(f"\n📋 PROJECT SUMMARY")
        print(f"   ├─ Project Name: {config['project_name']}")
        print(f"   ├─ Model: {model_info['name']}")
        print(f"   ├─ Task: {config['model']['task'].upper()}")
        print(f"   ├─ Size: {config['model']['size'].upper()}")
        print(f"   ├─ Weights: {config['model']['weights']}")
        print(f"   ├─ Classes: {list(config['classes'].values())}")
        print(f"   ├─ Min Detections: {config.get('annotation', {}).get('min_detections', 1)}")
        
        if config['split'] == "auto":
            print(f"   └─ Split: Automatic")
        else:
            print(f"   └─ Split: Train={config['split']['train']}, Val={config['split']['val']}, Test={config['split']['test']}")
    
    def load_project(self, project_name: str = None) -> Optional[Dict]:
        """Load existing project"""
        
        if project_name is None:
            # Show project list
            projects = self.list_projects()
            
            if not projects:
                print("\n❌ No projects found!")
                print("   Create a new project first.")
                return None
            
            print("\n📂 EXISTING PROJECTS")
            print("-" * 40)
            
            for i, proj in enumerate(projects, 1):
                # Read project info
                config_file = self.projects_dir / proj / "project_config.json"
                with open(config_file, 'r', encoding='utf-8') as f:
                    proj_config = json.load(f)
                
                model_info = f"{proj_config['model']['family']} {proj_config['model']['task']} ({proj_config['model']['size']})"
                print(f"   [{i}] {proj}")
                print(f"       Model: {model_info}")
            
            print(f"   [0] Cancel")
            
            while True:
                choice = input(f"\nSelect project (0-{len(projects)}): ").strip()
                
                if choice == '0':
                    return None
                
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(projects):
                        project_name = projects[idx]
                        break
                except ValueError:
                    pass
                print("❌ Invalid selection!")
        
        # Load project
        config_file = self.projects_dir / project_name / "project_config.json"
        
        if not config_file.exists():
            print(f"❌ Project not found: {project_name}")
            return None
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.current_project = project_name
        self.project_config = config
        
        print(f"\n✅ Project loaded: {project_name}")
        self._print_project_summary(config)
        
        return config
    
    def get_project_path(self) -> Path:
        """Return active project directory"""
        if self.current_project:
            return self.projects_dir / self.current_project
        return None
    
    def get_split_ratios(self, num_images: int) -> Dict[str, float]:
        """Return train/val/test ratios based on image count"""
        
        if self.project_config is None:
            return {"train": 0.8, "val": 0.1, "test": 0.1}
        
        split = self.project_config.get("split", "auto")
        
        if split != "auto":
            return split
        
        # Automatic selection
        if num_images < 1000:
            ratios = RECOMMENDED_SPLITS["small"]
        elif num_images < 5000:
            ratios = RECOMMENDED_SPLITS["medium"]
        else:
            ratios = RECOMMENDED_SPLITS["large"]
        
        print(f"\n📊 Automatic split selected ({ratios['description']}):")
        print(f"   Train: {ratios['train']*100:.0f}%")
        print(f"   Val: {ratios['val']*100:.0f}%")
        print(f"   Test: {ratios['test']*100:.0f}%")
        
        return {"train": ratios["train"], "val": ratios["val"], "test": ratios["test"]}
    
    def update_project_config(self, updates: Dict):
        """Update project configuration"""
        
        if self.current_project is None or self.project_config is None:
            return
        
        self.project_config.update(updates)
        
        config_file = self.projects_dir / self.current_project / "project_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.project_config, f, indent=2, ensure_ascii=False)
    
    def edit_project_settings(self):
        """Edit project settings"""
        
        if self.project_config is None:
            print("❌ Load a project first!")
            return
        
        print("\n" + "="*50)
        print("⚙️ EDIT PROJECT SETTINGS")
        print("="*50)
        
        print("""
    [1] Change model size
    [2] Change training parameters
    [3] Change annotation settings
    [4] Change Train/Val/Test ratios
    [0] Back
        """)
        
        choice = input("Choice: ").strip()
        
        if choice == '1':
            self._edit_model_size()
        elif choice == '2':
            self._edit_training_params()
        elif choice == '3':
            self._edit_annotation_params()
        elif choice == '4':
            self._edit_split_ratios()
    
    def _edit_model_size(self):
        """Change model size"""
        model_family = self.project_config["model"]["family"]
        sizes = SUPPORTED_MODELS[model_family]["sizes"]
        
        print("\nCurrent size:", self.project_config["model"]["size"].upper())
        print("\nAvailable options:")
        
        size_options = list(sizes.keys())
        for i, size_key in enumerate(size_options, 1):
            print(f"   [{i}] {size_key.upper()} - {sizes[size_key]}")
        
        while True:
            choice = input(f"\nNew size (1-{len(size_options)}): ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(size_options):
                    new_size = size_options[idx]
                    task = self.project_config["model"]["task"]
                    new_weights = SUPPORTED_MODELS[model_family]["weights"][task].format(size=new_size)
                    
                    self.project_config["model"]["size"] = new_size
                    self.project_config["model"]["weights"] = new_weights
                    self.update_project_config(self.project_config)
                    
                    print(f"✅ Model size updated: {new_size.upper()}")
                    print(f"   Weights: {new_weights}")
                    break
            except ValueError:
                pass
            print("❌ Invalid selection!")
    
    def _edit_training_params(self):
        """Change training parameters"""
        current = self.project_config["training"]
        
        print(f"\nCurrent values:")
        print(f"   Epochs: {current['epochs']}")
        print(f"   Batch Size: {current['batch_size']}")
        print(f"   Image Size: {current['image_size']}")
        
        try:
            epochs = input(f"\nEpochs ({current['epochs']}): ").strip()
            batch = input(f"Batch Size ({current['batch_size']}): ").strip()
            imgsz = input(f"Image Size ({current['image_size']}): ").strip()
            
            if epochs:
                current['epochs'] = int(epochs)
            if batch:
                current['batch_size'] = int(batch)
            if imgsz:
                current['image_size'] = int(imgsz)
            
            self.update_project_config({"training": current})
            print("✅ Training parameters updated!")
            
        except ValueError:
            print("❌ Invalid value!")
    
    def _edit_annotation_params(self):
        """Change annotation parameters"""
        current = self.project_config["annotation"]
        
        print(f"\nCurrent values:")
        print(f"   Confidence Threshold: {current['confidence_threshold']}")
        print(f"   IoU Threshold: {current['iou_threshold']}")
        print(f"   Min Detections: {current['min_detections']}")
        
        try:
            conf = input(f"\nConfidence Threshold ({current['confidence_threshold']}): ").strip()
            iou = input(f"IoU Threshold ({current['iou_threshold']}): ").strip()
            min_det = input(f"Min Detections ({current['min_detections']}): ").strip()
            
            if conf:
                current['confidence_threshold'] = float(conf)
            if iou:
                current['iou_threshold'] = float(iou)
            if min_det:
                current['min_detections'] = int(min_det)
            
            self.update_project_config({"annotation": current})
            print("✅ Annotation parameters updated!")
            
        except ValueError:
            print("❌ Invalid value!")
    
    def _edit_split_ratios(self):
        """Change split ratios"""
        print("\n[1] Automatic (based on data size)")
        print("[2] Manual")
        
        choice = input("\nChoice: ").strip()
        
        if choice == '1':
            self.update_project_config({"split": "auto"})
            print("✅ Automatic split selected!")
        elif choice == '2':
            try:
                train = float(input("Train ratio: ").strip())
                val = float(input("Val ratio: ").strip())
                test = float(input("Test ratio: ").strip())
                
                if abs(train + val + test - 1.0) > 0.01:
                    print("❌ Ratios must sum to 1.0!")
                    return
                
                self.update_project_config({"split": {"train": train, "val": val, "test": test}})
                print("✅ Split ratios updated!")
                
            except ValueError:
                print("❌ Invalid value!")


def interactive_project_setup():
    """Interactive project setup"""
    
    manager = ProjectManager()
    
    print("\n" + "="*60)
    print("📁 PROJECT MANAGEMENT")
    print("="*60)
    print("""
    [1] Create new project
    [2] Load existing project
    [3] List projects
    [0] Back
    """)
    
    choice = input("Choice: ").strip()
    
    if choice == '1':
        return manager.create_project()
    elif choice == '2':
        return manager.load_project()
    elif choice == '3':
        projects = manager.list_projects()
        if projects:
            print("\n📂 Projects:")
            for p in projects:
                print(f"   - {p}")
        else:
            print("\n❌ No projects yet!")
        return None
    
    return None


if __name__ == "__main__":
    interactive_project_setup()
