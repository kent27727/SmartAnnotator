"""
Auto-Annotation Tool - Utilities
================================
Helper functions and data processing tools
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from tqdm import tqdm
import yaml


def load_yolo_segmentation_label(label_path: Path) -> List[Dict]:
    """
    Read label file in YOLO segmentation format
    
    Format: class_id x1 y1 x2 y2 x3 y3 ... (normalized coordinates)
    
    Returns:
        List of dicts with 'class_id' and 'polygon' keys
    """
    annotations = []
    
    if not label_path.exists():
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # minimum: class_id + 3 points (6 coords)
                continue
            
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            # Convert coordinates to (x, y) pairs
            polygon = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            
            annotations.append({
                'class_id': class_id,
                'polygon': polygon
            })
    
    return annotations


def save_yolo_segmentation_label(label_path: Path, annotations: List[Dict]):
    """
    Save label file in YOLO segmentation format
    
    Args:
        label_path: File path to save
        annotations: List of dicts with 'class_id' and 'polygon' keys
    """
    with open(label_path, 'w') as f:
        for ann in annotations:
            class_id = ann['class_id']
            polygon = ann['polygon']
            
            # Flatten polygon coordinates
            coords = []
            for x, y in polygon:
                coords.extend([f"{x:.6f}", f"{y:.6f}"])
            
            line = f"{class_id} " + " ".join(coords)
            f.write(line + "\n")


def polygon_to_mask(polygon: List[Tuple[float, float]], 
                    image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert normalized polygon coordinates to binary mask
    
    Args:
        polygon: List of (x, y) normalized coordinates
        image_shape: (height, width)
        
    Returns:
        Binary mask (0 or 255)
    """
    h, w = image_shape
    
    # Convert normalized coordinates to pixel coordinates
    points = np.array([[int(x * w), int(y * h)] for x, y in polygon], dtype=np.int32)
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    
    return mask


def mask_to_polygon(mask: np.ndarray, 
                    normalize: bool = True) -> List[Tuple[float, float]]:
    """
    Convert binary mask to polygon coordinates
    
    Args:
        mask: Binary mask
        normalize: If True, normalize coordinates
        
    Returns:
        List of (x, y) coordinates
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify contour
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    h, w = mask.shape
    
    polygon = []
    for point in approx:
        x, y = point[0]
        if normalize:
            polygon.append((x / w, y / h))
        else:
            polygon.append((x, y))
    
    return polygon


def load_coco_annotations(json_path: Path) -> Dict:
    """Read annotation file in COCO format"""
    with open(json_path, 'r') as f:
        return json.load(f)


def coco_to_yolo_segmentation(coco_data: Dict, 
                               output_dir: Path,
                               images_dir: Path):
    """
    Convert COCO format annotations to YOLO segmentation format
    
    Args:
        coco_data: COCO format dict
        output_dir: Directory to save YOLO label files
        images_dir: Directory containing image files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Image ID -> filename mapping
    image_map = {img['id']: img for img in coco_data['images']}
    
    # Category ID -> class index mapping
    category_map = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Create YOLO label file for each image
    for img_id, img_info in tqdm(image_map.items(), desc="Converting annotations"):
        filename = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        label_filename = Path(filename).stem + '.txt'
        label_path = output_dir / label_filename
        
        yolo_annotations = []
        
        if img_id in annotations_by_image:
            for ann in annotations_by_image[img_id]:
                class_id = category_map.get(ann['category_id'], 0)
                
                # Segmentation polygon
                if 'segmentation' in ann and ann['segmentation']:
                    for seg in ann['segmentation']:
                        # COCO: [x1, y1, x2, y2, ...] pixel coordinates
                        polygon = []
                        for i in range(0, len(seg), 2):
                            x = seg[i] / width
                            y = seg[i + 1] / height
                            polygon.append((x, y))
                        
                        yolo_annotations.append({
                            'class_id': class_id,
                            'polygon': polygon
                        })
        
        save_yolo_segmentation_label(label_path, yolo_annotations)
    
    print(f"✅ {len(image_map)} annotations converted!")


def roboflow_to_yolo(roboflow_dir: Path, output_dir: Path):
    """
    Convert Roboflow export to YOLO format
    
    Roboflow usually exports in these formats:
    - YOLOv5/v8 format (already compatible)
    - COCO format
    - Pascal VOC format
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO format check
    coco_files = list(roboflow_dir.glob("**/annotations*.json")) + \
                 list(roboflow_dir.glob("**/_annotations.coco.json"))
    
    if coco_files:
        print("📦 COCO format detected, converting...")
        for coco_file in coco_files:
            coco_data = load_coco_annotations(coco_file)
            coco_to_yolo_segmentation(coco_data, output_dir, roboflow_dir)
        return
    
    # YOLO format check (txt files)
    txt_files = list(roboflow_dir.glob("**/*.txt"))
    txt_files = [f for f in txt_files if f.name != 'classes.txt' and f.name != 'data.yaml']
    
    if txt_files:
        print("📦 YOLO format detected, copying...")
        for txt_file in tqdm(txt_files, desc="Copying labels"):
            shutil.copy(txt_file, output_dir / txt_file.name)
        print(f"✅ {len(txt_files)} labels copied!")
        return
    
    print("⚠️ Unknown format! Manual conversion may be required.")


def create_dataset_yaml(dataset_dir: Path, 
                        class_names: List[str],
                        output_path: Optional[Path] = None) -> Path:
    """
    Create dataset.yaml file for YOLOv11
    
    Args:
        dataset_dir: Dataset main directory
        class_names: List of class names
        output_path: Path to save YAML file
        
    Returns:
        Path to YAML file
    """
    if output_path is None:
        output_path = dataset_dir / "dataset.yaml"
    
    yaml_content = {
        'path': str(dataset_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"✅ Dataset YAML created: {output_path}")
    return output_path


def split_dataset(images_dir: Path,
                  labels_dir: Path,
                  output_dir: Path,
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.2,
                  test_ratio: float = 0.1,
                  seed: int = 42):
    """
    Split dataset into train/val/test
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing label files
        output_dir: Output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
    """
    np.random.seed(seed)
    
    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    for ext in image_extensions:
        images.extend(images_dir.glob(f"*{ext}"))
    
    # Only keep images that have labels
    valid_images = []
    for img in images:
        label_path = labels_dir / (img.stem + '.txt')
        if label_path.exists():
            valid_images.append(img)
    
    # Shuffle
    np.random.shuffle(valid_images)
    
    # Split
    n = len(valid_images)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    splits = {
        'train': valid_images[:train_end],
        'val': valid_images[train_end:val_end],
        'test': valid_images[val_end:]
    }
    
    # Copy files
    for split_name, split_images in splits.items():
        img_out = output_dir / split_name / "images"
        lbl_out = output_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        
        for img in tqdm(split_images, desc=f"Copying {split_name}"):
            # Copy image
            shutil.copy(img, img_out / img.name)
            
            # Copy label
            label_path = labels_dir / (img.stem + '.txt')
            shutil.copy(label_path, lbl_out / label_path.name)
    
    print(f"✅ Dataset split:")
    print(f"   Train: {len(splits['train'])} images")
    print(f"   Val: {len(splits['val'])} images")
    print(f"   Test: {len(splits['test'])} images")


def visualize_annotation(image_path: Path,
                         label_path: Path,
                         output_path: Optional[Path] = None,
                         show: bool = False) -> np.ndarray:
    """
    Visualize annotation
    
    Args:
        image_path: Image file path
        label_path: Label file path
        output_path: Output file path (None to skip saving)
        show: If True, display the image
        
    Returns:
        Annotated image
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    h, w = image.shape[:2]
    
    # Read annotations
    annotations = load_yolo_segmentation_label(label_path)
    
    # Copy for overlay
    overlay = image.copy()
    
    # For each annotation
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    
    for i, ann in enumerate(annotations):
        color = colors[ann['class_id'] % len(colors)]
        polygon = ann['polygon']
        
        # Convert to pixel coordinates
        points = np.array([[int(x * w), int(y * h)] for x, y in polygon], dtype=np.int32)
        
        # Draw polygon
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(image, [points], True, color, 2)
    
    # Apply overlay
    alpha = 0.4
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    if output_path:
        cv2.imwrite(str(output_path), image)
    
    if show:
        cv2.imshow("Annotation", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return image


def count_images(directory: Path) -> int:
    """Count images in directory"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    count = 0
    for ext in image_extensions:
        count += len(list(directory.glob(f"*{ext}")))
    return count


def get_unlabeled_images(images_dir: Path, labels_dir: Path) -> List[Path]:
    """Find images that haven't been labeled yet"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    all_images = []
    for ext in image_extensions:
        all_images.extend(images_dir.glob(f"*{ext}"))
    
    unlabeled = []
    for img in all_images:
        label_path = labels_dir / (img.stem + '.txt')
        if not label_path.exists():
            unlabeled.append(img)
    
    return unlabeled


if __name__ == "__main__":
    # Test
    print("Utils module loaded successfully!")
