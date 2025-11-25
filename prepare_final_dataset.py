"""
Auto-Annotation Tool - Final Dataset Preparation
=================================================
Merge manual + automatic annotations and export in YOLOv11 format
"""

import os
import sys
from pathlib import Path
import shutil
from datetime import datetime
import yaml

from tqdm import tqdm
import numpy as np

import config
from utils import (
    count_images,
    split_dataset,
    create_dataset_yaml
)


class FinalDatasetPreparer:
    """Final dataset preparer"""
    
    def __init__(self):
        self.stats = {
            'manual_images': 0,
            'auto_images': 0,
            'total_images': 0,
            'train': 0,
            'val': 0,
            'test': 0
        }
    
    def merge_annotations(self,
                          manual_dir: Path = None,
                          auto_dir: Path = None,
                          output_dir: Path = None):
        """
        Merge manual and automatic annotations
        
        Args:
            manual_dir: Manual annotation directory (Roboflow export)
            auto_dir: Automatic annotation directory
            output_dir: Merged output directory
        """
        auto_dir = auto_dir or config.AUTO_ANNOTATIONS_DIR
        output_dir = output_dir or config.TEMP_DIR / "merged"
        
        merged_images = output_dir / "images"
        merged_labels = output_dir / "labels"
        merged_images.mkdir(parents=True, exist_ok=True)
        merged_labels.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*50)
        print("🔀 MERGING ANNOTATIONS")
        print("="*50)
        
        # Copy manual annotations (priority)
        if manual_dir and manual_dir.exists():
            print(f"\n📥 Manual annotations: {manual_dir}")
            self._copy_annotations(manual_dir, merged_images, merged_labels, "manual")
        
        # Copy automatic annotations (don't overwrite manual)
        if auto_dir.exists():
            print(f"\n📥 Automatic annotations: {auto_dir}")
            auto_images = auto_dir / "images"
            auto_labels = auto_dir / "labels"
            
            if auto_images.exists() and auto_labels.exists():
                self._copy_annotations_skip_existing(
                    auto_images, auto_labels, 
                    merged_images, merged_labels
                )
        
        self.stats['total_images'] = count_images(merged_images)
        
        print(f"\n📊 Merge Results:")
        print(f"   Manual: {self.stats['manual_images']}")
        print(f"   Automatic: {self.stats['auto_images']}")
        print(f"   Total: {self.stats['total_images']}")
        
        return merged_images, merged_labels
    
    def _copy_annotations(self, source_dir: Path, 
                          images_out: Path, labels_out: Path,
                          source_type: str = ""):
        """Copy annotations"""
        
        # Support different structures
        possible_image_dirs = [
            source_dir / "images",
            source_dir / "train" / "images",
            source_dir
        ]
        
        possible_label_dirs = [
            source_dir / "labels",
            source_dir / "train" / "labels",
            source_dir
        ]
        
        images_found = []
        labels_found = []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Find images
        for img_dir in possible_image_dirs:
            if img_dir.exists():
                for ext in image_extensions:
                    images_found.extend(img_dir.glob(f"*{ext}"))
        
        # Find labels
        for lbl_dir in possible_label_dirs:
            if lbl_dir.exists():
                labels_found.extend(lbl_dir.glob("*.txt"))
        
        # Filter
        labels_found = [l for l in labels_found 
                       if l.name not in ['classes.txt', 'data.yaml', 'README.txt']]
        
        # Copy only matching pairs
        label_stems = {l.stem for l in labels_found}
        
        count = 0
        for img in tqdm(images_found, desc=f"Copying ({source_type})"):
            if img.stem in label_stems:
                # Copy image
                shutil.copy(img, images_out / img.name)
                
                # Find and copy label
                for lbl in labels_found:
                    if lbl.stem == img.stem:
                        shutil.copy(lbl, labels_out / lbl.name)
                        break
                
                count += 1
        
        if source_type == "manual":
            self.stats['manual_images'] = count
        
        print(f"   {count} images copied")
    
    def _copy_annotations_skip_existing(self,
                                         images_in: Path,
                                         labels_in: Path,
                                         images_out: Path,
                                         labels_out: Path):
        """Copy annotations, skip existing"""
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images_found = []
        
        for ext in image_extensions:
            images_found.extend(images_in.glob(f"*{ext}"))
        
        count = 0
        skipped = 0
        
        for img in tqdm(images_found, desc="Copying (automatic)"):
            label_in = labels_in / (img.stem + '.txt')
            
            if not label_in.exists():
                continue
            
            # Check if already exists
            if (images_out / img.name).exists():
                skipped += 1
                continue
            
            # Copy
            shutil.copy(img, images_out / img.name)
            shutil.copy(label_in, labels_out / label_in.name)
            count += 1
        
        self.stats['auto_images'] = count
        print(f"   {count} images copied, {skipped} skipped (already exists)")
    
    def prepare_yolov11_dataset(self,
                                 merged_images: Path,
                                 merged_labels: Path,
                                 output_dir: Path = None):
        """
        Prepare dataset in YOLOv11 format
        
        Args:
            merged_images: Merged images directory
            merged_labels: Merged labels directory
            output_dir: Output directory
        """
        output_dir = output_dir or config.FINAL_DATASET_DIR
        
        print("\n" + "="*50)
        print("📦 PREPARING YOLOv11 DATASET")
        print("="*50)
        
        # Clean existing final dataset
        if output_dir.exists():
            print(f"\n⚠️ Cleaning existing dataset: {output_dir}")
            shutil.rmtree(output_dir)
        
        # Split dataset
        print("\n📂 Splitting dataset...")
        split_dataset(
            images_dir=merged_images,
            labels_dir=merged_labels,
            output_dir=output_dir,
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
            test_ratio=config.TEST_RATIO
        )
        
        # Update statistics
        self.stats['train'] = count_images(output_dir / "train" / "images")
        self.stats['val'] = count_images(output_dir / "val" / "images")
        self.stats['test'] = count_images(output_dir / "test" / "images")
        
        # Create dataset YAML
        class_names = list(config.CLASSES.values())
        yaml_path = create_dataset_yaml(output_dir, class_names)
        
        # Save summary info
        self._save_dataset_info(output_dir)
        
        print(f"\n✅ Dataset ready: {output_dir}")
        return output_dir
    
    def _save_dataset_info(self, output_dir: Path):
        """Save dataset info"""
        
        info = {
            'created': datetime.now().isoformat(),
            'statistics': self.stats,
            'classes': config.CLASSES,
            'split_ratios': {
                'train': config.TRAIN_RATIO,
                'val': config.VAL_RATIO,
                'test': config.TEST_RATIO
            },
            'model_config': {
                'image_size': config.IMAGE_SIZE,
                'yolo_model': config.YOLO_SEG_MODEL
            }
        }
        
        info_file = output_dir / "dataset_info.yaml"
        with open(info_file, 'w') as f:
            yaml.dump(info, f, default_flow_style=False)
        
        print(f"📄 Dataset info: {info_file}")
    
    def print_summary(self):
        """Print summary"""
        print("\n" + "="*50)
        print("📊 DATASET SUMMARY")
        print("="*50)
        print(f"Manual Annotations: {self.stats['manual_images']}")
        print(f"Automatic Annotations: {self.stats['auto_images']}")
        print(f"Total Images: {self.stats['total_images']}")
        print(f"\nSplit:")
        print(f"   Train: {self.stats['train']}")
        print(f"   Val: {self.stats['val']}")
        print(f"   Test: {self.stats['test']}")


def main():
    """Main script"""
    
    print("\n" + "="*60)
    print("📦 FINAL DATASET PREPARATION")
    print("="*60)
    
    preparer = FinalDatasetPreparer()
    
    # Get directories from user
    print("\n📂 Specify manual annotation directory (Roboflow export)")
    manual_dir = input("Directory path (or Enter to skip): ").strip()
    
    manual_dir = Path(manual_dir) if manual_dir else None
    
    # Merge
    merged_images, merged_labels = preparer.merge_annotations(
        manual_dir=manual_dir
    )
    
    # Prepare YOLOv11 dataset
    preparer.prepare_yolov11_dataset(merged_images, merged_labels)
    
    # Summary
    preparer.print_summary()
    
    print("\n" + "="*50)
    print("💡 NEXT STEP")
    print("="*50)
    print("To train final model:")
    print("   python train_model.py")
    print(f"\nDataset location: {config.FINAL_DATASET_DIR}")


if __name__ == "__main__":
    main()
