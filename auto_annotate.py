"""
Auto-Annotation Tool - Auto Annotator
======================================
Automatic labeling with trained model
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import shutil

import cv2
import numpy as np
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics not installed!")
    print("Install with: pip install ultralytics")
    sys.exit(1)

import config
from utils import (
    save_yolo_segmentation_label,
    mask_to_polygon,
    get_unlabeled_images,
    visualize_annotation,
    count_images
)


class AutoAnnotator:
    """Automatic labeling engine"""
    
    def __init__(self, model_path: Path = None, min_detections: int = None):
        """
        Args:
            model_path: Trained model path
            min_detections: Minimum detection count per image (default: 1)
        """
        if model_path is None:
            model_path = config.MODELS_DIR / "latest_model.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Train a model first with train_model.py!"
            )
        
        print(f"📥 Loading model: {model_path}")
        self.model = YOLO(str(model_path))
        self.model_path = model_path
        
        # Minimum detection count - from project config or parameter
        if min_detections is not None:
            self.min_detections = max(1, min_detections)  # Minimum 1
        else:
            # Get from project config
            project_config = config.get_active_project_config()
            if project_config:
                self.min_detections = project_config.get("annotation", {}).get("min_detections", 1)
            else:
                self.min_detections = 1  # Default: 1 detection
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'high_confidence': 0,
            'low_confidence': 0,
            'no_detection': 0,
            'insufficient_detection': 0,  # Less than min_detections
            'errors': 0
        }
        
        # Review list for low confidence predictions
        self.review_list = []
    
    def annotate_single(self, 
                        image_path: Path,
                        output_label_path: Path = None,
                        confidence_threshold: float = None,
                        save_visualization: bool = False) -> Dict:
        """
        Label a single image
        
        Args:
            image_path: Image file path
            output_label_path: Output label file path
            confidence_threshold: Minimum confidence threshold
            save_visualization: Save visualization
            
        Returns:
            Annotation results dict
        """
        confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return {'success': False, 'error': 'Image could not be read'}
        
        h, w = image.shape[:2]
        
        # Predict
        results = self.model.predict(
            source=str(image_path),
            conf=confidence_threshold,
            iou=config.IOU_THRESHOLD,
            verbose=False,
            retina_masks=True  # Higher quality mask
        )
        
        annotations = []
        confidences = []
        
        if results and len(results) > 0:
            result = results[0]
            
            # Process segmentation masks
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes
                
                for i, mask in enumerate(masks):
                    # Confidence value
                    conf = float(boxes.conf[i])
                    class_id = int(boxes.cls[i])
                    
                    confidences.append(conf)
                    
                    # Resize mask to original size
                    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                    
                    # Convert mask to polygon
                    polygon = mask_to_polygon(mask_binary, normalize=True)
                    
                    if len(polygon) >= 3:  # Minimum 3 points required
                        annotations.append({
                            'class_id': class_id,
                            'polygon': polygon,
                            'confidence': conf
                        })
        
        # ===== DETECTION COUNT CHECK =====
        num_detections = len(annotations)
        is_valid = num_detections >= self.min_detections  # Minimum detection check
        
        # Result
        result_data = {
            'success': True,
            'image_path': str(image_path),
            'annotations': annotations,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'num_detections': num_detections,
            'is_valid': is_valid
        }
        
        # Save label file - ONLY if minimum detections met
        if output_label_path and annotations and is_valid:
            output_label_path.parent.mkdir(parents=True, exist_ok=True)
            save_yolo_segmentation_label(output_label_path, annotations)
            result_data['label_path'] = str(output_label_path)
        
        # Visualization - ONLY if minimum detections met
        if save_visualization and output_label_path and annotations and is_valid:
            vis_path = output_label_path.parent.parent / "visualizations" / (image_path.stem + "_vis.jpg")
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            visualize_annotation(image_path, output_label_path, vis_path)
            result_data['visualization_path'] = str(vis_path)
        
        return result_data
    
    def annotate_batch(self,
                       images_dir: Path = None,
                       output_dir: Path = None,
                       confidence_threshold: float = None,
                       skip_labeled: bool = True,
                       save_visualizations: bool = False,
                       max_images: int = None) -> Dict:
        """
        Batch automatic labeling
        
        Args:
            images_dir: Directory containing images
            output_dir: Output directory
            confidence_threshold: Minimum confidence threshold
            skip_labeled: Skip already labeled images
            save_visualizations: Save visualizations
            max_images: Maximum number of images (None = all)
            
        Returns:
            Processing results
        """
        images_dir = images_dir or config.RAW_IMAGES_DIR
        output_dir = output_dir or config.AUTO_ANNOTATIONS_DIR
        confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
        
        print("\n" + "="*60)
        print("🤖 AUTO ANNOTATION STARTING")
        print("="*60)
        
        # Create output directories
        labels_dir = output_dir / "labels"
        images_out_dir = output_dir / "images"
        unvalid_dir = output_dir / "unvalid"  # Invalid images
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_out_dir.mkdir(parents=True, exist_ok=True)
        unvalid_dir.mkdir(parents=True, exist_ok=True)
        
        # Find images to label
        if skip_labeled:
            images_to_process = get_unlabeled_images(images_dir, labels_dir)
            print(f"📊 Unlabeled images: {len(images_to_process)}")
        else:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
            images_to_process = []
            for ext in image_extensions:
                images_to_process.extend(images_dir.glob(f"*{ext}"))
            print(f"📊 Total images: {len(images_to_process)}")
        
        if max_images:
            images_to_process = images_to_process[:max_images]
            print(f"   (Processing first {max_images} images)")
        
        if not images_to_process:
            print("✅ No images to label!")
            return self.stats
        
        print(f"\n⚙️ Settings:")
        print(f"   Confidence Threshold: {confidence_threshold}")
        print(f"   Minimum Detections: {self.min_detections}")
        print(f"   Output Directory: {output_dir}")
        print(f"   Save Visualizations: {save_visualizations}")
        
        # Process each image
        print(f"\n🏃 Processing...")
        
        self.review_list = []
        
        for img_path in tqdm(images_to_process, desc="Labeling"):
            try:
                # Label file path
                label_path = labels_dir / (img_path.stem + '.txt')
                
                # Label
                result = self.annotate_single(
                    image_path=img_path,
                    output_label_path=label_path,
                    confidence_threshold=confidence_threshold,
                    save_visualization=save_visualizations
                )
                
                self.stats['total_processed'] += 1
                
                if result['success']:
                    num_det = result['num_detections']
                    
                    if result.get('is_valid', False):  # Minimum detections met - valid
                        # Copy image
                        shutil.copy(img_path, images_out_dir / img_path.name)
                        
                        # Confidence check
                        if result['avg_confidence'] >= config.LOW_CONFIDENCE_THRESHOLD:
                            self.stats['high_confidence'] += 1
                        else:
                            self.stats['low_confidence'] += 1
                            self.review_list.append({
                                'image': str(img_path),
                                'label': str(label_path),
                                'confidence': result['avg_confidence']
                            })
                    
                    elif num_det > 0 and num_det < self.min_detections:
                        # Insufficient detections - invalid, copy to unvalid
                        self.stats['insufficient_detection'] += 1
                        shutil.copy(img_path, unvalid_dir / img_path.name)
                    
                    else:
                        # No detections - copy to unvalid
                        self.stats['no_detection'] += 1
                        shutil.copy(img_path, unvalid_dir / img_path.name)
                else:
                    self.stats['errors'] += 1
                    
            except Exception as e:
                self.stats['errors'] += 1
                print(f"\n⚠️ Error ({img_path.name}): {e}")
        
        # Print results
        self._print_stats()
        
        # Save review list
        if self.review_list:
            review_file = output_dir / "review_list.json"
            with open(review_file, 'w') as f:
                json.dump(self.review_list, f, indent=2)
            print(f"\n📝 Review list saved: {review_file}")
        
        return self.stats
    
    def _print_stats(self):
        """Print statistics"""
        print("\n" + "="*50)
        print("📊 AUTO ANNOTATION RESULTS")
        print("="*50)
        print(f"Total Processed: {self.stats['total_processed']}")
        print(f"✅ High Confidence ({self.min_detections}+ detections): {self.stats['high_confidence']}")
        print(f"⚠️ Low Confidence ({self.min_detections}+ detections): {self.stats['low_confidence']} (review needed)")
        if self.min_detections > 1:
            print(f"👁️ Insufficient Detections (< {self.min_detections}): {self.stats['insufficient_detection']}")
        print(f"❌ No Detection: {self.stats['no_detection']}")
        print(f"🔴 Errors: {self.stats['errors']}")
        
        labeled = self.stats['high_confidence'] + self.stats['low_confidence']
        unlabeled = self.stats['insufficient_detection'] + self.stats['no_detection']
        
        print(f"\n📁 Labeled: {labeled} images (images/ and labels/)")
        print(f"📁 Unlabeled: {unlabeled} images (unvalid/)")
        
        if self.stats['total_processed'] > 0:
            success_rate = labeled / self.stats['total_processed'] * 100
            print(f"\n📈 Labeling Rate: {success_rate:.1f}%")
    
    def interactive_review(self, review_file: Path = None):
        """
        Interactively review low confidence predictions
        
        Args:
            review_file: Review list JSON file
        """
        if review_file is None:
            review_file = config.AUTO_ANNOTATIONS_DIR / "review_list.json"
        
        if not review_file.exists():
            print("Review list not found!")
            return
        
        with open(review_file, 'r') as f:
            review_list = json.load(f)
        
        print(f"\n📝 {len(review_list)} images waiting for review...")
        print("\nCommands:")
        print("  [Enter] - Accept and continue")
        print("  [d] - Delete (remove annotation)")
        print("  [s] - Skip")
        print("  [q] - Quit")
        
        approved = 0
        deleted = 0
        skipped = 0
        
        for i, item in enumerate(review_list):
            print(f"\n[{i+1}/{len(review_list)}] {Path(item['image']).name}")
            print(f"   Confidence: {item['confidence']:.2f}")
            
            # Show image
            try:
                img_path = Path(item['image'])
                label_path = Path(item['label'])
                
                if img_path.exists() and label_path.exists():
                    annotated = visualize_annotation(img_path, label_path, show=False)
                    
                    cv2.imshow("Review - Press key to continue", annotated)
                    key = cv2.waitKey(0) & 0xFF
                    cv2.destroyAllWindows()
                    
                    if key == ord('q'):
                        break
                    elif key == ord('d'):
                        # Delete label file
                        label_path.unlink()
                        deleted += 1
                        print("   ❌ Deleted")
                    elif key == ord('s'):
                        skipped += 1
                        print("   ⏭️ Skipped")
                    else:
                        approved += 1
                        print("   ✅ Approved")
                        
            except Exception as e:
                print(f"   Error: {e}")
                skipped += 1
        
        print(f"\n📊 Review Results:")
        print(f"   Approved: {approved}")
        print(f"   Deleted: {deleted}")
        print(f"   Skipped: {skipped}")


class SemiSupervisedPipeline:
    """
    Semi-supervised learning pipeline
    
    1. Manual annotation (200-300 samples)
    2. Initial model training
    3. Auto annotation
    4. Human review (optional)
    5. Final model training
    """
    
    def __init__(self):
        self.trainer = None
        self.annotator = None
        
    def run_full_pipeline(self,
                          roboflow_export_dir: Path,
                          raw_images_dir: Path = None,
                          initial_epochs: int = None,
                          confidence_threshold: float = None):
        """
        Run full pipeline
        
        Args:
            roboflow_export_dir: Roboflow export directory
            raw_images_dir: Unlabeled images directory
            initial_epochs: Initial training epochs
            confidence_threshold: Auto-annotation confidence threshold
        """
        from train_model import DarkCircleTrainer
        
        raw_images_dir = raw_images_dir or config.RAW_IMAGES_DIR
        initial_epochs = initial_epochs or config.INITIAL_TRAINING_EPOCHS
        confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
        
        print("\n" + "="*60)
        print("🚀 SEMI-SUPERVISED LEARNING PIPELINE")
        print("="*60)
        
        # 1. Prepare training data
        print("\n📌 STEP 1: Prepare training data")
        self.trainer = DarkCircleTrainer()
        success = self.trainer.prepare_training_data(
            roboflow_export_dir=roboflow_export_dir
        )
        
        if not success:
            print("❌ Data preparation failed!")
            return
        
        # 2. Train initial model
        print("\n📌 STEP 2: Initial model training")
        self.trainer.train(epochs=initial_epochs)
        
        # 3. Auto annotation
        print("\n📌 STEP 3: Auto annotation")
        self.annotator = AutoAnnotator()
        self.annotator.annotate_batch(
            images_dir=raw_images_dir,
            confidence_threshold=confidence_threshold,
            save_visualizations=True
        )
        
        # 4. Summary
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE")
        print("="*60)
        print(f"\n📁 Auto annotations: {config.AUTO_ANNOTATIONS_DIR}")
        print(f"📁 Final dataset: {config.FINAL_DATASET_DIR}")
        print(f"\n💡 Next steps:")
        print("   1. Check results in auto_annotations folder")
        print("   2. Fix or delete incorrect annotations")
        print("   3. Run prepare_final_dataset.py for final model training")


def main():
    """Main auto annotation script"""
    
    print("\n" + "="*60)
    print("🤖 AUTO-ANNOTATION TOOL")
    print("="*60)
    
    # Model check
    model_path = config.MODELS_DIR / "latest_model.pt"
    
    if not model_path.exists():
        print(f"\n❌ Trained model not found: {model_path}")
        print("\n💡 Train a model first:")
        print("   python train_model.py")
        return
    
    # Create annotator
    annotator = AutoAnnotator(model_path)
    
    # Options
    print("\nOptions:")
    print("  [1] Auto-label all images")
    print("  [2] Label specific number of images (test)")
    print("  [3] Review low confidence predictions")
    print("  [4] Label single image")
    
    choice = input("\nYour choice (1-4): ").strip()
    
    if choice == '1':
        annotator.annotate_batch(
            images_dir=config.RAW_IMAGES_DIR,
            save_visualizations=True
        )
        
    elif choice == '2':
        n = input("How many images? (default: 10): ").strip()
        n = int(n) if n else 10
        
        annotator.annotate_batch(
            images_dir=config.RAW_IMAGES_DIR,
            max_images=n,
            save_visualizations=True
        )
        
    elif choice == '3':
        annotator.interactive_review()
        
    elif choice == '4':
        img_path = input("Image path: ").strip()
        if img_path:
            result = annotator.annotate_single(
                image_path=Path(img_path),
                output_label_path=config.AUTO_ANNOTATIONS_DIR / "labels" / (Path(img_path).stem + '.txt'),
                save_visualization=True
            )
            print(f"\nResult: {result}")
    
    print("\n✅ Process complete!")


if __name__ == "__main__":
    main()
