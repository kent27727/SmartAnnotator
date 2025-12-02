"""
YOLOv11 Segmentation Model Test Script
=======================================
Test your trained segmentation model on images

Usage:
    python test.py --image_path test.jpg --model_path eyefineline.pt
    python test.py --image_path test.jpg --model_path eyefineline.pt --conf 0.5
    python test.py --image_path ./images/ --model_path eyefineline.pt  # For folder
    python test.py --image_path test.jpg --model_path eyefineline.pt --save --no-show
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install ultralytics opencv-python")
    sys.exit(1)


def test_model(image_path: str, model_path: str, conf: float = 0.25, iou: float = 0.3,
               save: bool = False, show: bool = True, output_dir: str = "results"):
    """
    Test segmentation model on image(s)
    
    Args:
        image_path: Path to image or folder
        model_path: Path to .pt model file
        conf: Confidence threshold (0.0 - 1.0)
        iou: IoU threshold for NMS - lower = less overlap allowed (0.0 - 1.0)
        save: Save results to file
        show: Display results
        output_dir: Directory to save results
    """
    
    # Check paths
    image_path = Path(image_path)
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    if not image_path.exists():
        print(f"❌ Image/folder not found: {image_path}")
        return
    
    # Load model
    print(f"\n🤖 Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # Get model info
    print(f"   Task: {model.task}")
    if hasattr(model, 'names'):
        print(f"   Classes: {model.names}")
    
    # Prepare output directory
    if save:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Results will be saved to: {output_path}")
    
    # Get image list
    if image_path.is_dir():
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        images = []
        for ext in extensions:
            images.extend(image_path.glob(f"*{ext}"))
            images.extend(image_path.glob(f"*{ext.upper()}"))
        print(f"\n📷 Found {len(images)} images in folder")
    else:
        images = [image_path]
    
    if not images:
        print("❌ No images found!")
        return
    
    # Process each image
    print(f"\n🔍 Running inference (conf={conf})...")
    print("-" * 50)
    
    for img_path in images:
        print(f"\n📷 Processing: {img_path.name}")
        
        # Run inference with NMS to remove overlapping detections
        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,              # Lower IoU = less overlap allowed (default 0.7)
            agnostic_nms=True,    # Apply NMS across all classes
            max_det=100,          # Max detections per image
            save=False,
            verbose=False
        )
        
        result = results[0]
        
        # Print detections
        if result.boxes is not None and len(result.boxes) > 0:
            print(f"   ✅ Found {len(result.boxes)} detection(s):")
            
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id] if hasattr(model, 'names') else f"class_{cls_id}"
                confidence = float(box.conf[0])
                
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                print(f"      [{i+1}] {cls_name}: {confidence:.2%}")
                print(f"          Box: ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")
                
                # Mask info (for segmentation)
                if result.masks is not None and i < len(result.masks):
                    mask = result.masks[i]
                    if hasattr(mask, 'xy') and mask.xy is not None:
                        points = len(mask.xy[0]) if len(mask.xy) > 0 else 0
                        print(f"          Mask: {points} polygon points")
        else:
            print(f"   ⚠️ No detections found")
        
        # Visualize (only masks, no boxes/labels)
        annotated_frame = result.plot(
            boxes=False,      # Hide bounding boxes
            labels=False,     # Hide labels
            conf=False        # Hide confidence scores
        )
        
        # Save result
        if save:
            save_path = Path(output_dir) / f"result_{img_path.name}"
            cv2.imwrite(str(save_path), annotated_frame)
            print(f"   💾 Saved: {save_path}")
        
        # Show result
        if show:
            # Resize if too large
            h, w = annotated_frame.shape[:2]
            max_size = 1200
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                annotated_frame = cv2.resize(annotated_frame, None, fx=scale, fy=scale)
            
            cv2.imshow(f"Result - {img_path.name}", annotated_frame)
            
            print(f"   👁️ Press any key to continue, 'q' to quit...")
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            
            if key == ord('q'):
                print("\n👋 Quit requested")
                break
    
    print("\n" + "=" * 50)
    print("✅ Testing complete!")
    if save:
        print(f"📁 Results saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Test YOLOv11 Segmentation Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py --image_path test.jpg --model_path eyefineline.pt
  python test.py --image_path test.jpg --model_path eyefineline.pt --conf 0.5
  python test.py --image_path ./images/ --model_path eyefineline.pt --save
  python test.py --image_path test.jpg --model_path eyefineline.pt --save --no-show
        """
    )
    
    parser.add_argument(
        "--image_path", "-i",
        type=str,
        required=True,
        help="Path to image or folder containing images"
    )
    
    parser.add_argument(
        "--model_path", "-m",
        type=str,
        required=True,
        help="Path to trained model (.pt file)"
    )
    
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.3,
        help="IoU threshold for NMS - lower = less overlap (default: 0.3)"
    )
    
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save results to output directory"
    )
    
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display results (useful for batch processing)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory for saved results (default: results)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 50)
    print("🎯 YOLO SEGMENTATION MODEL TEST")
    print("=" * 50)
    
    test_model(
        image_path=args.image_path,
        model_path=args.model_path,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        show=not args.no_show,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()

