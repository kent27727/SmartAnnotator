"""
Server Training Script
======================
Project: eyefinelines_annotated
Task: SEGMENTATION
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import config
from train_model import ModelTrainer

def main():
    print("=" * 60)
    print("🎯 EYEFINELINES_ANNOTATED - SERVER TRAINING")
    print("=" * 60)
    print(f"Task: {config.MODEL_TASK.upper()}")
    print(f"Model: {config.YOLO_SEG_MODEL}")
    
    config.create_directories()
    
    # Dataset info
    train_dir = config.FINAL_DATASET_DIR / "train" / "images"
    val_dir = config.FINAL_DATASET_DIR / "val" / "images"
    
    train_count = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
    val_count = len(list(val_dir.glob("*.jpg"))) if val_dir.exists() else 0
    
    print(f"\nDataset: Train={train_count}, Val={val_count}")
    print(f"Model: {config.YOLO_SEG_MODEL}")
    print(f"Epochs: {config.INITIAL_TRAINING_EPOCHS}")
    
    response = input("\nStart training? (y/n): ")
    if response.lower() == 'y':
        trainer = ModelTrainer()
        trainer.train(epochs=config.INITIAL_TRAINING_EPOCHS)
        
        response = input("\nRun evaluation? (y/n): ")
        if response.lower() == 'y':
            trainer.evaluate()

if __name__ == "__main__":
    main()
