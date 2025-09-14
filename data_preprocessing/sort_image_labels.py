import os
import shutil
import random
import yaml
from pathlib import Path
import logging

def split_dataset(image_dir, label_dir, config_path="../config.yaml", logger=None):
    # Initialize logger
    logger = logging.getLogger(__name__) if logger is None else logger
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    split_ratio = config['prelim']['train_validation_split']
    if not (0 < split_ratio < 1):
        raise ValueError("train_val_split must be between 0 and 1")

    # Ensure input paths exist
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    assert image_dir.exists(), f"Image directory {image_dir} does not exist"
    assert label_dir.exists(), f"Label directory {label_dir} does not exist"

    # Collect and shuffle image files
    image_files = sorted([f for f in image_dir.glob("*.tif")])
    random.shuffle(image_files)

    # Split into train and val
    split_index = int(len(image_files) * split_ratio)
    train_images = image_files[:split_index]
    val_images = image_files[split_index:]

    # YOLO-style output paths
    base_path = Path("data/training_data_object_detection")
    paths = {
        "images/train": base_path / "images/train",
        "images/val": base_path / "images/val",
        "labels/train": base_path / "labels/train",
        "labels/val": base_path / "labels/val",
    }

    # Create all necessary directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    def copy_files(images, split):
        for img_path in images:
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                logger.warning(f"No label for {img_path.name}, skipping.")
                continue

            # Copy image and label
            shutil.copy(img_path, paths[f"images/{split}"])
            shutil.copy(label_path, paths[f"labels/{split}"])

    # Perform the copying
    copy_files(train_images, "train")
    copy_files(val_images, "val")

    logger.info(f"Split complete. Train: {len(train_images)} images, Val: {len(val_images)} images.")
