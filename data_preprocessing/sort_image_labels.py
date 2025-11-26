import shutil
import random
from pathlib import Path
import logging
import json

def split_dataset(image_dir, coco_json_path, output_base="data/training_data_object_detection", split_ratio=0.8, logger=None):
    # Initialize logger
    logger = logging.getLogger(__name__) if logger is None else logger

    image_dir = Path(image_dir)
    coco_json_path = Path(coco_json_path)
    output_base = Path(output_base)

    # Load COCO JSON
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    # Collect image filenames available in directory
    image_files = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif"]])
    image_filenames = [p.name for p in image_files]

    # Shuffle and split images
    random.shuffle(image_files)
    split_point = int(len(image_files) * split_ratio)

    train_imgs = image_files[:split_point]
    val_imgs = image_files[split_point:]

    train_fnames = set([p.name for p in train_imgs])
    val_fnames = set([p.name for p in val_imgs])

    # Output folders
    img_train_out = output_base / "images/train"
    img_val_out   = output_base / "images/val"
    ann_out       = output_base / "annotations"

    img_train_out.mkdir(parents=True, exist_ok=True)
    img_val_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)

    # Copy images to new structure
    for p in train_imgs:
        shutil.copy(p, img_train_out)

    for p in val_imgs:
        shutil.copy(p, img_val_out)

    # Split COCO JSON
    train_images_json = []
    val_images_json = []
    train_ids = set()
    val_ids = set()

    for img in images:
        fname = Path(img["file_name"]).name
        if fname in train_fnames:
            train_images_json.append(img)
            train_ids.add(img["id"])
        elif fname in val_fnames:
            val_images_json.append(img)
            val_ids.add(img["id"])

    train_annotations = [a for a in annotations if a["image_id"] in train_ids]
    val_annotations   = [a for a in annotations if a["image_id"] in val_ids]

    # Save train.json
    train_json = {
        "images": train_images_json,
        "annotations": train_annotations,
        "categories": categories
    }
    with open(ann_out / "train.json", "w") as f:
        json.dump(train_json, f, indent=4)

    # Save val.json
    val_json = {
        "images": val_images_json,
        "annotations": val_annotations,
        "categories": categories
    }
    with open(ann_out / "val.json", "w") as f:
        json.dump(val_json, f, indent=4)

    print(f"Done!")
    print(f"Train images: {len(train_imgs)}")
    print(f"Val images:   {len(val_imgs)}")
    print(f"Saved train.json and val.json in: {ann_out}")
    logger.info(f"Split complete. Train: {len(train_imgs)} images, Val: {len(val_imgs)} images.")



# ---- HOW TO RUN ----
# split_images_and_coco(
#     image_dir="data/train_images",
#     coco_json_path="data/coco_annotations.json",
#     output_base="data/training_data_object_detection",
#     split_ratio=0.8,
#     logger=logger
# )




# def split_dataset(image_dir, label_dir, split_ratio, logger=None):
#     # Initialize logger
#     logger = logging.getLogger(__name__) if logger is None else logger
#     # Load configuration
#     # with open(config_path, "r") as file:
#     #     config = yaml.safe_load(file)
    
#     # split_ratio = config['prelim']['train_validation_split']
#     if not (0 < split_ratio < 1):
#         raise ValueError("train_val_split must be between 0 and 1")

#     # Ensure input paths exist
#     image_dir = Path(image_dir)
#     label_dir = Path(label_dir)
#     assert image_dir.exists(), f"Image directory {image_dir} does not exist"
#     assert label_dir.exists(), f"Label directory {label_dir} does not exist"

#     # Collect and shuffle image files
#     image_files = sorted([f for f in image_dir.glob("*.tif")])
#     random.shuffle(image_files)

#     # Split into train and val
#     split_index = int(len(image_files) * split_ratio)
#     train_images = image_files[:split_index]
#     val_images = image_files[split_index:]

#     # YOLO-style output paths
#     base_path = Path("data/training_data_object_detection")
#     paths = {
#         "images/train": base_path / "images/train",
#         "images/val": base_path / "images/val",
#         "labels/train": base_path / "labels/train",
#         "labels/val": base_path / "labels/val",
#     }

#     # Create all necessary directories
#     for path in paths.values():
#         path.mkdir(parents=True, exist_ok=True)

#     def copy_files(images, split):
#         for img_path in images:
#             label_path = label_dir / f"{img_path.stem}.txt"
#             if not label_path.exists():
#                 logger.warning(f"No label for {img_path.name}, skipping.")
#                 continue

#             # Copy image and label
#             shutil.copy(img_path, paths[f"images/{split}"])
#             shutil.copy(label_path, paths[f"labels/{split}"])

#     # Perform the copying
#     copy_files(train_images, "train")
#     copy_files(val_images, "val")

#     logger.info(f"Split complete. Train: {len(train_images)} images, Val: {len(val_images)} images.")
