import shutil
import random
from pathlib import Path
import logging
import json
from PIL import Image

def split_dataset(image_dir, coco_json_path, output_base="data/training_data_object_detection", split_ratio=0.8, seed=42, logger=None):
    """
    Splits a dataset of images and COCO annotations into train and validation sets.

    Args:
        image_dir (str or Path): Folder containing all images.
        coco_json_path (str or Path): Path to COCO annotations JSON file.
        output_base (str or Path): Base output folder to store split images and annotations.
        split_ratio (float): Ratio of train images (e.g., 0.8 for 80% train, 20% val).
        seed (int): Random seed for reproducibility.
        logger (logging.Logger): Optional logger.
    """
    # Initialize logger
    logger = logging.getLogger(__name__) if logger is None else logger
    random.seed(seed)

    image_dir = Path(image_dir)
    coco_json_path = Path(coco_json_path)
    output_base = Path(output_base)

    # Load COCO JSON
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    # Collect image filenames in directory
    image_files = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif"]])

    # Shuffle and split images
    random.shuffle(image_files)
    split_point = int(len(image_files) * split_ratio)
    train_imgs = image_files[:split_point]
    val_imgs = image_files[split_point:]

    # Output folders
    img_train_out = output_base / "images/train"
    img_val_out   = output_base / "images/val"
    ann_out       = output_base / "annotations"

    img_train_out.mkdir(parents=True, exist_ok=True)
    img_val_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)

    def save_image(p, out_dir):
        """Convert .tif to .png and save, otherwise copy."""
        if p.suffix.lower() == ".tif":
            png_name = p.stem + ".png"
            out_path = out_dir / png_name
            im = Image.open(p)
            im.save(out_path, format="PNG")
            return png_name
        else:
            shutil.copy2(p, out_dir / p.name)
        return p.name

    # Copy images
    for p in train_imgs:
        save_image(p, img_train_out)

    for p in val_imgs:
        save_image(p, img_val_out)

    # image_files = sorted([p.with_suffix(".png") if p.suffix.lower() == ".tif" else p 
    #                   for p in image_dir.glob("*") 
    #                   if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif"]])
    # train_imgs = image_files[:split_point]
    # val_imgs = image_files[split_point:]
    def actual_saved_name(path):
        if path.suffix.lower() == ".tif":
            return path.stem + ".png"  # converted
        else:
            return path.name           # original

    train_fnames = { actual_saved_name(p) for p in img_train_out.glob("*") }
    val_fnames   = { actual_saved_name(p) for p in img_val_out.glob("*") }

    # Split COCO JSON
    train_images_json = [img for img in images if Path(img["file_name"]).name in train_fnames]
    val_images_json   = [img for img in images if Path(img["file_name"]).name in val_fnames]

    train_ids = set(img["id"] for img in train_images_json)
    val_ids   = set(img["id"] for img in val_images_json)

    train_annotations = [a for a in annotations if a["image_id"] in train_ids]
    val_annotations   = [a for a in annotations if a["image_id"] in val_ids]

    # Save train.json
    with open(ann_out / "train.json", "w") as f:
        json.dump({"images": train_images_json, "annotations": train_annotations, "categories": categories}, f, indent=4)

    # Save val.json
    with open(ann_out / "val.json", "w") as f:
        json.dump({"images": val_images_json, "annotations": val_annotations, "categories": categories}, f, indent=4)

    print(f"Done! Train images: {len(train_imgs)}, Val images: {len(val_imgs)}")
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
