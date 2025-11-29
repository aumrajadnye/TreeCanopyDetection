import yaml
import logging
import sys
import numpy as np
from pathlib import Path

# from data_preprocessing.augment_images import  augment_dataset
from data_preprocessing.convert_labels import  convert_to_coco_format, create_segmentation_masks, create_mmseg_masks
from data_preprocessing.sort_image_labels import split_dataset

DEBUG_MODE = True

logger = logging.getLogger(__name__)
stdout_log_formatter = logging.Formatter('%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d | %(message)s')
stdout_log_handler = logging.StreamHandler(stream=sys.stdout)
stdout_log_handler.setFormatter(stdout_log_formatter)
if DEBUG_MODE:
    stdout_log_handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
else:
    stdout_log_handler.setLevel(logging.WARNING)
    logger.setLevel(logging.WARNING)
logger.addHandler(stdout_log_handler)

def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # add a file to check og type of annotation - rn we have a normal json 
    # file so will convert that to coco
    dataset_ann = "data/train_annotations.json"
    coco_ann = "data/coco_annotations.json"
    convert_to_coco_format(dataset_ann, coco_ann)

    split_dataset(
        image_dir="data/train_images",
        coco_json_path="data/coco_annotations.json",
        test_imgs="data/evaluation_images",
        output_base="data/training_data_object_detection", 
        split_ratio=0.8,
        logger=logger
    )

    create_segmentation_masks(base_dir="data/training_data_object_detection", logger=logger)
    create_mmseg_masks(base_dir="data/training_data_object_detection", logger=logger)

if __name__ == "__main__":
    main()