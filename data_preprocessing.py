import yaml
import logging
import sys
import numpy as np
from pathlib import Path

from data_preprocessing.augment_images import  augment_dataset
from data_preprocessing.convert_labels import  create_txt_files, create_txt_files_coco_format
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

    # bbox_annotations = config['data']['bbox_training_annotations']
    segment_annotations = config['data']['segment_training_annotations']
    labeltype = config['data']['labeltype']
    aug_config = config.get("augmentation", {})
    input_dir = "data/train_images/"
    output_dir = "data/augmented_images"
    augmented_images = augment_dataset(input_dir, output_dir, aug_config, logger)

    if labeltype == 'coco':
        # create_txt_files_coco_format(bbox_annotations, 'data/labels/', logger=logger)
        create_txt_files_coco_format(segment_annotations, 'data/labels/', logger=logger)
    if labeltype == 'yolo':
        # create_txt_files(bbox_annotations, 'data/labels/', logger=logger)
        create_txt_files(segment_annotations, 'data/labels/', logger=logger)
    
    # split_dataset(config['data']['segmentation_train'], 'data/labels/', config_path="config.yaml", logger=logger)

if __name__ == "__main__":
    main()