import yaml
import logging
import sys
import random
import numpy as np

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

    bbox_annotations = config['data']['bbox_training_annotations']
    segment_annotations = config['data']['segment_training_annotations']
    labeltype = config['data']['labeltype']
    aug_cfg = config.get("augmentation", {})

    def augment_image(image):
        # Gamma correction
        if aug_cfg.get("gamma", {}).get("apply", False):
            gamma_min, gamma_max = aug_cfg["gamma"]["range"]
            gamma = random.uniform(gamma_min, gamma_max)
            image = np.power(image / 255.0, gamma)
            image = np.clip(image * 255, 0, 255).astype(np.uint8)

        # Contrast adjustment
        if aug_cfg.get("contrast", {}).get("apply", False):
            contrast_min, contrast_max = aug_cfg["contrast"]["range"]
            contrast = random.uniform(contrast_min, contrast_max)
            mean = np.mean(image)
            image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
        return image

    if labeltype == 'coco':
        # create_txt_files_coco_format(bbox_annotations, 'data/labels/', logger=logger)
        create_txt_files_coco_format(segment_annotations, 'data/labels/', logger=logger)
    if labeltype == 'yolo':
        create_txt_files(bbox_annotations, 'data/labels/', logger=logger)
        create_txt_files(segment_annotations, 'data/labels/', logger=logger)
    
    split_dataset(config['data']['bbox_train'], 'data/labels/', config_path="config.yaml", logger=logger)

if __name__ == "__main__":
    main()