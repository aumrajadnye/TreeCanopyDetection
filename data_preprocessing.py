import yaml
import logging
import sys

from data_preprocessing.convert_labels_to_coco import  create_txt_files, create_txt_files_coco_format
from data_preprocessing.sort_images_labels import split_dataset

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

    if labeltype == 'coco':
        create_txt_files_coco_format(bbox_annotations, 'data/labels/', logger=logger)
        create_txt_files_coco_format(segment_annotations, 'data/labels/', logger=logger)
    if labeltype == 'yolo':
        create_txt_files(bbox_annotations, 'data/labels/', logger=logger)
        create_txt_files(segment_annotations, 'data/labels/', logger=logger)
    
    split_dataset(config['data']['bbox_train'], 'data/labels/', config_path="config.yaml", logger=logger)

if __name__ == "__main__":
    main()