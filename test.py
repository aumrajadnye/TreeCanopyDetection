from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

config_file = 'configs/trcnpy_cnfg.py'
checkpoint_file = 'work_dirs/trcnpy_cnfg/iter_200.pth'

# Initialize the model
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# Input image
# img = 'data/training_data_object_detection/images/test/10cm_evaluation_1.png'

img = 'data/training_data_object_detection/images/val/10cm_train_10.png'
overlay_path = 'data/training_data_object_detection/seg_overlays/val/10cm_train_10.png'

image = np.array(Image.open(img))
mask = np.array(Image.open(overlay_path))
# Inference
result = inference_segmentor(model, img)
pred_mask = result[0] 

# show_result_pyplot(model, img, result)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original image
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Mask
axes[1].imshow(mask)
axes[1].set_title('Expected result')
axes[1].axis('off')

# Overlay
axes[2].imshow(pred_mask, cmap='tab20')
axes[2].set_title('Actual result')
axes[2].axis('off')

plt.show()

# -------------------------------
# # check if can be converted to coco
# # coco requires valid polygons for segmentation
# import json

# path = "data/train_annotations/train.json"

# with open(path, "r") as f:
#     data = json.load(f)

# valid_count = 0
# odd_count = 0

# def is_polygon(seg):
#     return len(seg) % 2 == 0

# for img in data["images"]:
#     for ann in img.get("annotations", []):
#         seg = ann.get("segmentation", [])
#         if is_polygon(seg):
#             valid_count += 1
#         else:
#             odd_count += 1

# print("Valid polygon segmentation count:", valid_count)
# print("Odd-length (invalid) segmentation count:", odd_count)

# -------------------------------------------------

# from mmseg.apis import inference_model, init_model, show_result_pyplot
# import mmcv

# config_file = 'configs/trcnpy_cnfg.py'
# checkpoint_file = 'work_dirs/iter_4000.pth'

# # build the model from a config file and a checkpoint file
# model = init_model(config_file, checkpoint_file, device='cuda:0')

# # test a single image and show the results
# img = 'data/training_data_object_detection/images/test/10cm_evaluation_1.png'  # or img = mmcv.imread(img), which will only load it once
# result = inference_model(model, img)
# # visualize the results in a new window or save the visualization results to image files
# show_result_pyplot(model, img, result, show=True)

# save as sample_check_dataset.py and run: python sample_check_dataset.py
# save as check_masks.py and run: python check_masks.py