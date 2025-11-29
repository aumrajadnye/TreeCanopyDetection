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

# config_file = '.py'
# checkpoint_file = 'fcn_unet_s5-d16_128x128_40k_stare_20201223_191051-7d77e78b.pth'

# # build the model from a config file and a checkpoint file
# model = init_model(config_file, checkpoint_file, device='cpu')

# # test a single image and show the results
# img = 'demo.png'  # or img = mmcv.imread(img), which will only load it once
# result = inference_model(model, img)
# # visualize the results in a new window or save the visualization results to image files
# show_result_pyplot(model, img, result, show=True)

# save as sample_check_dataset.py and run: python sample_check_dataset.py
# save as check_masks.py and run: python check_masks.py