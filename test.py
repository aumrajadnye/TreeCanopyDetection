# check if can be converted to coco
# coco requires valid polygons for segmentation
import json

path = "data/train_annotations/train.json"

with open(path, "r") as f:
    data = json.load(f)

valid_count = 0
odd_count = 0

def is_polygon(seg):
    return len(seg) % 2 == 0

for img in data["images"]:
    for ann in img.get("annotations", []):
        seg = ann.get("segmentation", [])
        if is_polygon(seg):
            valid_count += 1
        else:
            odd_count += 1

print("Valid polygon segmentation count:", valid_count)
print("Odd-length (invalid) segmentation count:", odd_count)




# from mmseg.apis import inference_model, init_model, show_result_pyplot
# import mmcv

# config_file = 'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
# checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# # build the model from a config file and a checkpoint file
# model = init_model(config_file, checkpoint_file, device='cpu')

# # test a single image and show the results
# img = 'demo.png'  # or img = mmcv.imread(img), which will only load it once
# result = inference_model(model, img)
# # visualize the results in a new window
# # show_result_pyplot(model, img, result, show=True)
# # or save the visualization results to image files
# # you can change the opacity of the painted segmentation map in (0, 1].
# show_result_pyplot(model, img, result, show=True, out_file='result.jpg', opacity=0.5)

