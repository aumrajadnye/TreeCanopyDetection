import os
import json 

def load(json_file="data/train_annotations.json", img_path="data/train_images"):
    with open(json_file) as f:
        data = json.load(f)

    img = []
    scene = []

    for item in data["images"]:
        img.append(os.path.join(img_path, item["file_name"]))
        scene.append(item["scene_type"])
    return img, scene
