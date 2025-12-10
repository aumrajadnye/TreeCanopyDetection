import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("build_outputs/scene_classifier.h5")
classes = np.load("build_outputs/label_encoder.npy", allow_pickle=True)

def predict_scene(img_path):
    img = image.load_img(img_path, target_size=(380, 380))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    pred = model.predict(x)
    label_idx = np.argmax(pred)
    return str(classes[label_idx])

# {
#     "images": [
#         {
#             "file_name": "10cm_evaluation_1.tif",
#             "scene_type": "industrial_area"
#         }...
#     ]
# }
images_list = []
folder = "data/evaluation_images"
for filename in os.listdir(folder):
    if filename.lower().endswith((".tif", ".png", ".jpg", ".jpeg")):
        images_list.append({
            "file_name": filename,
            "scene_type": predict_scene(os.path.join(folder,filename))
        })

output = {'images': images_list}
with open("build_outputs/evaluation_annotations.json", "w") as f:
    json.dump(output, f, indent=4)
