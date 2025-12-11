from preprocessing.load_annotations import load
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocessing.data_generator import generator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np


image_paths, scene_labels = load(json_file="data/train_annotations.json", img_path="data/train_images")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(scene_labels)

train_paths, val_paths, y_train, y_val = train_test_split(
    image_paths, y, test_size=0.2, random_state=42, stratify=y
)

img_size = 380  # EfficientNetB4 recommended size
train_data = generator(train_paths, y_train)
val_data = generator(val_paths, y_val)

# ------------------------------------------------------
# Build Model (EfficientNetB4 + Classifier)
# ------------------------------------------------------

base_model = EfficientNetB4(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(img_size, img_size, 3)
)

base_model.trainable = False  # Freeze for first stage

model = models.Sequential([
    base_model,
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------------------------------
# Train First Stage (Frozen Base)
# ------------------------------------------------------
steps_train = len(train_paths) // 8
steps_val = len(val_paths) // 8

history = model.fit(
    train_data,
    validation_data=val_data,
    steps_per_epoch=steps_train,
    validation_steps=steps_val,
    epochs=5
)

# ------------------------------------------------------
# Fine-Tune Last 20 Layers (Big Accuracy Boost)
# ------------------------------------------------------

for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_data,
    validation_data=val_data,
    steps_per_epoch=steps_train,
    validation_steps=steps_val,
    epochs=10
    )

# ------------------------------------------------------
# Save Model and Label Encoder
# ------------------------------------------------------

model.save("build_outputs/scene_classifier.h5")
np.save("build_outputs/label_encoder.npy", le.classes_)

print("Training complete!")