from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import tensorflow as tf


def augmentation():
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=45,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=True
    )

    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    return train_gen,val_gen

def generator(paths, labels, batch_size=8, img_size=380):
    while True:
        idx = np.random.permutation(len(paths))
        paths = np.array(paths)[idx]
        labels = labels[idx]

        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            batch_imgs = []
            for p in batch_paths:
                img = tf.keras.utils.load_img(p, target_size=(img_size, img_size))
                img = tf.keras.utils.img_to_array(img)
                img = np.expand_dims(img, 0)
                img = preprocess_input(img)
                batch_imgs.append(img[0])

            yield np.array(batch_imgs), np.array(batch_labels)