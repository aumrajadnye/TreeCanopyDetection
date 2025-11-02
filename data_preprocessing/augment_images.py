import cv2
import os
import random
from pathlib import Path
import numpy as np

def augment_image(image_path, config):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: could not read image {image_path}")
            return None
        
        """Applies augmentations as defined in config.yaml"""
        if not config.get("enable", False):
            return image

        aug_img = image.copy()

        # Gamma correction
        if config.get("gamma", {}):
            gamma_min, gamma_max = config["gamma"]["range"]
            gamma = random.uniform(gamma_min, gamma_max)
            inv_gamma = 1.0 / max(gamma, 0.001)
            table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
            aug_img = cv2.LUT(aug_img, table)

        # Contrast adjustment
        if config.get("contrast", {}):
            cmin, cmax = config["contrast"]["range"]
            contrast = random.uniform(cmin, cmax)
            mean = np.mean(aug_img)
            aug_img = np.clip((aug_img - mean) * contrast + mean, 0, 255).astype(np.uint8)
        
        # Zoom augmentation
        if config.get("zoom", {}):
            zmin, zmax = config["zoom"]["range"]
            zoom_factor = 1 + (random.uniform(zmin, zmax) - 5) / 10  # normalize around 1
            h, w = aug_img.shape[:2]
            new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
            aug_img = cv2.resize(aug_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # Crop or pad to original size
            if zoom_factor > 1:
                x_start = (new_w - w) // 2
                y_start = (new_h - h) // 2
                aug_img = aug_img[y_start:y_start + h, x_start:x_start + w]
            else:
                pad_x = (w - new_w) // 2
                pad_y = (h - new_h) // 2
                aug_img = cv2.copyMakeBorder(aug_img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT)

        # Rotation augmentation
        if config.get("rotation", {}):
            rmin, rmax = config["rotation"]["range"]
            angle = random.uniform(rmin, rmax)
            h, w = aug_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Flipping
        if config.get("flip", {}):
            flip_cfg = config["flip"]
            h_prob = random.uniform(*flip_cfg.get("horizontal", [0.0, 1.0]))
            v_prob = random.uniform(*flip_cfg.get("vertical", [0.0, 1.0]))
            if h_prob > 0.5:
                aug_img = cv2.flip(aug_img, 1)
            if v_prob > 0.5:
                aug_img = cv2.flip(aug_img, 0)

        # Hue/Saturation adjustments
        if config.get("hue_saturation", {}):
            h_range = config["hue_saturation"]["h_range"]
            s_range = config["hue_saturation"]["s_range"]
            v_range = config["hue_saturation"]["v_range"]
            h_shift = random.uniform(*h_range)
            s_shift = random.uniform(*s_range)
            v_shift = random.uniform(*v_range)

            hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + h_shift) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] + s_shift, 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] + v_shift, 0, 255)
            aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return aug_img

def augment_dataset(input_dir, output_dir, aug_config, logger):
    """Apply augmentations to all images in input_dir and save to output_dir."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_path in input_path.glob("*.*"):
        augmented = augment_image(img_path, aug_config)
        if augmented is None:
            continue
        save_path = output_path / img_path.name
        cv2.imwrite(str(save_path), augmented)

    logger.info(f"Augmentation complete. Augmented images saved to {output_dir}")
    return output_dir