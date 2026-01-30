import os
import random
import numpy as np
import torch
import logging
import cv2
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
import configuration

# -----------------------------------------------------------------------------
# image_augmentation.py
# -----------------------------------------------------------------------------
# Für On-the-fly Augmentierung während des DataLoader-Ladens (z.B. Rotation, Perspektive).
# -----------------------------------------------------------------------------

logging.basicConfig(level=configuration.LOGGING_LEVEL, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def get_augmentation_transform():
    """
    Erstellt eine Compose-Transform für zufällige Affine-Transformation,
    Perspektivverzerrung und anschließendes ToTensor().
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=0.1
        ),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.ToTensor()
    ])

def random_thickness_variation(image_tensor, severity=0.05):
    """
    Simuliert Variation in der Strichstärke (Erodieren oder Dilatieren).
    severity => Intensität (z.B. 0.05)
    """
    img_np = image_tensor.squeeze(0).numpy()
    kernel_size = np.random.randint(1, 3)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if random.random() < 0.5:
        eroded = cv2.erode((img_np*255).astype(np.uint8), kernel, iterations=1)
        new_img = (1-severity)*img_np*255 + severity*eroded
        new_img /= 255.0
    else:
        dilated = cv2.dilate((img_np*255).astype(np.uint8), kernel, iterations=1)
        new_img = (1-severity)*img_np*255 + severity*dilated
        new_img /= 255.0
    return new_img[np.newaxis, ...]

def augment_image(image):
    """
    Wendet die definierte Transform-Pipeline + random_thickness_variation an.
    """
    transform = get_augmentation_transform()
    try:
        if image.dtype != np.float32:
            image = image.astype(np.float32)/255.0
        if image.ndim == 3 and image.shape[0] == 1:
            # => (1,H,W), ToPILImage erwartet (H,W,1) => konvertieren
            image = np.squeeze(image, axis=0)  # => (H,W)
        # Tensor
        image_t = to_tensor(image)
        image_t = transform(image_t)
        if random.random() < 0.5:
            image_t = torch.from_numpy(random_thickness_variation(image_t, 0.05))
        return image_t.numpy()
    except Exception as e:
        logger.error(f"Fehler bei augment_image(): {e}")
        return None
