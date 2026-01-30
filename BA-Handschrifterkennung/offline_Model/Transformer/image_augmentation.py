# image_augmentation.py
import os
import random
import numpy as np
import torch
import logging
import cv2
from torchvision import transforms
from torchvision.transforms import functional as F_trans # Import von functional
import configuration

logger = logging.getLogger(__name__)

class SafeRandomAffine(transforms.RandomAffine):
    """ Verhindert Fehler bei leeren Bildern. """
    def forward(self, img):
        if isinstance(img, torch.Tensor): img_np = img.cpu().numpy()
        elif isinstance(img, np.ndarray): img_np = img
        else: img_np = np.array(img) # PIL Image
        if img_np.size == 0 or img_np.max() < 1e-6: return img # Prüfe auch auf sehr dunkle Bilder
        return super().forward(img)

class SafeRandomPerspective(transforms.RandomPerspective):
    """ Verhindert Fehler bei leeren Bildern. """
    def forward(self, img):
        if isinstance(img, torch.Tensor): img_np = img.cpu().numpy()
        elif isinstance(img, np.ndarray): img_np = img
        else: img_np = np.array(img)
        if img_np.size == 0 or img_np.max() < 1e-6: return img
        return super().forward(img)

def get_offline_augmentation_transform(use_affine=True, use_perspective=True, use_color=True):
    """
    Erstellt eine torchvision Compose-Transform für Offline-Bilder.
    Input: Tensor (1, H, W), float [0, 1]
    Output: Tensor (1, H, W), float [0, 1]
    """
    transform_list = []
    # Konvertierung zu PIL für geometrische Transformationen
    transform_list.append(transforms.ToPILImage())

    if use_affine:
        transform_list.append(SafeRandomAffine(
            degrees=(-4, 4),          # Etwas mehr Rotation als bei Online
            translate=(0.05, 0.05),   # Etwas mehr Verschiebung
            scale=(0.85, 1.15),     # Etwas mehr Skalierung
            shear=(-5, 5),            # Etwas mehr Scherung
            fill=255 # Fülle mit Weiß (für uint8 PIL)
        ))
    if use_perspective:
        transform_list.append(SafeRandomPerspective(
            distortion_scale=0.3, p=0.4, # Etwas stärkere Perspektive
            fill=255
        ))
    if use_color:
        # Helligkeit/Kontrast Anpassung (relativ robust)
        transform_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.3))

    # Zurück zu Tensor (C, H, W)
    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)

def random_elastic_distortion(image_tensor_chw, alpha=35, sigma=5):
    """
    Wendet elastische Verformung auf einen Tensor an (konvertiert intern).
    Input: Tensor (1, H, W)
    Output: Tensor (1, H, W)
    """
    if image_tensor_chw is None or image_tensor_chw.nelement() == 0: return image_tensor_chw
    try:
        # Tensor -> Numpy (H, W)
        image_np_hw = image_tensor_chw.squeeze(0).cpu().numpy()
        if image_np_hw.size == 0: return image_tensor_chw # Sicherheit
        shape = image_np_hw.shape

        dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1) * alpha, (0,0), sigma)
        dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1) * alpha, (0,0), sigma)
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32); map_y = (y + dy).astype(np.float32)

        # Wende auf Float-Bild an, fülle mit 1.0 (weiß)
        distorted_image = cv2.remap(image_np_hw, map_x, map_y,
                                    interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)

        # Numpy (H, W) -> Tensor (1, H, W)
        distorted_tensor = torch.from_numpy(distorted_image).unsqueeze(0).to(image_tensor_chw.device)
        return distorted_tensor

    except Exception as e: logger.warning(f"Fehler Elastic Distortion: {e}. Nutze Original."); return image_tensor_chw

def augment_offline_image(image_tensor_1hw):
    """
    Haupt-Augmentierungsfunktion. Wendet die definierte Transform-Pipeline an
    und optional Elastic Distortion.
    Input/Output: Tensor (1, H, W), float32 [0, 1]
    """
    if image_tensor_1hw is None or image_tensor_1hw.nelement() == 0:
         logger.warning("Leerer Tensor an augment_offline_image übergeben.")
         return image_tensor_1hw

    # Aktivierungs-Wahrscheinlichkeiten für Augmentierungen
    apply_geometic = random.random() < 0.8  # 80% Chance auf Affine/Perspective/Color
    apply_elastic = random.random() < 0.4   # 40% Chance auf Elastic Distortion

    if not apply_geometic and not apply_elastic:
        return image_tensor_1hw # Keine Augmentierung angewendet

    img_out = image_tensor_1hw

    try:
        if apply_geometic:
            transform = get_offline_augmentation_transform()
            # Transform erwartet PIL, gibt Tensor (1,H,W) zurück
            img_out = transform(img_out)

        if apply_elastic:
            img_out = random_elastic_distortion(img_out, alpha=40, sigma=5)

        # Stelle sicher, dass der Output immer noch das korrekte Format hat
        if img_out.shape != image_tensor_1hw.shape:
            logger.warning(f"Augmentierung änderte Shape von {image_tensor_1hw.shape} zu {img_out.shape}. Versuche Resize.")
            # Versuche, zurück zu resizen/padden? Schwierig, sollte nicht passieren.
            # Einfacher: Original zurückgeben
            return image_tensor_1hw

        return img_out

    except Exception as e:
        logger.exception(f"Fehler bei augment_offline_image(): {e}", exc_info=True)
        return image_tensor_1hw # Gebe Original zurück bei Fehlern