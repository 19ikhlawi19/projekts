import os
import cv2
import numpy as np
import logging
import configuration
import utilities
from skimage import restoration

# -----------------------------------------------------------------------------
# image_preprocessing.py
# -----------------------------------------------------------------------------
# Enthält Funktionen zur Bildvorverarbeitung (Deskew, Hintergrundentfernung, etc.)
# -----------------------------------------------------------------------------

logging.basicConfig(level=configuration.LOGGING_LEVEL, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def enhanced_deskew(image):
    """
    Versucht, das Bild basierend auf minimalem Begrenzungsrechteck zu begradigen.
    Für Wortbilder kann dies nützlich sein, wenn Words schief eingescant wurden.
    """
    coords = np.column_stack(np.where(image > 0))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return image

def background_remove(image):
    """
    Entfernt Hintergrund via OTSU Threshold und Morphologie.
    Für kontrastreiche Scans oft ausreichend.
    """
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cleaned

def denoise_image(image):
    """
    Rauschreduktion mittels Non-Local Means (skimage).
    """
    if image.dtype != np.float32:
        tmp = image.astype(np.float32) / 255.0
    else:
        tmp = image
    denoised = restoration.denoise_nl_means(tmp, patch_size=5, patch_distance=7, fast_mode=True, h=0.1)
    return (denoised * 255).astype(np.uint8)

def preprocess_image(image):
    """
    Gesamtpipeline: Deskew, Background-Remove, Denoise, Resize, Normalisierung.
    """
    try:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Deskew
        image = enhanced_deskew(image)
        # Hintergrund entfernen
        image = background_remove(image)
        # Denoising
        image = denoise_image(image)

        # Resize (dynamische Breite oder fix) => Für Wortbilder sehr sinnvoll.
        if configuration.USE_DYNAMIC_WIDTH:
            target_h = configuration.IMG_HEIGHT
            ratio = image.shape[1] / float(image.shape[0])
            new_w = int(target_h * ratio)
            new_w = min(new_w, configuration.MAX_IMG_WIDTH)
            image = cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            image = cv2.resize(image, (configuration.IMG_WIDTH, configuration.IMG_HEIGHT), interpolation=cv2.INTER_AREA)

        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=-1)  # => (H,W,1)
        return image

    except Exception as e:
        logger.error(f"Fehler bei der Vorverarbeitung: {e}")
        return None

def preprocess_folder(input_folder, output_folder):
    """
    Wendet preprocess_image() auf alle Bilder im input_folder an und
    speichert sie im output_folder.
    """
    try:
        utilities.create_directory(output_folder)
        files = os.listdir(input_folder)
        for file in files:
            if file.lower().endswith(('.png','.jpg','.jpeg')):
                file_path = os.path.join(input_folder, file)
                image = cv2.imread(file_path)
                if image is not None:
                    processed = preprocess_image(image)
                    if processed is not None:
                        save_path = os.path.join(output_folder, file)
                        utilities.save_image(processed, save_path)
        logger.info(f"Vorverarbeitung für Folder '{input_folder}' abgeschlossen.")
    except Exception as e:
        logger.error(f"Fehler beim Vorverarbeiten des Ordners {input_folder}: {e}")

def preprocess_all_datasets():
  
    try:
        logger.info("Preprocessing: Bitte Ordnerpfade in preprocess_all_datasets() anpassen.")
        # z.B.:
        # preprocess_folder("<TRAIN_WORDS_FOLDER>", "<OUTPUT_TRAIN_WORDS_FOLDER>")
        # preprocess_folder("<VAL_WORDS_FOLDER>",   "<OUTPUT_VAL_WORDS_FOLDER>")
        # ...
    except Exception as e:
        logger.error(f"Fehler bei preprocess_all_datasets(): {e}")