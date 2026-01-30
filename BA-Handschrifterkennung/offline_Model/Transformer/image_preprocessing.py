# image_preprocessing.py
import os
import cv2
import numpy as np
import logging
import configuration
import utilities
from skimage import restoration
import tqdm

# Enthält OPTIONALE Bildvorverarbeitungsroutinen (Denoise, Deskew etc.)
# Finale Größenanpassung/Padding geschieht in utilities.load_image!

logger = logging.getLogger(__name__)

# --- Einzelne Preprocessing Funktionen ---

def enhanced_deskew(image_u8):
    """ Begradigt Bild basierend auf minAreaRect (Input/Output: uint8 H,W). """
    if image_u8.ndim != 2 or image_u8.dtype != np.uint8: return image_u8 # Erwarte uint8 grau
    img_inverted = 255 - image_u8
    coords = cv2.findNonZero(img_inverted)
    if coords is None: return image_u8
    try:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45: angle = -(90 + angle)
        else: angle = -angle
        if abs(angle) < 1.0: return image_u8 # Zu kleiner Winkel

        logger.debug(f"Deskewing: Winkel {angle:.2f}°")
        (h, w) = image_u8.shape; center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image_u8, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        return rotated
    except Exception as e: logger.warning(f"Deskewing Fehler: {e}. Nutze Original."); return image_u8

def background_remove_otsu(image_u8):
    """ Binarisierung via OTSU (Input/Output: uint8 H,W). """
    if image_u8.ndim != 2 or image_u8.dtype != np.uint8: return image_u8
    try:
        _, thresh = cv2.threshold(image_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = 255 - thresh # Schwarzer Text auf weißem Grund
        return thresh
    except Exception as e: logger.warning(f"OTSU Fehler: {e}. Nutze Original."); return image_u8

def denoise_image_nlm(image_u8):
    """ Rauschreduktion mit Non-Local Means (Input/Output: uint8 H,W). """
    if image_u8.ndim != 2 or image_u8.dtype != np.uint8: return image_u8
    try:
        img_float = image_u8.astype(np.float32) / 255.0
        denoised_float = restoration.denoise_nl_means(img_float, h=0.05, patch_size=5, patch_distance=7, fast_mode=True, preserve_range=True)
        denoised_uint8 = np.clip(denoised_float * 255, 0, 255).astype(np.uint8)
        return denoised_uint8
    except Exception as e: logger.warning(f"Denoising Fehler: {e}. Nutze Original."); return image_u8

# --- Orchestrator für ein Bild ---
def preprocess_image_pipeline(image_u8_gray):
    """
    Wendet die ausgewählten *optionalen* Vorverarbeitungsschritte an.
    Input/Output: uint8 (H, W)
    """
    img = image_u8_gray.copy() # Arbeite auf Kopie
    # --- Aktivieren / Deaktivieren / Reihenfolge anpassen nach Bedarf ---
    # img = enhanced_deskew(img)
    # img = denoise_image_nlm(img)
    # img = background_remove_otsu(img)
    # --- ---
    return img

# --- Funktion zum Anwenden auf einen Ordner ---
def preprocess_folder(input_folder, output_folder):
    """
    Wendet preprocess_image_pipeline auf alle Bilder im input_folder an und
    speichert sie im output_folder. Behält die Ordnerstruktur bei.
    """
    utilities.ensure_dir(output_folder)
    logger.info(f"Starte optionale Vorverarbeitung für Ordner: {input_folder} -> {output_folder}")
    image_files = []
    for root, _, files in os.walk(input_folder):
         for file in files:
              if file.lower().endswith(('.png','.jpg','.jpeg', '.tif', '.tiff')):
                   image_files.append(os.path.join(root, file))
    if not image_files: logger.warning(f"Keine Bilder in {input_folder}"); return

    processed_count = 0; error_count = 0
    progress_bar = tqdm(image_files, desc="Vorverarbeite Bilder", unit=" Bild")
    for file_path_in in progress_bar:
        relative_path = os.path.relpath(file_path_in, input_folder)
        file_path_out = os.path.join(output_folder, relative_path)
        file_path_out = os.path.splitext(file_path_out)[0] + ".png" # Konsistentes Format
        utilities.ensure_dir(os.path.dirname(file_path_out))

        try:
            original_image = cv2.imread(file_path_in, cv2.IMREAD_GRAYSCALE) # Direkt Graustufen laden
            if original_image is not None:
                processed_image = preprocess_image_pipeline(original_image) # Wende Pipeline an
                if processed_image is not None:
                    save_success = utilities.save_image(processed_image, file_path_out)
                    if save_success: processed_count += 1
                    else: error_count += 1; logger.warning(f"Speichern fehlgeschlagen: {file_path_out}")
                else: error_count += 1; logger.warning(f"Vorverarbeitung fehlgeschlagen (None): {file_path_in}")
            else: error_count += 1; logger.warning(f"Laden fehlgeschlagen: {file_path_in}")
        except Exception as e: error_count += 1; logger.error(f"Fehler Verarbeitung {file_path_in}: {e}", exc_info=False)

    logger.info(f"Vorverarbeitung Ordner '{input_folder}' abgeschlossen.")
    logger.info(f"  Bilder gefunden: {len(image_files)}")
    logger.info(f"  Erfolgreich verarbeitet/gespeichert: {processed_count}")
    logger.info(f"  Fehler/Übersprungen: {error_count}")

def preprocess_all_image_data():
    """ Führt preprocess_folder für den in configuration.py definierten Ordner aus. """
    input_folder = configuration.LINES_FOLDER
    base_input_dir = os.path.dirname(input_folder)
    input_folder_name = os.path.basename(input_folder)

    if input_folder_name.endswith("_preprocessed"):
         output_folder = input_folder
         logger.warning(f"Eingabeordner '{input_folder}' scheint bereits Preprocessing-Ordner zu sein. Ausgabe in denselben Ordner (Überschreiben!).")
    else:
        output_folder_name = input_folder_name + "_preprocessed"
        output_folder = os.path.join(base_input_dir, output_folder_name)

    if not os.path.isdir(input_folder): logger.error(f"Eingabeordner {input_folder} nicht gefunden."); return
    if input_folder == output_folder and not input_folder_name.endswith("_preprocessed"):
        logger.error("Eingabe == Ausgabe! Breche ab, um Überschreiben zu verhindern."); return

    preprocess_folder(input_folder, output_folder)
    logger.info(f"Optionale Bild-Vorverarbeitung abgeschlossen. Ergebnisse in: {output_folder}")
    logger.info("-> Passen Sie ggf. 'configuration.LINES_FOLDER' an, um diesen Ordner zu nutzen.")

if __name__ == "__main__":
    logger.info("image_preprocessing.py wird direkt ausgeführt (Test).")
    preprocess_all_image_data()