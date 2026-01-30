# model_inference.py
"""
Einzelbild-Inferenz: Lädt Modell und Bild, gibt erkannten Text zurück.
"""
import os
import logging
import torch
import numpy as np
import configuration
import utilities
import ocr_model

logger = logging.getLogger(__name__)


def run_inference(image_path: str, model_path: str) -> str | None:
    """
    Führt Inferenz auf einem einzelnen Bild aus.
    :param image_path: Pfad zum Bild (z.B. Wort-Scan).
    :param model_path: Pfad zur Modell-Datei (.pth).
    :return: Erkanntes Wort als String oder None bei Fehler.
    """
    if not os.path.exists(image_path):
        logger.error(f"Bild nicht gefunden: {image_path}")
        return None
    if not os.path.exists(model_path):
        logger.error(f"Modell nicht gefunden: {model_path}")
        return None

    device = torch.device(configuration.DEVICE)
    image = utilities.load_image(image_path)
    if image is None:
        logger.error(f"Bild konnte nicht geladen werden: {image_path}")
        return None

    try:
        model = ocr_model.build_crnn_model()
        if model is None:
            return None
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        # (1, H, W) -> (1, 1, H, W) Batch-Dimension
        x = torch.from_numpy(image).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)  # (T, 1, C)
        pred_np = logits.cpu().numpy()
        decoded = utilities.decode_predictions(pred_np, configuration.IDX_TO_CHAR, configuration.NUM_CLASSES)
        return decoded[0].strip() if decoded else ""
    except Exception as e:
        logger.error(f"Inferenz fehlgeschlagen: {e}", exc_info=True)
        return None
