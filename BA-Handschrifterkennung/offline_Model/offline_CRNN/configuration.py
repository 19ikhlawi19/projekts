# configuration.py

import os
import random
import numpy as np
import torch
from datetime import datetime
import argparse
import logging

# --- (Keep existing imports and parser setup) ---

# ----
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# ----
# Hauptverzeichnis
# Beispiel: "Dataset/offline"
# ----
DATASET_PATH        = os.path.join(BASE_PATH, "Dataset")
DATASET_PATH_OFFLINE = os.path.join(DATASET_PATH, "offline")

# ----
# Archive oder ASCII-Files für IAM
# ----
IAM_TGZ_FOLDER   = os.path.join(BASE_PATH, "data")
IAM_ASCII_FOLDER = os.path.join(BASE_PATH, "data", "ascii")

# ----
# Wort-Basierte Pfade
#   - words_filename_label.csv (CSV mit [filename, label])
#   - words/ (Ordner mit den Bildern)
# ----
# !! WICHTIG: Pfade hier ggf. anpassen !!
# Beispielpfade (bitte durch Ihre tatsächlichen Pfade ersetzen):
WORDS_CSV    = os.path.join(DATASET_PATH_OFFLINE, "words_filename_label.csv") # Beispiel
WORDS_FOLDER = os.path.join(DATASET_PATH_OFFLINE, "words") # Beispiel
# WORDS_CSV    = "/home/im.ikhlawi/test1/BA-V1/Dataset/offline/words_filename_label.csv" # Ihr Beispiel
# WORDS_FOLDER = "/home/im.ikhlawi/test1/BA-V1/Dataset/offline/words" # Ihr Beispiel


# ----
# Zeilen-Basiert (IAM Lines) - Falls benötigt
# ----
IAM_LINES_CSV = os.path.join(BASE_PATH, "data", "iam_lines.csv")
LINES_FOLDER  = os.path.join(DATASET_PATH_OFFLINE, "lines")

# ----
# Optionale Train/Test-Splits für die Wort-Bilder
# ----
TRAIN_LABELS_FILE = os.path.join(DATASET_PATH_OFFLINE, "train_labels_words.csv")
VAL_LABELS_FILE = os.path.join(DATASET_PATH_OFFLINE, "val_labels_words.csv")
TEST_LABELS_FILE  = os.path.join(DATASET_PATH_OFFLINE, "test_labels_words.csv")

# Speicherorte für Modelle, Logs, etc.
MODEL_VERSION   = datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "models", MODEL_VERSION)
LOGS_PATH       = os.path.join(BASE_PATH, "logs") # Logs besser nicht versionieren
CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints", MODEL_VERSION)
RESULTS_PATH    = os.path.join(BASE_PATH, "results", MODEL_VERSION)
METRICS_PATH    = os.path.join(BASE_PATH, "metrics", MODEL_VERSION)

# Neue Einstellungen für Daten-Splits
TRAIN_SPLIT = 0.8  # 80% für Training
VAL_SPLIT = 0.1    # 10% für Validierung
TEST_SPLIT = 0.1   # 10% für Test

# Alte Einstellung für Kompatibilität (wird nicht mehr verwendet)
# VALIDATION_SPLIT = 0.1 # Entfernt

# Metriken zur Berechnung - Aktualisiert für Klarheit und Character-Level
METRICS_TO_COMPUTE = [
    "loss",           # Loss-Werte
    "cer",            # Character Error Rate
    "wer",            # Word Error Rate
    "word_accuracy",  # Exakte Wortgenauigkeit
    "char_f1",        # Character-Level F1-Score
    "char_precision", # Character-Level Precision
    "char_recall",    # Character-Level Recall
    "bleu_4",         # BLEU-4 Score (als repräsentativer BLEU)
]

# Bild-Dimensionen
IMG_HEIGHT = 64
USE_DYNAMIC_WIDTH = True # Belassen oder auf False setzen
MAX_IMG_WIDTH = 512      # Maximale Breite nach Resize, falls USE_DYNAMIC_WIDTH=True
PAD_TO_WIDTH = 512       # Breite, auf die *alle* Bilder gepadded werden (nach Resize)
IMG_CHANNELS = 1

# Trainingskonfiguration
LEARNING_RATE = 0.0001
BATCH_SIZE    = 32
EPOCHS        = 50 # Reduziert für schnellere Testläufe, ggf. wieder erhöhen
OPTIMIZER     = 'adam'   # 'adam' | 'adamw' | 'sgd'
USE_MIXED_PRECISION = False # Kann auf True gesetzt werden, wenn GPU es unterstützt & Performance benötigt wird
SCHEDULER     = 'ReduceLROnPlateau' # 'ReduceLROnPlateau' | 'none'
ENABLE_TRANSFORMER  = False
ENABLE_DISTRIBUTED  = False # Für Multi-GPU Training (komplexer)
USE_DEEPER_CNN      = True
EARLY_STOPPING_PATIENCE = 15

# Zeichenvorrat
CHAR_LIST = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-'.?!,;:+/()[]{}=%&$€ ÜÄÖüäöß")
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHAR_LIST)}
CHAR_TO_IDX['blank'] = len(CHAR_LIST)  # Extra-Index fürs CTC-Blank
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
NUM_CLASSES = len(CHAR_TO_IDX)

MAX_LABEL_LENGTH = 30 # Kann angepasst werden, falls längere Worte vorkommen

# Logging und Gerät
LOGGING_LEVEL = logging.INFO # Geändert zu logging.INFO statt String "INFO"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Zufallseeds setzen für Reproduzierbarkeit
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED) # Für deterministisches Verhalten auf GPU

# Ordner anlegen
PATH_CONFIG = {
    "dataset":           DATASET_PATH,
    "dataset_offline":   DATASET_PATH_OFFLINE,
    "model_save":        MODEL_SAVE_PATH,
    "logs":              LOGS_PATH, # Log-Ordner wird jetzt auch erstellt
    "checkpoints":       CHECKPOINT_PATH,
    "results":           RESULTS_PATH,
    "metrics":           METRICS_PATH
}
for name, path in PATH_CONFIG.items():
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            # print(f"Verzeichnis erstellt: {path}") # Optionales Debugging
    except OSError as e:
        print(f"Fehler beim Erstellen des Verzeichnisses {path}: {e}")

# Logging Konfiguration nur einmal zentral (z.B. in main.py oder model_training.py)
# logging.basicConfig(level=LOGGING_LEVEL, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s') # Entfernen
