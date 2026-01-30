# configuration.py
import os
import random
import numpy as np
import torch
from datetime import datetime
import argparse
import string # Hinzugefügt für string.printable als Fallback

# Minimales Argument Parsing bleibt OK
try:
    parser = argparse.ArgumentParser(description="Konfiguration der Offline-OCR-Pipeline mit Transformer")
    _, _ = parser.parse_known_args()
except Exception:
    pass # Fehler hier ignorieren, Haupt-Parsing in main.py

# ---- Basis Pfad ---
try:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = BASE_PATH
except Exception:
    import sys # Nur für diese Fehlermeldung
    print("[FATAL] Konnte PROJECT_ROOT nicht bestimmen.", file=sys.stderr)
    sys.exit(1)

# ---- Hauptverzeichnis der Datensätze ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_PATH_OFFLINE = os.path.join(DATA_DIR, "offline")
IAM_TGZ_FOLDER = DATA_DIR
LINES_XLSX_PATH = os.path.join(DATASET_PATH_OFFLINE, "iam_lines_gt.xlsx")
LINES_FOLDER = os.path.join(DATASET_PATH_OFFLINE, "lineImages")

# ---- Speicherorte für Splits ----
TRAIN_LABELS_LINES_FILE = os.path.join(DATASET_PATH_OFFLINE, "offline_prepared_train.csv")
VAL_LABELS_LINES_FILE   = os.path.join(DATASET_PATH_OFFLINE, "offline_prepared_val.csv")
TEST_LABELS_LINES_FILE  = os.path.join(DATASET_PATH_OFFLINE, "offline_prepared_test.csv")

# ---- Speicherorte für Modelle, Logs, Ergebnisse etc. ----
MODEL_TYPE_TAG = "offline_transformer"
MODEL_VERSION = f"{MODEL_TYPE_TAG}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
BASE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs")
RUN_OUTPUT_PATH = os.path.join(BASE_OUTPUT_PATH, MODEL_VERSION)

MODEL_SAVE_PATH = os.path.join(RUN_OUTPUT_PATH, "models")
LOGS_PATH = os.path.join(RUN_OUTPUT_PATH, "logs") # Wird nur noch für Pfadkonstruktion gebraucht
CHECKPOINT_PATH = os.path.join(RUN_OUTPUT_PATH, "checkpoints")
RESULTS_PATH = os.path.join(RUN_OUTPUT_PATH, "results")
PLOTS_PATH = os.path.join(RUN_OUTPUT_PATH, "plots")
METRICS_PATH = os.path.join(RUN_OUTPUT_PATH, "metrics")

# ---- Daten-Splits ----
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ---- Bilddimensionen ----
IMG_HEIGHT = 64
PAD_TO_WIDTH = 1024
IMG_CHANNELS = 1

# ---- Text Vokabular ----
TARGET_MAXLEN = 150
# *** EMPFOHLENE ANPASSUNG BASIEREND AUF LOGS ***
# Fügen Sie hier ALLE Zeichen hinzu, die in Ihren Transkripten vorkommen.
# Die vorherigen Logs zeigten, dass '%', '[', ']' fehlten.
CHAR_LIST = list(" !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz[]%{}~`^|") # Fehlende Zeichen hinzugefügt
# Überprüfen Sie dies sorgfältig für den IAM-Datensatz oder Ihren spezifischen Datensatz!

# ---- Trainingskonfiguration ----
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-5
BATCH_SIZE    = 16
ACCUMULATION_STEPS = 2
EPOCHS        = 100
OPTIMIZER     = 'adamw'
USE_MIXED_PRECISION = True
GRAD_CLIP     = 1.0
LABEL_SMOOTHING = 0.1

# ---- LR Scheduler ----
WARMUP_EPOCHS = 10
WARMUP_INIT_LR = 1e-7
FINAL_LR = 1e-6

# ---- Early Stopping ----
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_METRIC = 'val_loss'
MIN_DELTA = 1e-5

# ---- Checkpointing ----
SAVE_CHECKPOINT_INTERVAL = 10

# --- Modell Architektur ---
CNN_OUTPUT_CHANNELS = 512
TRANSFORMER_EMBED_DIM = 512
TRANSFORMER_NUM_HEADS = 8
TRANSFORMER_FFN_DIM = 1024
TRANSFORMER_ENCODER_LAYERS = 4
TRANSFORMER_DECODER_LAYERS = 3
TRANSFORMER_DROPOUT = 0.15

# ---- Sonstiges ----
LOGGING_LEVEL = "INFO"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = min(4, os.cpu_count() if os.cpu_count() else 1)
RANDOM_SEED = 42