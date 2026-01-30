# utilities.py
import os
import random
import string
import logging
from datetime import datetime
from collections import Counter
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import sys # For error exit
import cv2 # Hinzugefügt für Bildladen
import csv # <<<----- HIER HINZUGEFÜGT
import Levenshtein # Import für Metriken
# BLEU Score ist für HTR oft weniger aussagekräftig als CER/WER, kann aber drin bleiben
try: from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError: logger.warning("nltk not installed, BLEU score calculation disabled. Run: pip install nltk"); sentence_bleu = None


import configuration # Importiere zentrale Konfiguration

logger = logging.getLogger(__name__)

# --- General Utilities ---
# get_timestamp, ensure_dir, set_seeds, get_device, get_project_root bleiben unverändert
def get_timestamp():
    """Generates a timestamp string in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def ensure_dir(directory):
    """Creates the directory if it doesn't exist. Logs creation."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {e}", exc_info=True)
            raise

def set_seeds(seed=configuration.RANDOM_SEED): # Verwende Seed aus config
    """Sets random seeds for Python, NumPy, and PyTorch for reproducibility."""
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Diese Einstellungen können das Training verlangsamen, sind aber gut für die Fehlersuche
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
            # logger.info("CuDNN set to deterministic (may impact performance).")
        logger.info(f"Global random seeds set to {seed}")
    except Exception as e:
        logger.error(f"Error setting random seeds: {e}", exc_info=True)

def get_device(force_gpu=False):
    """Gets the appropriate PyTorch device (GPU if available/forced, else CPU)."""
    effective_force_gpu = force_gpu or (configuration.DEVICE.lower() == 'cuda') # Berücksichtige config
    device_name = configuration.DEVICE.lower() # Default aus config

    if effective_force_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(device.index)
            logger.info(f"GPU forced/configured and found: {gpu_name}")
        else:
            logger.error("GPU forced/configured, but no CUDA-compatible GPU found! Aborting.")
            sys.exit(1) # Exit if forced GPU is not available
    elif torch.cuda.is_available() and device_name != 'cpu':
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(device.index)
        logger.info(f"GPU available and selected: {gpu_name}")
    else:
        device = torch.device("cpu")
        logger.info("No GPU found or CPU configured, using CPU.")
    return device

def get_project_root():
    """Finds the project root directory (assuming configuration.py is at the root)."""
    try:
        project_root = configuration.PROJECT_ROOT
        if not os.path.exists(os.path.join(project_root, 'configuration.py')):
             project_root_from_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
             if os.path.exists(os.path.join(project_root_from_file, 'configuration.py')):
                   logger.warning(f"Project root derived from utils.py location: {project_root_from_file} instead of configuration.PROJECT_ROOT")
                   return project_root_from_file
        return project_root
    except Exception as e:
         logger.error(f"Error determining project root: {e}", exc_info=True)
         return None


# --- Text Vectorization ---
# VectorizeChar Klasse bleibt unverändert wie in der vorherigen Antwort
class VectorizeChar:
    """
    Maps characters to integer indices and back, handling special tokens.
    Used for the Transformer decoder target sequences.
    Special Tokens: <PAD>=0, <UNK>=1, <START>=2, <END>=3
    """
    def __init__(self, max_len=configuration.TARGET_MAXLEN):
        self.special_tokens = ["<PAD>", "<UNK>", "<START>", "<END>"] # PAD must be 0
        if self.special_tokens[0] != "<PAD>":
            raise ValueError("<PAD> token must be at index 0")

        if hasattr(configuration, 'CHAR_LIST') and isinstance(configuration.CHAR_LIST, list) and configuration.CHAR_LIST:
            base_chars = list(configuration.CHAR_LIST)
            logger.info(f"Verwende CHAR_LIST aus configuration.py ({len(base_chars)} Zeichen).")
        else:
            logger.warning("configuration.CHAR_LIST nicht definiert, ungültig oder leer. Verwende Standard ASCII (letters+digits+punct+space).")
            base_chars = list(string.ascii_letters + string.digits + string.punctuation + ' ')

        vocab_set = set(base_chars)
        for token in self.special_tokens:
            if token in vocab_set:
                logger.warning(f"Special token '{token}' was present in CHAR_LIST, removing it.")
                vocab_set.remove(token)

        self.vocab = self.special_tokens + sorted(list(vocab_set))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}

        self.pad_token_id = self.char_to_idx["<PAD>"]
        self.unk_token_id = self.char_to_idx["<UNK>"]
        self.start_token_id = self.char_to_idx["<START>"]
        self.end_token_id = self.char_to_idx["<END>"]
        self.max_len = max_len

        logger.info(f"Vectorizer initialized: Vocab Size = {len(self.vocab)}, Max Target Length = {self.max_len}")
        logger.debug(f"Vocabulary: {self.vocab}")
        if self.pad_token_id != 0:
             logger.critical(f"PAD token ID is not 0 ({self.pad_token_id})! This will break padding/loss calculation.")
             raise ValueError("PAD token index MUST be 0")

    def __call__(self, text: str) -> list[int]:
        if not isinstance(text, str):
             try: text = str(text)
             except: logger.error(f"Could not convert input to string: {text}"); return []

        indices = [self.start_token_id]
        unknown_chars_found = []
        for char in text:
            idx = self.char_to_idx.get(char, self.unk_token_id)
            indices.append(idx)
            if idx == self.unk_token_id:
                 unknown_chars_found.append(char)

        indices.append(self.end_token_id)

        if unknown_chars_found:
            logger.debug(f"Unknown characters found in text '{text}': {set(unknown_chars_found)}")

        if len(indices) > self.max_len:
            original_len = len(indices)
            indices = indices[:self.max_len - 1] + [self.end_token_id]
            logger.debug(f"Input text ('{text[:30]}...') truncated from {original_len} to {len(indices)} tokens (max_len={self.max_len}).")

        if len(indices) <= 2:
            logger.warning(f"Text '{text}' resulted in empty or invalid token sequence (len={len(indices)}).")
            return []
        return indices

    def decode(self, token_ids: list[int] or torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor): token_ids = token_ids.cpu().tolist()
        special_ids_to_ignore = {self.pad_token_id, self.start_token_id, self.end_token_id, self.unk_token_id}
        chars = []
        for idx in token_ids:
            if idx == self.end_token_id: break
            if idx not in special_ids_to_ignore:
                chars.append(self.idx_to_char.get(idx, "?"))
        return "".join(chars)

    def get_vocabulary(self) -> list[str]: return self.vocab
    def get_vocab_size(self) -> int: return len(self.vocab)
    def get_idx_to_char_map(self) -> dict[int, str]: return self.idx_to_char
    def get_char_to_idx_map(self) -> dict[str, int]: return self.char_to_idx

# --- Bild Laden & Speichern ---
# load_image und save_image bleiben unverändert wie in der vorherigen Antwort
def load_image(absolute_image_path):
    """
    Lädt ein Bild, konvertiert zu Graustufen, passt Höhe an configuration.IMG_HEIGHT an,
    padded auf configuration.PAD_TO_WIDTH, normalisiert auf [0,1] und gibt es als
    float32 Tensor (1, H, W) zurück.
    """
    if not absolute_image_path or not isinstance(absolute_image_path, str):
        logger.error(f"Ungültiger Pfad an load_image übergeben: {absolute_image_path}")
        return None

    absolute_image_path = absolute_image_path.replace("\\", "/")
    if not os.path.exists(absolute_image_path):
        logger.error(f"Bilddatei existiert nicht: {absolute_image_path}")
        return None

    try:
        img = cv2.imread(absolute_image_path, cv2.IMREAD_UNCHANGED)
        if img is None: logger.error(f"Bild nicht geladen (cv2): {absolute_image_path}"); return None

        if img.ndim == 3:
            if img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            elif img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else: logger.error(f"Unerwartete Kanäle ({img.shape[2]}) in {absolute_image_path}"); return None
        elif img.ndim != 2: logger.error(f"Unerwartete Bilddimensionen ({img.ndim}) in {absolute_image_path}"); return None

        h, w = img.shape
        if h == 0 or w == 0: logger.error(f"Ungültige Dimensionen (h={h}, w={w}): {absolute_image_path}"); return None

        target_h = configuration.IMG_HEIGHT
        target_w = configuration.PAD_TO_WIDTH

        if h == target_h:
            resized_h_img = img
            new_w = w
        else:
            scale_ratio = target_h / float(h)
            new_w = int(round(w * scale_ratio))
            new_w = max(1, new_w)
            resized_h_img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA if scale_ratio < 1 else cv2.INTER_LINEAR)

        current_h, current_w = resized_h_img.shape

        if current_w == target_w:
            final_img = resized_h_img
        elif current_w < target_w:
            pad_width = target_w - current_w
            final_img = np.pad(resized_h_img, ((0, 0), (0, pad_width)), mode='constant', constant_values=255)
        else:
            logger.warning(f"Bild '{os.path.basename(absolute_image_path)}' ({current_w}px breit nach Skalierung) ist breiter als PAD_TO_WIDTH ({target_w}). Schneide rechts ab.")
            final_img = resized_h_img[:, :target_w]

        img_float = final_img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_float).unsqueeze(0)

        if img_tensor.shape != (1, target_h, target_w):
             logger.error(f"Unerwartete finale Tensor-Dimensionen {img_tensor.shape} für {absolute_image_path}. Erwartet: (1, {target_h}, {target_w})")
             return None

        return img_tensor

    except cv2.error as cv_err: logger.error(f"OpenCV Fehler bei {absolute_image_path}: {cv_err}", exc_info=False); return None
    except Exception as e: logger.exception(f"Fehler beim Laden/Verarbeiten von Bild {absolute_image_path}: {e}", exc_info=True); return None

def save_image(image_data, save_path):
    """ Speichert Numpy/Tensor Bild als Datei. """
    if image_data is None: logger.error(f"Kann kein None-Bild speichern: {save_path}"); return False
    ensure_dir(os.path.dirname(save_path))
    try:
        if isinstance(image_data, torch.Tensor): image_data = image_data.detach().cpu().numpy()
        if image_data.ndim == 3 and image_data.shape[0] == 1: img_to_save = np.squeeze(image_data, axis=0)
        elif image_data.ndim == 2: img_to_save = image_data
        else: logger.error(f"Unerwartetes Format zum Speichern für {save_path}: {image_data.shape}"); return False

        if img_to_save.dtype == np.float32 or img_to_save.dtype == np.float64:
            if img_to_save.min() >= 0 and img_to_save.max() <= 1.0 + 1e-6:
                 img_to_save = (img_to_save * 255.0).clip(0, 255).astype(np.uint8)
            else:
                 logger.warning(f"Float Bild {save_path} außerhalb [0,1]. Clamping auf [0, 255]. Range: [{img_to_save.min():.2f}, {img_to_save.max():.2f}]")
                 img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)
        elif img_to_save.dtype != np.uint8:
             img_to_save = img_to_save.astype(np.uint8)

        success = cv2.imwrite(save_path, img_to_save)
        if not success: logger.error(f"cv2.imwrite fehlgeschlagen für: {save_path}"); return False
        else: return True
    except Exception as e: logger.exception(f"Fehler beim Speichern von Bild {save_path}: {e}", exc_info=True); return False

# --- Dekodierung ---
# decode_prediction bleibt unverändert wie in der vorherigen Antwort
def decode_prediction(token_ids, idx_to_char_map, special_token_ids_tuple):
    """
    Decodes a sequence of token IDs from the Transformer decoder into text.
    Uses the Vectorizer's decode method logic implicitly (stop at end, ignore specials).
    Fallback provided in case Vectorizer isn't easily available.
    """
    start_token_id, end_token_id, pad_token_id, unk_token_id = special_token_ids_tuple
    end_pos = -1
    if isinstance(token_ids, torch.Tensor): token_ids = token_ids.cpu().tolist()

    for i, idx in enumerate(token_ids):
        if idx == end_token_id:
            end_pos = i
            break

    text = ""
    limit = end_pos if end_pos != -1 else len(token_ids)
    special_ids_set = {start_token_id, end_token_id, pad_token_id, unk_token_id}

    for idx in token_ids[1:limit]: # Starte nach dem START-Token (Index 0)
        if idx not in special_ids_set:
            text += idx_to_char_map.get(idx, "?") # Fragezeichen für unbekannte *innerhalb* der Sequenz
    return text

# --- CSV / Label Handling ---
# clean_relative_path bleibt unverändert
def clean_relative_path(raw_path):
    """ Bereinigt relative Pfade (ersetzt \\, entfernt Präfixe, führende /). """
    if not isinstance(raw_path, str): logger.warning(f"Ungültiger Typ für clean_relative_path: {type(raw_path)}"); return ""
    cleaned = raw_path.strip().replace("\\", "/")
    prefixes_to_remove = ["data/lineImages/", "Dataset/offline/lineImages/", "lineImages/"] # Beispiele anpassen
    original = cleaned
    changed = False
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix): cleaned = cleaned[len(prefix):]; changed = True
    if cleaned.startswith('/'): cleaned = cleaned.lstrip('/'); changed = True
    if changed: logger.debug(f"Pfad bereinigt: '{original}' -> '{cleaned}'")
    return cleaned

# ** KORREKTUR & VERBESSERUNG für load_labels_from_csv **
def load_labels_from_csv(labels_csv_path, is_split_file=False):
    """
    Liest CSV (file_path, transcript), gibt dict {rel_path: label} zurück.
    Wenn is_split_file=True, wird ein festes Format erwartet (Komma-sep, Header).
    """
    labels = {}
    if not os.path.exists(labels_csv_path):
        logger.error(f"CSV nicht gefunden: {labels_csv_path}")
        return labels

    line_num = 0
    skipped_count = 0
    expected_header = ['file_path', 'transcript']
    path_idx, label_idx = 0, 1 # Feste Indizes für Split-Dateien

    try:
        with open(labels_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Wenn es eine Split-Datei ist (von uns erstellt), erzwinge Format
            if is_split_file:
                reader = csv.reader(f, delimiter=',')
                logger.debug(f"Lese Split-Datei '{os.path.basename(labels_csv_path)}' mit Komma-Trenner.")
                try:
                    header = next(reader)
                    line_num = 1
                    # Validiere Header (strippe whitespace, ignoriere Groß/Kleinschreibung)
                    header_cleaned = [h.strip().lower() for h in header]
                    expected_header_cleaned = [h.strip().lower() for h in expected_header]
                    if header_cleaned != expected_header_cleaned:
                        logger.critical(f"Unerwarteter Header in Split-Datei '{labels_csv_path}'! Erwartet: {expected_header}, Gefunden: {header}. Daten können nicht korrekt gelesen werden!")
                        return {} # Leeres Dict bei falschem Header zurückgeben
                    logger.info(f"Split-Datei Header validiert: {header}")
                except StopIteration:
                    logger.error(f"Split-Datei '{labels_csv_path}' ist leer oder hat keinen Header.")
                    return {}
            else:
                # Behalte die flexible Logik für andere CSVs (z.B. Excel-Konvertierung) bei
                sniffer = csv.Sniffer()
                try:
                    sample = f.read(2048)
                    dialect = sniffer.sniff(sample)
                    f.seek(0)
                    has_header = sniffer.has_header(sample)
                    logger.info(f"CSV Dialekt erkannt: Delimiter='{dialect.delimiter}', Header={has_header}")
                    reader = csv.reader(f, dialect)
                    if has_header:
                        header = next(reader); line_num = 1
                    else:
                         header = ['path_col_0', 'label_col_1'] # Fallback wenn kein Header
                         f.seek(0) # Zurück zum Anfang für Datenlesen
                except csv.Error:
                    logger.warning(f"Konnte CSV-Dialekt für {labels_csv_path} nicht automatisch erkennen, versuche Komma als Delimiter.")
                    f.seek(0); first_line = f.readline().strip(); f.seek(0)
                    if 'file_path' in first_line.lower() or 'transcript' in first_line.lower():
                         has_header = True; reader = csv.reader(f, delimiter=','); header = next(reader); line_num = 1
                    else:
                         has_header = False; reader = csv.reader(f, delimiter=','); header = ['path_col_0', 'label_col_1']
                # Flexible Spaltenfindung nur wenn nicht is_split_file
                path_col_name, label_col_name = None, None
                possible_path_cols = ['file_path', 'filename', 'path', 'image', 'path_col_0']
                possible_label_cols = ['transcript', 'label', 'text', 'ground_truth', 'label_col_1']
                header_lower = [h.strip().lower() for h in header]
                temp_path_idx, temp_label_idx = -1, -1
                for idx, col_name_lower in enumerate(header_lower):
                    if temp_path_idx == -1 and col_name_lower in possible_path_cols: temp_path_idx = idx; path_col_name = header[idx]
                    if temp_label_idx == -1 and col_name_lower in possible_label_cols: temp_label_idx = idx; label_col_name = header[idx]
                if temp_path_idx == -1 or temp_label_idx == -1:
                     logger.error(f"Benötigte Spalten (Pfad/Label) nicht in CSV {labels_csv_path}: {header}"); return {}
                path_idx, label_idx = temp_path_idx, temp_label_idx # Verwende gefundene Indizes
                logger.info(f"Verwende Spalten: Path='{path_col_name}' (Idx={path_idx}), Label='{label_col_name}' (Idx={label_idx}).")


            # Lese Datenzeilen
            for row in reader:
                line_num += 1
                if not row or len(row) <= max(path_idx, label_idx): # Prüfe auf leere Zeilen oder zu wenige Spalten
                    if skipped_count < 10 or skipped_count % 100 == 0:
                        logger.debug(f"Zeile {line_num} übersprungen: Ungültiges Format oder zu wenige Spalten ({len(row)}). Inhalt: '{row}'. Skip #{skipped_count+1}")
                    skipped_count += 1
                    continue

                try:
                    raw_path = row[path_idx].strip()
                    label_str = row[label_idx].strip()
                except IndexError: # Sollte durch obige Längenprüfung abgedeckt sein, aber sicher ist sicher
                    logger.warning(f"Zeile {line_num}: IndexError bei Zugriff auf Indizes {path_idx}/{label_idx}. Row: {row}. Übersprungen.")
                    skipped_count += 1
                    continue

                relative_path = clean_relative_path(raw_path)

                if relative_path and label_str: # Nur gültige Paare verwenden
                    labels[relative_path] = label_str
                else:
                    if skipped_count < 10 or skipped_count % 100 == 0:
                         logger.debug(f"Zeile {line_num} übersprungen: Pfad ('{relative_path}') oder Label ('{label_str}') leer. Skip #{skipped_count+1}")
                    skipped_count += 1

        logger.info(f"Labels aus CSV '{os.path.basename(labels_csv_path)}' ({'Split-Format' if is_split_file else 'Generisch'}): {len(labels)} gültig (Übersprungen: {skipped_count}).")
        return labels

    except Exception as e:
        logger.exception(f"Kritischer Fehler beim Lesen von CSV {labels_csv_path} (Zeile ~{line_num}): {e}", exc_info=True)
        return {} # Leeres Dict bei kritischem Fehler

# ** KORREKTUR für save_labels_as_csv **
def save_labels_as_csv(relative_paths, string_labels, output_csv_file):
    """ Speichert Listen als CSV (file_path, transcript) mit Komma als Trennzeichen. """
    if len(relative_paths) != len(string_labels):
        logger.error(f"Ungleiche Länge Path ({len(relative_paths)}) / Label ({len(string_labels)}) für {output_csv_file}")
        return False
    ensure_dir(os.path.dirname(output_csv_file))
    try:
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as f:
            # Explizit Komma als Trennzeichen und Standard-Quoting verwenden
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['file_path', 'transcript']) # Schreibe Header
            for path, label in zip(relative_paths, string_labels):
                writer.writerow([path, label]) # Schreibe Datenzeile
        logger.info(f"Labels ({len(relative_paths)}) gespeichert als CSV (Komma-getrennt): {output_csv_file}")
        return True
    except Exception as e:
        logger.exception(f"Fehler Speichern CSV {output_csv_file}: {e}", exc_info=True)
        return False

# ** KORREKTUR für load_split_data **
def load_split_data(split_csv_file, vectorizer):
    """
    Lädt Daten aus Split-CSV (erwartet Komma-getrennt mit Header 'file_path', 'transcript')
    und kodiert Labels mit VectorizeChar.
    Gibt zurück: (liste_bereinigter_relativer_pfade, liste_kodierter_labels)
    """
    if not os.path.exists(split_csv_file):
        logger.error(f"Split-CSV nicht gefunden: {split_csv_file}")
        return [], []

    relative_paths = []
    encoded_labels_list = []
    # Rufe load_labels_from_csv mit is_split_file=True auf
    labels_dict = load_labels_from_csv(split_csv_file, is_split_file=True)

    if not labels_dict:
        logger.error(f"Keine gültigen Daten aus Split-CSV {split_csv_file} geladen oder Formatfehler.")
        return [], []

    skipped_encoding_errors = 0
    for rel_path, label_str in labels_dict.items():
        try:
            encoded = vectorizer(label_str) # Kodierung (inkl. Start/End/UNK/Truncate)
            if not encoded or len(encoded) <= 2: # Prüfe auf gültige Kodierung
                 logger.warning(f"Ungültige/leere Kodierung für Label '{label_str}' (Pfad: {rel_path}). Übersprungen.")
                 skipped_encoding_errors += 1
                 continue
            relative_paths.append(rel_path)
            encoded_labels_list.append(encoded)
        except Exception as enc_e:
             logger.error(f"Fehler beim Kodieren von Label '{label_str}' für Pfad '{rel_path}': {enc_e}. Übersprungen.", exc_info=False)
             skipped_encoding_errors += 1

    logger.info(f"Split '{os.path.basename(split_csv_file)}': {len(relative_paths)} Samples erfolgreich geladen und kodiert. (Kodierfehler/Leer: {skipped_encoding_errors})")
    if len(relative_paths) == 0 and len(labels_dict) > 0:
         logger.error(f"Alle {len(labels_dict)} Samples aus {split_csv_file} konnten nicht kodiert werden!")
    return relative_paths, encoded_labels_list

# save_split_data bleibt wie in der vorherigen Antwort, ruft das korrigierte save_labels_as_csv auf
def save_split_data(relative_paths, encoded_labels_list, split_name, vectorizer):
    """ Speichert Splits als CSV (dekodiert Labels zuerst mit vectorizer.decode). """
    if split_name == 'train': output_file = configuration.TRAIN_LABELS_LINES_FILE
    elif split_name == 'val': output_file = configuration.VAL_LABELS_LINES_FILE
    elif split_name == 'test': output_file = configuration.TEST_LABELS_LINES_FILE
    else: logger.error(f"Unbekannter Split-Name: {split_name}"); return False

    string_labels = []
    for encoded in encoded_labels_list:
         label_str = vectorizer.decode(encoded)
         string_labels.append(label_str)

    # Ruft das korrigierte save_labels_as_csv auf
    return save_labels_as_csv(relative_paths, string_labels, output_file)

# --- Evaluation Metrics ---
# levenshtein, wer, cer, compute_precision_recall_f1, compute_f1, compute_precision, compute_recall, compute_bleu, compute_all_metrics bleiben unverändert wie in der vorherigen Antwort

def levenshtein(s1, s2):
    if not isinstance(s1, (str, list)): s1 = str(s1)
    if not isinstance(s2, (str, list)): s2 = str(s2)
    len1, len2 = len(s1), len(s2)
    if len1 == 0: return len2
    if len2 == 0: return len1
    if len1 > len2: s1, s2, len1, len2 = s2, s1, len2, len1
    prev_row = list(range(len1 + 1))
    for i in range(1, len2 + 1):
        curr_row = [i] + [0] * len1
        for j in range(1, len1 + 1):
            cost = 0 if s1[j-1] == s2[i-1] else 1
            curr_row[j] = min(curr_row[j - 1] + 1, prev_row[j] + 1, prev_row[j - 1] + cost)
        prev_row = curr_row
    return prev_row[len1]

def wer(ground_truth: str, prediction: str) -> float:
    if not isinstance(ground_truth, str): ground_truth = str(ground_truth)
    if not isinstance(prediction, str): prediction = str(prediction)
    gt_words = ground_truth.strip().split()
    pred_words = prediction.strip().split()
    if not gt_words: return 0.0 if not pred_words else 1.0
    dist = levenshtein(gt_words, pred_words)
    denom = len(gt_words)
    return dist / denom

def cer(ground_truth: str, prediction: str) -> float:
    if not isinstance(ground_truth, str): ground_truth = str(ground_truth)
    if not isinstance(prediction, str): prediction = str(prediction)
    if not ground_truth: return 0.0 if not prediction else 1.0
    dist = levenshtein(ground_truth, prediction)
    denom = len(ground_truth)
    return dist / denom

def compute_precision_recall_f1(target: str, prediction: str, level='char'):
    if not isinstance(target, str): target = str(target)
    if not isinstance(prediction, str): prediction = str(prediction)
    if level == 'word':
        target_tokens = target.strip().split()
        prediction_tokens = prediction.strip().split()
    elif level == 'char':
        target_tokens = list(target)
        prediction_tokens = list(prediction)
    else:
        logger.error(f"Ungültiges Level '{level}' für Precision/Recall/F1.")
        return 0.0, 0.0, 0.0
    if not target_tokens and not prediction_tokens: return 1.0, 1.0, 1.0
    if not target_tokens: return 0.0, 1.0, 0.0
    if not prediction_tokens: return 1.0, 0.0, 0.0
    target_counts = Counter(target_tokens)
    prediction_counts = Counter(prediction_tokens)
    common_counts = target_counts & prediction_counts
    num_common = sum(common_counts.values())
    precision = num_common / len(prediction_tokens)
    recall = num_common / len(target_tokens)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1_score

def compute_f1(target: str, prediction: str, level='char') -> float:
    _, _, f1 = compute_precision_recall_f1(target, prediction, level); return f1
def compute_precision(target: str, prediction: str, level='char') -> float:
    prec, _, _ = compute_precision_recall_f1(target, prediction, level); return prec
def compute_recall(target: str, prediction: str, level='char') -> float:
    _, rec, _ = compute_precision_recall_f1(target, prediction, level); return rec

def compute_bleu(target: str, prediction: str) -> float:
    if sentence_bleu is None: return 0.0
    if not isinstance(target, str): target = str(target)
    if not isinstance(prediction, str): prediction = str(prediction)
    if not target and not prediction: return 1.0
    if not target or not prediction: return 0.0
    target_list = [list(target)]
    prediction_list = list(prediction)
    try:
        chencherry = SmoothingFunction().method4
        return sentence_bleu(target_list, prediction_list, smoothing_function=chencherry)
    except Exception as e:
        logger.warning(f"NLTK BLEU Berechnung fehlgeschlagen für T='{target}', P='{prediction}': {e}")
        return 0.0

def compute_all_metrics(true_strs: list[str], pred_strs: list[str]) -> dict:
    num_samples = len(true_strs)
    default_metrics = {'cer': 1.0, 'wer': 1.0, 'bleu': 0.0, 'char_f1': 0.0, 'char_precision': 0.0, 'char_recall': 0.0}
    if not true_strs or not pred_strs or len(true_strs) != len(pred_strs):
        logger.warning(f"Ungültige Eingabe für compute_all_metrics (Längen: {len(true_strs) if true_strs else 'None'}, {len(pred_strs) if pred_strs else 'None'}).")
        return default_metrics
    if num_samples == 0:
        return {'cer': 0.0, 'wer': 0.0, 'bleu': 1.0, 'char_f1': 1.0, 'char_precision': 1.0, 'char_recall': 1.0}
    total_cer, total_wer, total_bleu = 0.0, 0.0, 0.0
    total_f1, total_precision, total_recall = 0.0, 0.0, 0.0
    valid_samples_count = 0
    for t, p in zip(true_strs, pred_strs):
        try:
             total_cer += cer(t, p)
             total_wer += wer(t, p)
             total_bleu += compute_bleu(t, p)
             prec, rec, f1 = compute_precision_recall_f1(t, p, level='char')
             total_precision += prec
             total_recall += rec
             total_f1 += f1
             valid_samples_count += 1
        except Exception as metric_err:
             logger.warning(f"Fehler bei Metrikberechnung für T='{t}', P='{p}': {metric_err}", exc_info=False)
    if valid_samples_count == 0:
        logger.error("Keine Metriken konnten für die Samples berechnet werden.")
        return default_metrics
    avg_cer = total_cer / valid_samples_count
    avg_wer = total_wer / valid_samples_count
    avg_bleu = total_bleu / valid_samples_count
    avg_f1 = total_f1 / valid_samples_count
    avg_precision = total_precision / valid_samples_count
    avg_recall = total_recall / valid_samples_count
    return {'cer': avg_cer, 'wer': avg_wer, 'bleu': avg_bleu, 'char_f1': avg_f1,
            'char_precision': avg_precision, 'char_recall': avg_recall}


# convert_metrics_to_serializable bleibt unverändert wie in der vorherigen Antwort
def convert_metrics_to_serializable(data):
    """ Konvertiert Dict/List mit Numpy/Torch Skalaren zu Python Typen für JSON. """
    if isinstance(data, list):
        return [convert_metrics_to_serializable(item) for item in data]
    elif isinstance(data, dict):
        serializable = {}
        for k, v in data.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                 serializable[k] = None
            elif isinstance(v, (np.float_, np.float16, np.float32, np.float64)):
                 if np.isnan(v) or np.isinf(v): serializable[k] = None
                 else: serializable[k] = float(v)
            elif isinstance(v, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                 serializable[k] = int(v)
            elif isinstance(v, torch.Tensor):
                 if v.numel() == 1:
                     val = v.item()
                     if isinstance(val, float) and (math.isnan(val) or math.isinf(val)): serializable[k] = None
                     else: serializable[k] = val
                 else:
                     serializable[k] = convert_metrics_to_serializable(v.tolist())
            elif isinstance(v, (dict, list)):
                 serializable[k] = convert_metrics_to_serializable(v)
            else:
                 try: json.dumps(v); serializable[k] = v
                 except TypeError: logger.debug(f"Konnte Wert vom Typ {type(v)} für Schlüssel '{k}' nicht serialisieren, konvertiere zu String."); serializable[k] = str(v)
        return serializable
    else:
        if isinstance(data, float) and (math.isnan(data) or math.isinf(data)): return None
        elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
             if np.isnan(data) or np.isinf(data): return None
             else: return float(data)
        elif isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(data)
        elif isinstance(data, torch.Tensor):
             if data.numel() == 1:
                 val = data.item()
                 if isinstance(val, float) and (math.isnan(val) or math.isinf(val)): return None
                 else: return val
             else: return convert_metrics_to_serializable(data.tolist())
        else:
             try: json.dumps(data); return data
             except TypeError: logger.debug(f"Konnte einzelnen Wert vom Typ {type(data)} nicht serialisieren, konvertiere zu String."); return str(data)