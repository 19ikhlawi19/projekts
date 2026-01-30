# utilities.py

import os
import csv
import logging
import cv2
import numpy as np
import json
import editdistance # Import hinzugefügt
import torch # Hinzugefügt für Dataset/collate
from torch.utils.data import Dataset # Hinzugefügt für Dataset
import configuration
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
# Importiere augment_image, wenn image_augmentation existiert und benötigt wird
try:
    from image_augmentation import augment_image
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    logging.warning("image_augmentation nicht gefunden oder augment_image nicht importierbar. Augmentierung deaktiviert.")
    def augment_image(image): # Dummy-Funktion
         return image

# -----------------------------------------------------------------------------
# utilities.py
# -----------------------------------------------------------------------------
# Nützliche Hilfsfunktionen. Enthält jetzt auch Dataset, Collate und Ladefunktionen.
# -----------------------------------------------------------------------------

# Logging wird in model_training.py konfiguriert
logger = logging.getLogger(__name__)

# === Standard Hilfsfunktionen ===

def create_directory(path):
    """Erstellt einen Ordner (rekursiv), wenn nicht vorhanden."""
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            # logger.debug(f"Verzeichnis erstellt: {path}")
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Verzeichnisses {path}: {e}", exc_info=True)

def load_image(image_path):
    """
    Lädt, verarbeitet (Größe anpassen, padden) und normalisiert ein Bild.
    Gibt ein Numpy Array (1, H, W) im Bereich [0, 1] oder None bei Fehler zurück.
    """
    # logger.debug(f"Versuche Bild zu laden: {image_path}")
    if not os.path.exists(image_path):
        # Versuche relativen Pfad zu WORDS_FOLDER, falls nicht absolut
        if not os.path.isabs(image_path) and configuration.WORDS_FOLDER:
            image_path = os.path.join(configuration.WORDS_FOLDER, image_path)
            # logger.debug(f"Neuer Versuch mit Pfad: {image_path}")
            if not os.path.exists(image_path):
                logger.error(f"Bilddatei existiert nicht: {image_path}")
                return None
        else:
            logger.error(f"Bilddatei existiert nicht: {image_path}")
            return None

    try:
        # Lade als Graustufenbild
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"Bild konnte nicht geladen werden (leer): {image_path}")
            return None

        # Höhe anpassen unter Beibehaltung des Seitenverhältnisses
        target_h = configuration.IMG_HEIGHT
        h, w = img.shape
        if h == 0 or w == 0:
             logger.error(f"Bild hat ungültige Dimensionen (HxW): {h}x{w} in {image_path}")
             return None

        ratio = target_h / h
        new_w = int(w * ratio)

        # Begrenze die Breite, falls dynamische Breite verwendet wird
        if configuration.USE_DYNAMIC_WIDTH:
            new_w = min(new_w, configuration.MAX_IMG_WIDTH)
        else:
             # Wenn keine dynamische Breite, direkt auf PAD_TO_WIDTH resizen?
             # Besser erst Höhe anpassen, dann padden/abschneiden
             pass # Breite wird später angepasst

        # Resizen auf Zielhöhe
        img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR) # INTER_LINEAR ist oft gut

        # Paddding oder Cropping auf feste Breite PAD_TO_WIDTH
        current_h, current_w = img.shape
        target_w = configuration.PAD_TO_WIDTH

        if current_w < target_w:
            # Pad rechts mit Weiß (255)
            pad_width = target_w - current_w
            # np.pad: ((top, bottom), (left, right))
            img = np.pad(img, ((0, 0), (0, pad_width)), mode='constant', constant_values=255)
        elif current_w > target_w:
            # Crop rechts
            img = img[:, :target_w]

        # Normalisierung auf [0, 1] und Typkonvertierung
        img = img.astype(np.float32) / 255.0

        # Dimension hinzufügen für Kanal -> (1, H, W)
        img = np.expand_dims(img, axis=0)
        # logger.debug(f"Bild {os.path.basename(image_path)} geladen und verarbeitet. Shape: {img.shape}")
        return img

    except Exception as e:
        logger.error(f"Fehler bei der Bildverarbeitung für {image_path}: {e}", exc_info=True)
        return None


def save_image(image, save_path):
    """Speichert ein Numpy-Array (1,H,W) oder (H,W) als PNG/JPG."""
    try:
        create_directory(os.path.dirname(save_path))
        # Konvertiere zurück in uint8 [0, 255]
        if image.max() <= 1.0: # Annahme: Bild ist im Bereich [0, 1]
             image = (image * 255).clip(0, 255).astype(np.uint8)
        else: # Annahme: Bild ist bereits [0, 255]
             image = image.clip(0, 255).astype(np.uint8)

        # Passe Dimensionen an für cv2.imwrite (erwartet H, W oder H, W, C)
        if image.ndim == 3 and image.shape[0] == 1:
            image = np.squeeze(image, axis=0)  # => (H, W)
        elif image.ndim == 3 and image.shape[0] == 3:
            # Falls Farbkanal zuerst ist (C, H, W) -> (H, W, C)
            image = np.transpose(image, (1, 2, 0))
        elif image.ndim != 2 and not (image.ndim == 3 and image.shape[2] in [1, 3, 4]):
             logger.error(f"Ungültige Bilddimensionen zum Speichern: {image.shape}")
             return

        cv2.imwrite(save_path, image)
        # logger.debug(f"Bild gespeichert: {save_path}")
    except Exception as e:
        logger.error(f"Fehler beim Speichern des Bildes {save_path}: {e}", exc_info=True)

def load_labels(labels_file_path):
    """
    Liest eine CSV-Datei [filename, label_string].
    Gibt ein dict zurück: {filename: label_string}.
    """
    labels = {}
    if not os.path.exists(labels_file_path):
        logger.warning(f"Labels-Datei nicht gefunden: {labels_file_path}")
        return labels
    try:
        with open(labels_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                 header = next(reader) # Header lesen
                 # logger.debug(f"CSV Header: {header}")
                 # Optional: Header validieren
                 # if header != ['filename', 'label']:
                 #    logger.warning(f"Unerwarteter Header in {labels_file_path}: {header}")
            except StopIteration:
                 logger.warning(f"Label-Datei ist leer: {labels_file_path}")
                 return labels

            line_num = 1
            for row in reader:
                line_num += 1
                if len(row) >= 2:
                    filename, label = row[0].strip(), row[1].strip()
                    if filename: # Nur wenn Dateiname nicht leer ist
                         labels[filename] = label
                    else:
                         logger.warning(f"Leerer Dateiname in Zeile {line_num} von {labels_file_path}")
                elif row: # Zeile nicht leer, aber zu wenige Spalten
                     logger.warning(f"Ungültige Zeile (weniger als 2 Spalten) in {labels_file_path}, Zeile {line_num}: {row}")

        logger.info(f"{len(labels)} Labels geladen aus {labels_file_path}")
        return labels
    except Exception as e:
        logger.error(f"Fehler beim Laden der Labels aus {labels_file_path}: {e}", exc_info=True)
        return {} # Leeres Dict bei Fehler zurückgeben

def save_labels(labels_dict, output_file):
    """Speichert dict {filename: label_string} in eine CSV [filename,label]."""
    try:
        create_directory(os.path.dirname(output_file))
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename','label']) # Header schreiben
            for filename, label in labels_dict.items():
                writer.writerow([filename, label])
        logger.info(f"{len(labels_dict)} Labels gespeichert in {output_file}")
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Labels in {output_file}: {e}", exc_info=True)

def encode_label(label, char_to_idx):
    """Wandelt einen String in eine Liste von Character-Indizes um."""
    encoded = []
    unknown_chars = set()
    for c in label:
        idx = char_to_idx.get(c) # Schneller als 'if c in ...'
        if idx is not None:
            encoded.append(idx)
        else:
            # Behandle unbekannte Zeichen: Ignorieren, 'blank' oder spezielles Symbol?
            # Aktuell: Ignorieren (fügt nichts zur Liste hinzu)
            # Alternativ: encoded.append(char_to_idx['blank'])
            if c not in unknown_chars:
                 # logger.warning(f"Unbekanntes Zeichen im Label '{label}': '{c}'. Wird ignoriert.")
                 unknown_chars.add(c)
            pass # Ignorieren
    # if unknown_chars:
    #     logger.warning(f"Label '{label}' enthielt unbekannte Zeichen: {unknown_chars}")
    return encoded

def decode_labels(encoded_labels, num_classes):
    """
    Dekodiert eine Sequenz von Indizes zurück in einen String.
    Entfernt *nicht* den CTC-Blank oder Wiederholungen (das macht decode_predictions).
    """
    blank_idx = num_classes - 1
    decoded_chars = []
    for idx in encoded_labels:
        if idx != blank_idx:
            # Finde Zeichen für Index, verwende '' falls nicht gefunden
            decoded_chars.append(configuration.IDX_TO_CHAR.get(idx, ''))
    return ["".join(decoded_chars)] # Gibt immer eine Liste mit einem String zurück

def decode_predictions(predictions_np, idx_to_char, num_classes):
    """
    Dekodiert Rohausgaben des Modells (Logits oder Probs) mit CTC-Logik.
    Shape (T, N, C) -> Liste von N Strings.
    Verwendet Greedy Decoding (ArgMax).
    """
    blank_idx = num_classes - 1
    T, N, C = predictions_np.shape

    # Argmax über Klassen-Dimension (C) -> Shape (T, N)
    pred_indices = np.argmax(predictions_np, axis=2)

    result_strings = []
    for n in range(N): # Iteriere über Batch
        raw_indices = pred_indices[:, n]
        decoded_sequence = []
        last_char_idx = blank_idx
        for t in range(T): # Iteriere über Zeit
            current_char_idx = raw_indices[t]
            # CTC Collapse: Ignoriere Blanks und aufeinanderfolgende gleiche Zeichen
            if current_char_idx != blank_idx and current_char_idx != last_char_idx:
                decoded_sequence.append(idx_to_char.get(current_char_idx, ''))
            last_char_idx = current_char_idx
        result_strings.append("".join(decoded_sequence))
    return result_strings


def levenshtein_distance(s1, s2):
    """Berechnet den Levenshtein-Abstand zwischen zwei Strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_wer(pred_str, true_str):
    """
    Berechnet die Word Error Rate (WER) zwischen zwei Strings.
    WER = Levenshtein-Distanz auf Wortebene / Anzahl der Wörter im wahren String.
    """
    pred_words = pred_str.split()
    true_words = true_str.split()

    # Levenshtein auf Wortebene
    dist = levenshtein_distance(pred_words, true_words)

    # Normalisieren durch Anzahl der wahren Wörter (mindestens 1, um Division durch Null zu vermeiden)
    num_true_words = max(1, len(true_words))
    wer = dist / num_true_words
    return wer


def calculate_bleu_scores(all_true, all_pred):
    """
    Berechnet BLEU-Scores (1-4) für die gegebenen wahren und vorhergesagten Labels.
    Nimmt Listen von Strings entgegen.
    """
    if not all_true or not all_pred:
        return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}

    try:
        # Referenzen müssen eine Liste von Listen von Listen sein für corpus_bleu
        # [[['ref1a', 'word'], ['ref1b']], [['ref2a']]] -> Hier: nur eine Referenz pro Sample
        references = [[t.split()] for t in all_true]
        hypotheses = [h.split() for h in all_pred]

        smoothing = SmoothingFunction().method1 # Methode 1 ist üblich

        bleu_scores = {}
        for n in range(1, 5):
            weights = tuple(1.0 / n for _ in range(n)) + tuple(0.0 for _ in range(4 - n))
            try:
                # Verwende corpus_bleu für den Durchschnitt über den gesamten Datensatz
                score = corpus_bleu(references, hypotheses, weights=weights, smoothing_function=smoothing)
            except ZeroDivisionError:
                 # logger.warning(f"ZeroDivisionError bei BLEU-{n} Berechnung. Setze auf 0.")
                 score = 0.0
            bleu_scores[f'bleu_{n}'] = score

        return bleu_scores
    except Exception as e:
        logger.error(f"Fehler bei BLEU-Berechnung: {e}", exc_info=True)
        return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}


def compute_character_level_metrics(all_true, all_pred):
    """
    Berechnet Precision, Recall und F1-Score auf Zeichenebene über den gesamten Datensatz.
    Verwendet eine interne Implementierung von Levenshtein mit Detail-Tracking.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_true_chars = 0
    # total_pred_chars = 0 # Nicht unbedingt nötig für Prec/Rec/F1

    if not all_true: # Handle empty input
        return 0.0, 0.0, 0.0

    def levenshtein_details(s1, s2):
        """ Berechnet Levenshtein-Distanz und Anzahl von s, i, d Operationen. """
        m, n = len(s1), len(s2)
        # Initialisiere DP-Tabelle für Distanzen und Operationen
        dp = np.zeros((m + 1, n + 1), dtype=int)
        # Speichere (s, i, d) Zählungen für jeden Pfad
        ops = [[(0, 0, 0)] * (n + 1) for _ in range(m + 1)]

        # Fülle erste Zeile und Spalte
        for i in range(m + 1):
            dp[i, 0] = i
            ops[i][0] = (0, 0, i) # Nur Deletionen von s1
        for j in range(n + 1):
            dp[0, j] = j
            ops[0][j] = (0, j, 0) # Nur Insertionen in s1

        # Fülle den Rest der Tabelle
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                # Kosten für jede Operation vom vorherigen Zustand
                sub_cost = dp[i - 1, j - 1] + cost
                ins_cost = dp[i, j - 1] + 1
                del_cost = dp[i - 1, j] + 1

                # Finde minimale Kosten
                min_cost = min(sub_cost, ins_cost, del_cost)
                dp[i, j] = min_cost

                # Bestimme die Operationen, die zu min_cost führten
                # Priorisiere Substitution (falls Kosten gleich), dann Deletion, dann Insertion (willkürlich, aber konsistent)
                if min_cost == sub_cost:
                    s_prev, i_prev, d_prev = ops[i-1][j-1]
                    ops[i][j] = (s_prev + cost, i_prev, d_prev) # cost ist 1 bei Sub, 0 bei Match
                elif min_cost == del_cost: # Beachte: del_cost bezieht sich auf Löschung aus s1
                     s_prev, i_prev, d_prev = ops[i-1][j]
                     ops[i][j] = (s_prev, i_prev, d_prev + 1)
                else: # min_cost == ins_cost # Beachte: ins_cost bezieht sich auf Einfügung in s1
                     s_prev, i_prev, d_prev = ops[i][j-1]
                     ops[i][j] = (s_prev, i_prev + 1, d_prev)

        # Die endgültige Anzahl der Operationen (s, i, d) an der Position [m][n]
        return ops[m][n] # Gibt (Substitutions, Insertions, Deletions) zurück


    for true_str, pred_str in zip(all_true, all_pred):
        # Leere Strings behandeln
        len_true = len(true_str)
        len_pred = len(pred_str)
        total_true_chars += len_true

        if len_true == 0 and len_pred == 0:
            continue # Korrektes Paar, keine Fehler, keine Chars
        elif len_true == 0: # Nur Prediction -> alle Zeichen FP
             total_fp += len_pred
             continue
        elif len_pred == 0: # Nur True -> alle Zeichen FN
             total_fn += len_true
             continue

        # Berechne s, i, d
        try:
            # Reihenfolge wichtig: levenshtein_details(prediction, true)
            s, i, d = levenshtein_details(pred_str, true_str)
        except Exception as e:
             logger.error(f"Fehler bei levenshtein_details für '{pred_str}' vs '{true_str}': {e}", exc_info=True)
             continue # Überspringe dieses Paar

        # Berechne TP, FP, FN basierend auf s, i, d
        # TP = Anzahl der korrekten Zeichen = len(true) - Substitutions - Deletions
        # FP = Anzahl der falsch von Modell eingefügten/substituierten = Insertions + Substitutions
        # FN = Anzahl der vom Modell übersehenen/falsch substituierten = Deletions + Substitutions
        tp = len_true - s - d
        fp = i + s
        fn = d + s

        # Stelle sicher, dass TP nicht negativ ist (kann bei Fehlern passieren)
        tp = max(0, tp)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Gesamte Metriken berechnen
    # Precision = TP / (TP + FP) = Korrekt erkannt / Gesamt erkannt vom Modell
    char_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    # Recall = TP / (TP + FN) = Korrekt erkannt / Gesamt in Wahrheit vorhanden
    char_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    # F1 = Harmonisches Mittel
    char_f1 = 2 * (char_precision * char_recall) / (char_precision + char_recall) if (char_precision + char_recall) > 0 else 0.0

    # logger.debug(f"Char Metrics Agg: TP={total_tp}, FP={total_fp}, FN={total_fn}")
    # logger.debug(f"Calculated Agg: Precision={char_precision:.4f}, Recall={char_recall:.4f}, F1={char_f1:.4f}")

    return char_precision, char_recall, char_f1


def compute_metrics(all_true, all_pred):
    """
    Berechnet alle relevanten Metriken für die gegebenen wahren und vorhergesagten Werte.
    Jetzt mit Character-Level F1/Prec/Rec.
    """
    metrics = {}

    if not all_true or len(all_true) == 0:
        logger.warning("Keine Daten für Metrikberechnung vorhanden.")
        # Setze Standardwerte für alle erwarteten Metriken
        # Verwende configuration.METRICS_TO_COMPUTE als Basis
        metrics = {m: 0.0 for m in configuration.METRICS_TO_COMPUTE if m != 'loss'}
        # Füge spezielle Metriken hinzu, die nicht direkt in der Liste stehen (z.B. andere BLEUs)
        metrics.update({'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0})
        metrics['loss'] = float('inf') # Oder einen anderen geeigneten Wert
        return metrics

    num_samples = len(all_true)

    # Character Error Rate (CER)
    total_chars = sum(max(1, len(tr)) for tr in all_true) # Mindestens 1 pro Sample zählen
    total_dist = sum(levenshtein_distance(pr, tr) for pr, tr in zip(all_pred, all_true))
    metrics['cer'] = total_dist / total_chars if total_chars > 0 else 0.0

    # Word Error Rate (WER)
    wer_sum = sum(compute_wer(pr, tr) for pr, tr in zip(all_pred, all_true))
    metrics['wer'] = wer_sum / num_samples if num_samples > 0 else 0.0

    # Word Accuracy (exakter Wort-Vergleich)
    correct_words = sum(1 for pr, tr in zip(all_pred, all_true) if pr == tr)
    metrics['word_accuracy'] = correct_words / num_samples if num_samples > 0 else 0.0

    # Character-Level Precision, Recall, F1
    char_precision, char_recall, char_f1 = compute_character_level_metrics(all_true, all_pred)
    metrics['char_precision'] = char_precision
    metrics['char_recall'] = char_recall
    metrics['char_f1'] = char_f1

    # BLEU Scores (berechnet jetzt alle 1-4)
    bleu_scores = calculate_bleu_scores(all_true, all_pred)
    metrics.update(bleu_scores) # Fügt bleu_1, bleu_2, bleu_3, bleu_4 hinzu

    # Filtere Metriken basierend auf configuration.METRICS_TO_COMPUTE für die finale Ausgabe
    # (Loss wird extern hinzugefügt)
    final_metrics = {}
    for key in configuration.METRICS_TO_COMPUTE:
        if key != 'loss':
            # Suche nach dem Schlüssel oder einem ähnlichen (z.B. bleu -> bleu_4)
            if key in metrics:
                 final_metrics[key] = metrics[key]
            elif key == 'bleu' and 'bleu_4' in metrics: # Falls nur 'bleu' gewünscht, nimm bleu_4
                 final_metrics[key] = metrics['bleu_4']
            else:
                 # logger.warning(f"Metrik '{key}' aus configuration.METRICS_TO_COMPUTE nicht in berechneten Metriken gefunden.")
                 final_metrics[key] = 0.0 # Defaultwert

    # Füge andere berechnete Metriken hinzu, die nicht explizit in der Liste waren (z.B. andere BLEUs)
    for k, v in metrics.items():
        if k not in final_metrics and k != 'loss':
             final_metrics[k] = v

    return final_metrics


def save_split_data(paths, labels, split_name):
    """
    Speichert die Datenpfade und (kodierten) Labels für einen bestimmten Split als CSV.
    Die Labels werden vor dem Speichern in Strings dekodiert.
    """
    # Bestimme Ausgabedatei basierend auf Split-Namen
    if split_name == 'train':
        output_file = configuration.TRAIN_LABELS_FILE
    elif split_name == 'val':
        output_file = configuration.VAL_LABELS_FILE
    elif split_name == 'test':
        output_file = configuration.TEST_LABELS_FILE
    else:
        logger.error(f"Unbekannter Split-Name zum Speichern: {split_name}")
        return

    # Speichere Labels als CSV (dekodiert)
    save_labels_as_csv(paths, labels, output_file)
    logger.info(f"{split_name.capitalize()}-Split gespeichert: {output_file} ({len(paths)} Bilder)")

def save_labels_as_csv(paths, labels, output_file):
    """
    Speichert Pfade und Labels als CSV-Datei.
    Nimmt Pfade und kodierte Labels (Listen von ints) entgegen und dekodiert die Labels.
    """
    try:
        create_directory(os.path.dirname(output_file))
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'label'])
            for path, label_data in zip(paths, labels):
                # Konvertiere kodierte Labels zurück in Strings
                if isinstance(label_data, list):
                    # Filtert Blank-Index heraus
                    label_str = "".join([configuration.IDX_TO_CHAR.get(idx, '') for idx in label_data
                                         if idx != configuration.CHAR_TO_IDX['blank']])
                elif isinstance(label_data, str):
                     label_str = label_data # Falls schon ein String übergeben wurde
                else:
                     logger.warning(f"Unbekannter Label-Typ in save_labels_as_csv für {path}: {type(label_data)}. Speichere leer.")
                     label_str = ""

                writer.writerow([path, label_str])
        # logger.debug(f"Labels als CSV gespeichert: {output_file}")
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Labels als CSV {output_file}: {e}", exc_info=True)


def save_metrics_to_file(metrics, split_name, output_dir=None):
    """
    Speichert berechnete Metriken für einen bestimmten Split in einer JSON-Datei.
    """
    if output_dir is None:
        output_dir = configuration.METRICS_PATH

    create_directory(output_dir)
    # Füge Suffix hinzu, um Konflikte mit Ordnernamen zu vermeiden
    output_file = os.path.join(output_dir, f"{split_name}_metrics.json")

    try:
        # Konvertiere numpy Typen etc. für JSON Serialisierung
        def default_serializer(obj):
             if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                 np.int16, np.int32, np.int64, np.uint8,
                                 np.uint16, np.uint32, np.uint64)):
                 return int(obj)
             elif isinstance(obj, (np.float_, np.float16, np.float32,
                                   np.float64)):
                 return float(obj)
             elif isinstance(obj, (np.ndarray,)):
                 return obj.tolist()
             # Optional: datetime etc. behandeln
             # elif isinstance(obj, datetime): return obj.isoformat()
             raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable in save_metrics_to_file")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, default=default_serializer)
        logger.info(f"Metriken für {split_name}-Split gespeichert: {output_file}")
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Metriken für {split_name}: {e}", exc_info=True)

def load_split_data(split_file):
    """
    Lädt Daten aus einer Split-Datei (CSV: filename, label_string).
    Gibt Pfade und *kodierte* Labels (Listen von ints) zurück.
    """
    if not os.path.exists(split_file):
        logger.warning(f"Split-Datei nicht gefunden: {split_file}")
        return [], []

    paths = []
    encoded_labels = [] # Wird jetzt Listen von Ints speichern

    try:
        with open(split_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                header = next(reader) # Header überspringen
            except StopIteration:
                 logger.warning(f"Split-Datei {split_file} ist leer.")
                 return [], []

            line_num = 1
            for row in reader:
                line_num += 1
                if len(row) >= 2:
                    path = row[0].strip()
                    label_str = row[1].strip()
                    if not path:
                         logger.warning(f"Leerer Pfad in Zeile {line_num} von {split_file}. Überspringe.")
                         continue

                    # Optional: Pfad validieren oder normalisieren
                    # z.B. sicherstellen, dass er relativ ist oder existiert

                    paths.append(path) # Speichere den Pfad so, wie er in der CSV steht
                    # Kodiere das Label direkt beim Laden
                    encoded = encode_label(label_str, configuration.CHAR_TO_IDX)
                    encoded_labels.append(encoded)
                elif row:
                     logger.warning(f"Ungültige Zeile in {split_file}, Zeile {line_num}: {row}")


        logger.info(f"Split-Daten geladen aus {split_file}: {len(paths)} Einträge.")
        return paths, encoded_labels
    except Exception as e:
        logger.error(f"Fehler beim Laden der Split-Daten aus {split_file}: {e}", exc_info=True)
        return [], []

# === Aus training_offline.py hierher verschobene Komponenten ===

class OfflineHandwritingDataset(Dataset):
    def __init__(self, image_paths, labels, augment=False):
        super().__init__()
        if len(image_paths) != len(labels):
            raise ValueError(f"Anzahl der Bildpfade ({len(image_paths)}) stimmt nicht mit Anzahl der Labels ({len(labels)}) überein.")
        self.image_paths = image_paths
        # Labels sollten bereits kodierte Listen von Ints sein
        self.labels = labels
        self.augment = augment and AUGMENTATION_AVAILABLE # Nur augmentieren, wenn verfügbar
        # logger.info(f"Dataset initialisiert: {len(self.image_paths)} Samples, Augment={self.augment}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path_relative = self.image_paths[idx]
        label = self.labels[idx] # Ist eine Liste von Ints

        # Lade Bild über utils.load_image (kümmert sich um Pfad und Verarbeitung)
        image = load_image(img_path_relative)

        if image is None:
            # load_image hat bereits geloggt, hier nur Platzhalter zurückgeben
            # Erzeuge ein leeres Bild als Platzhalter
            image = np.zeros((1, configuration.IMG_HEIGHT, configuration.PAD_TO_WIDTH), dtype=np.float32)
            # Gib das leere Bild und das Original-Label zurück
            return (image, label)

        if self.augment:
            try:
                # augment_image erwartet (1, H, W) numpy array oder (H, W)
                # und gibt numpy array zurück
                aug_img_np = augment_image(image)
                if aug_img_np is not None:
                    # Sicherstellen, dass die Form (1, H, W) ist
                    if aug_img_np.ndim == 2:
                        image = np.expand_dims(aug_img_np, axis=0)
                    elif aug_img_np.ndim == 3 and aug_img_np.shape[0] == 1:
                        image = aug_img_np
                    else: # Unerwartete Form
                         image = image # Behalte Original bei
                         # logger.warning(f"Unerwartete Form von augmentiertem Bild: {aug_img_np.shape}. Verwende Original.")
                # else: logger.warning(f"Augmentierung lieferte None für {img_path_relative}. Verwende Original.")
            except Exception as e:
                 logger.error(f"Fehler bei Augmentierung für {img_path_relative}: {e}", exc_info=True)
                 # Behalte Originalbild bei Fehler

        # Stelle sicher, dass das finale Bild die korrekte Form hat
        if image.shape != (1, configuration.IMG_HEIGHT, configuration.PAD_TO_WIDTH):
             # Sollte durch load_image abgedeckt sein, aber als Sicherheitsnetz
             logger.error(f"Inkonsistente Bildform nach Laden/Augmentieren für {img_path_relative}: {image.shape}. Erwarte {(1, configuration.IMG_HEIGHT, configuration.PAD_TO_WIDTH)}")
             # Versuche zu korrigieren oder leeres Bild zurückgeben
             image = np.zeros((1, configuration.IMG_HEIGHT, configuration.PAD_TO_WIDTH), dtype=np.float32)

        return (image, label) # Gibt (1, H, W) numpy array und Liste von Ints zurück


def collate_fn(batch):
    """
    Collate-Funktion für den DataLoader.
    Verarbeitet eine Liste von Tupeln (image, label_list) vom Dataset.
    Behandelt potenzielle Fehler im Batch.
    """
    images = []
    labels = []
    label_lengths = []
    valid_batch_indices = [] # Indizes der gültigen Samples im Original-Batch

    for i, item in enumerate(batch):
        # Prüfe, ob das Item (Bild, Label) gültig ist
        if item is not None and isinstance(item, tuple) and len(item) == 2 and \
           isinstance(item[0], np.ndarray) and item[0].shape == (1, configuration.IMG_HEIGHT, configuration.PAD_TO_WIDTH) and \
           isinstance(item[1], list):
            valid_batch_indices.append(i)
            images.append(item[0])
            labels.extend(item[1])
            label_lengths.append(len(item[1]))
        else:
             # Logge das Problem nur einmal pro Batch-Typ
             if not hasattr(collate_fn, f"_logged_invalid_batch_{type(item)}"):
                  logger.warning(f"Ungültiges Sample im Batch gefunden (Index {i}, Typ: {type(item)}). Überspringe. Weitere Warnungen dieses Typs unterdrückt.")
                  setattr(collate_fn, f"_logged_invalid_batch_{type(item)}", True)


    if not images:
         # Wenn der gesamte Batch ungültig ist
         logger.error("Ganzer Batch war ungültig in collate_fn. Kann nicht fortfahren.")
         # Rückgabe leerer Tensoren kann zu Fehlern im Loop führen. Besser Fehler werfen?
         # Oder spezielle Behandlung im Trainingsloop.
         # Wir geben leere Tensoren zurück, Loop muss prüfen.
         return (torch.empty(0, 1, configuration.IMG_HEIGHT, configuration.PAD_TO_WIDTH, dtype=torch.float32),
                 torch.empty(0, dtype=torch.long),
                 torch.empty(0, dtype=torch.long))

    try:
        # Stapel die Bilder entlang einer neuen Dimension (Batch-Dimension)
        # Ergebnis-Shape: (N, 1, H, W)
        batch_images_np = np.stack(images, axis=0)
        # Konvertiere zu FloatTensor
        batch_images = torch.from_numpy(batch_images_np).float()

        # Labels und Längen
        batch_labels = torch.LongTensor(labels)
        label_lengths = torch.LongTensor(label_lengths)

        return batch_images, batch_labels, label_lengths

    except Exception as e:
         logger.error(f"Fehler beim Konvertieren/Stacken des Batches in collate_fn: {e}", exc_info=True)
         # Rückgabe leerer Tensoren
         return (torch.empty(0, 1, configuration.IMG_HEIGHT, configuration.PAD_TO_WIDTH, dtype=torch.float32),
                 torch.empty(0, dtype=torch.long),
                 torch.empty(0, dtype=torch.long))


def load_data_for_ctc(csv_file, root_folder):
    """
    Lädt die Daten aus einer CSV-Datei (filename, label_string)
    und konvertiert die Labels in kodierte Listen von Indizes.
    Gibt relative Pfade (oder Original-Dateinamen aus CSV) und kodierte Labels zurück.
    """
    if not os.path.exists(csv_file):
         logger.error(f"CSV-Datei nicht gefunden: {csv_file}")
         return [], []

    label_dict = load_labels(csv_file) # Lädt {filename: label_str}
    if not label_dict:
        logger.error(f"Keine Labels konnten aus {csv_file} geladen werden.")
        return [], []

    all_image_paths = [] # Pfade wie in CSV oder relativ zu root_folder
    all_encoded_labels = []
    skipped_count = 0
    long_label_count = 0

    for fname_csv, text_label in label_dict.items():
        # Prüfe Pfad: Absolut, relativ zu root, oder nur Dateiname?
        path_to_check = fname_csv
        final_path_to_store = fname_csv # Standard: Speichere, was in CSV steht

        if not os.path.isabs(fname_csv) and root_folder:
             path_to_check = os.path.join(root_folder, fname_csv)
             # Falls der Pfad in CSV Verzeichnisse enthält, behalte diese bei
             # final_path_to_store bleibt fname_csv (relativ zum root_folder)

        # Prüfe Existenz
        if not os.path.exists(path_to_check):
            # Alternativ: Suche nur nach Dateiname in root_folder
            basename_path = os.path.join(root_folder, os.path.basename(fname_csv))
            if os.path.exists(basename_path):
                 path_to_check = basename_path
                 # Wenn wir nur den Basisnamen verwenden, speichern wir auch nur diesen
                 final_path_to_store = os.path.basename(fname_csv)
                 # logger.debug(f"Verwende Basisnamen für {fname_csv}: {final_path_to_store}")
            else:
                # logger.warning(f"Bild nicht gefunden: {path_to_check} (aus {fname_csv}). Überspringe.")
                skipped_count += 1
                continue

        # Kodiere Label
        encoded = encode_label(text_label, configuration.CHAR_TO_IDX)

        # Filtere ungültige/leere/zu lange Labels
        if encoded and 0 < len(encoded) <= configuration.MAX_LABEL_LENGTH:
             all_image_paths.append(final_path_to_store)
             all_encoded_labels.append(encoded)
        elif encoded and len(encoded) > configuration.MAX_LABEL_LENGTH:
             # logger.warning(f"Label für {final_path_to_store} zu lang ({len(encoded)} > {configuration.MAX_LABEL_LENGTH}): '{text_label}'. Überspringe.")
             long_label_count += 1
             skipped_count += 1
        else: # Label leer oder nur unbekannte Zeichen
             # logger.debug(f"Label für {final_path_to_store} ist leer nach Kodierung. Überspringe.")
             skipped_count += 1


    logger.info(f"Daten für CTC geladen aus {csv_file}: {len(all_image_paths)} gültige Samples.")
    if skipped_count > 0:
        logger.warning(f"  {skipped_count} Samples übersprungen (davon {long_label_count} wegen zu langer Labels).")
    return all_image_paths, all_encoded_labels

# === Ende der verschobenen Komponenten ===