# training_offline.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utilities.data import DataLoader, Dataset
from torch.nn.utilities.rnn import pad_sequence
from torch.amp import autocast # Neuer Import für autocast
from torch.cuda.amp import GradScaler
import numpy as np
import logging
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
import time
import random
import math
import traceback # Für detailliertere Fehlermeldungen

# Lokale Modulimporte
import configuration
import utilities
import ocr_model
from data_augmentation import augment_offline_image
from data_preparation import create_offline_splits

logger = logging.getLogger(__name__)

# CustomSchedule Klasse bleibt unverändert...
class CustomSchedule(optim.lr_scheduler._LRScheduler):
    """Custom LR scheduler: Linear warmup followed by cosine decay."""
    def __init__(self, optimizer, init_lr, peak_lr, final_lr,
                 warmup_steps, decay_steps, last_epoch=-1):
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.final_lr = final_lr
        self.warmup_steps = max(1, warmup_steps)
        self.decay_steps = max(1, decay_steps)
        self.decay_start_step = self.warmup_steps
        self.total_steps = self.warmup_steps + self.decay_steps
        super().__init__(optimizer, last_epoch)
        logger.info("CustomSchedule (Warmup + Cosine Decay) Initialized:")
        logger.info(f"  Warmup Steps: {self.warmup_steps}, Init LR: {self.init_lr:.2e}, Peak LR: {self.peak_lr:.2e}")
        logger.info(f"  Decay Steps: {self.decay_steps}, Final LR: {self.final_lr:.2e}")

    def get_lr(self):
        current_step = self.last_epoch # Zählt Optimizer-Schritte
        if current_step < self.warmup_steps:
            lr = self.init_lr + (self.peak_lr - self.init_lr) * (current_step / self.warmup_steps) if self.warmup_steps > 0 else self.peak_lr
        elif current_step < self.total_steps:
            steps_into_decay = current_step - self.decay_start_step
            cosine_factor = 0.5 * (1 + math.cos(math.pi * steps_into_decay / self.decay_steps)) if self.decay_steps > 0 else 0.0
            lr = self.final_lr + cosine_factor * (self.peak_lr - self.final_lr)
        else:
            lr = self.final_lr
        lr = max(lr, self.final_lr if self.final_lr > 0 else 1e-9) # Mindest-LR
        return [lr for _ in self.optimizer.param_groups]


# OfflineHandwritingDataset und collate_fn bleiben unverändert...
class OfflineHandwritingDataset(Dataset):
    """ Lädt Offline-Bilder und kodierte Labels für den Transformer. """
    def __init__(self, relative_image_paths, encoded_labels, image_root_folder, augment=False):
        super().__init__()
        if len(relative_image_paths) != len(encoded_labels):
             logger.error(f"FATAL: Pfade ({len(relative_image_paths)}) != Labels ({len(encoded_labels)})")
             raise ValueError("Path and label counts do not match!")
        self.relative_image_paths = relative_image_paths
        self.encoded_labels = encoded_labels
        self.image_root_folder = image_root_folder
        self.augment = augment
        self.fail_count = 0
        logger.info(f"OfflineDataset '{'Augmented' if augment else 'Standard'}' created: {len(self)} samples from root {image_root_folder}")

    def __len__(self): return len(self.relative_image_paths)

    def __getitem__(self, idx):
        if idx >= len(self): logger.error(f"Index {idx} out of bounds"); return self._get_fallback_item(idx) # Korrektur: idx übergeben
        rel_path = self.relative_image_paths[idx]
        label_ids = self.encoded_labels[idx]
        abs_path = os.path.join(self.image_root_folder, rel_path).replace("\\", "/")

        try:
            image_tensor = utilities.load_image(abs_path)
            if image_tensor is None: raise FileNotFoundError(f"Failed to load image: {abs_path}")

            if self.augment:
                 augmented_tensor = augment_offline_image(image_tensor)
                 if augmented_tensor is not None: image_tensor = augmented_tensor
                 else: logger.warning(f"Augmentierung gab None zurück for {rel_path}. Nutze Original.")

            # Stelle sicher, dass label_ids eine Liste von Integers ist
            if not isinstance(label_ids, list) or not all(isinstance(i, int) for i in label_ids):
                 raise TypeError(f"Invalid label format for {rel_path}. Expected list of ints, got {type(label_ids)}")

            label_tensor = torch.tensor(label_ids, dtype=torch.long)
            return image_tensor, label_tensor, rel_path
        except Exception as e:
             self.fail_count += 1
             if self.fail_count <= 20 or self.fail_count % 100 == 0:
                  logger.warning(f"Fehler Laden/Augmentieren Sample {idx} ({rel_path}): {e}. Nutze Fallback. (Fehler #{self.fail_count})", exc_info=False)
             return self._get_fallback_item(idx)

    def _get_fallback_item(self, index=None):
        path_info = f"Index {index}" if index is not None else "Unbekannt"
        fallback_image = torch.ones((configuration.IMG_CHANNELS, configuration.IMG_HEIGHT, configuration.PAD_TO_WIDTH), dtype=torch.float32)
        # Erstelle Fallback-Label nur mit Start/End, da Vektorizer evtl. nicht verfügbar ist
        fallback_label = torch.tensor([2, 3], dtype=torch.long) # Annahme: START=2, END=3
        try:
            vectorizer = utilities.VectorizeChar()
            fallback_label = torch.tensor([vectorizer.start_token_id, vectorizer.end_token_id], dtype=torch.long)
        except Exception: pass # Ignoriere Fehler hier, nimm Hardcoded Wert
        fallback_path = f"FALLBACK/PATH/ERROR-{path_info}"
        return fallback_image, fallback_label, fallback_path

def collate_fn(batch):
    """ Collate für Offline-Transformer. """
    valid_batch = []
    # Filtere ungültige Samples (z.B. Ladefehler) strenger
    for item in batch:
        if not (isinstance(item, tuple) and len(item) == 3): continue
        img, lbl, path = item
        if not (isinstance(img, torch.Tensor) and img.nelement() > 0 and
                img.shape == (configuration.IMG_CHANNELS, configuration.IMG_HEIGHT, configuration.PAD_TO_WIDTH)):
             logger.warning(f"CollateFn: Ungültiges Bild-Tensor-Format übersprungen: {img.shape if isinstance(img, torch.Tensor) else type(img)} für Pfad {path}")
             continue
        if not (isinstance(lbl, torch.Tensor) and lbl.nelement() > 0 and lbl.ndim == 1 and lbl.dtype == torch.long):
             logger.warning(f"CollateFn: Ungültiges Label-Tensor-Format übersprungen: {lbl.shape if isinstance(lbl, torch.Tensor) else type(lbl)} für Pfad {path}")
             continue
        if not isinstance(path, str):
            logger.warning(f"CollateFn: Ungültiger Pfad-Typ übersprungen: {type(path)}")
            continue
        valid_batch.append(item)


    if not valid_batch:
         logger.error("CollateFn: Batch nach Filterung leer!")
         # Erstelle leere Tensoren mit korrekten Dimensionen außer Batch=0
         return {'images': torch.empty((0, configuration.IMG_CHANNELS, configuration.IMG_HEIGHT, configuration.PAD_TO_WIDTH), dtype=torch.float),
                 'targets': torch.empty((0, 0), dtype=torch.long), # Batch=0, SeqLen=0
                 'paths': []}

    images, labels, paths = zip(*valid_batch)
    try:
        batch_images = torch.stack(images, dim=0)
    except Exception as stack_e:
         logger.error(f"Fehler Stacken Bilder: {stack_e}. Größen: {[img.shape for img in images]}")
         # Fallback wie oben
         return {'images': torch.empty((0, configuration.IMG_CHANNELS, configuration.IMG_HEIGHT, configuration.PAD_TO_WIDTH), dtype=torch.float),
                 'targets': torch.empty((0, 0), dtype=torch.long),
                 'paths': []}

    # Padde Target-Sequenzen
    try:
        batch_labels_padded = pad_sequence(labels, batch_first=True, padding_value=configuration.vectorizer.pad_token_id if hasattr(config, 'vectorizer') else 0)
    except Exception as pad_e:
        logger.error(f"Fehler Padden Labels: {pad_e}. Label Längen: {[len(lbl) for lbl in labels]}")
        return {'images': torch.empty((0, configuration.IMG_CHANNELS, configuration.IMG_HEIGHT, configuration.PAD_TO_WIDTH), dtype=torch.float),
                 'targets': torch.empty((0, 0), dtype=torch.long),
                 'paths': []}


    return {'images': batch_images, 'targets': batch_labels_padded, 'paths': list(paths)}

# --- Hinzugefügte Funktion für Live-Logging im Training ---
@torch.no_grad()
def log_training_samples(model, batch, vectorizer, device, epoch_num):
    """ Loggt zufällige Trainingsbeispiele mit Vorhersagen. """
    model.eval() # In Eval-Modus für Generierung wechseln
    images = batch['images'].to(device, non_blocking=True)
    targets = batch['targets'] # Auf CPU für Dekodierung
    paths = batch['paths']
    if images.nelement() == 0 or targets.nelement() == 0: return
    B = images.size(0)
    num_samples_to_log = min(3, B) # Logge bis zu 3 Samples

    logger.info(f"--- Trainingsbeispiele Epoche {epoch_num} (Live Generierung) ---")
    indices_to_log = random.sample(range(B), num_samples_to_log)

    try:
        # Generiere Vorhersagen für die ausgewählten Bilder
        images_to_predict = images[indices_to_log]
        pred_ids = model.generate(src_img=images_to_predict, max_len=configuration.TARGET_MAXLEN, temperature=0.6, top_k=5)
        pred_ids_list = pred_ids.cpu().tolist()

        for i, log_idx in enumerate(indices_to_log):
            true_label_ids = targets[log_idx].cpu().tolist()
            pred_label_ids = pred_ids_list[i]
            rel_path = paths[log_idx]

            # Dekodiere beide mit vectorizer.decode()
            true_str = vectorizer.decode(true_label_ids)
            pred_str = vectorizer.decode(pred_label_ids)

            is_correct = "✅" if true_str == pred_str else "❌"
            logger.info(f"  Train Sample (Bild: ...{rel_path[-40:]}) {is_correct}")
            logger.info(f"    TARGET: '{true_str}'")
            logger.info(f"    PRED  : '{pred_str}'")

    except Exception as e:
        logger.error(f"Fehler bei Live-Generierung/Logging für Trainingsbeispiele: {e}", exc_info=True)
    finally:
        model.train() # Zurück in den Trainingsmodus!
    logger.info("-" * 50)



# --- Trainings- / Validierungs-Helfer (Überarbeitet für Fehleranalyse) ---
def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler, scaler, vectorizer, epoch_num):
    """ Trainiert für eine Epoche (Transformer Offline). """
    model.train()
    total_loss = 0.0; train_samples_processed = 0
    progress_bar = tqdm(dataloader, desc=f"Training Ep {epoch_num}", unit="batch", leave=False)
    optimizer.zero_grad()
    device_type = device.type
    amp_enabled = configuration.USE_MIXED_PRECISION and device_type == 'cuda'
    last_batch_data = None # Für Live-Logging am Ende

    for i, batch in enumerate(progress_bar):
        # Daten holen und prüfen (minimal, da CollateFn bereits prüft)
        images = batch['images']
        targets = batch['targets']
        if images.nelement() == 0 or targets.nelement() == 0:
             logger.warning(f"Train Batch {i}: Leeres Image/Target Tensor nach Collate übersprungen."); continue

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        B = images.size(0)
        if i == len(dataloader) - 1: last_batch_data = batch # Letzten Batch für Logging merken

        decoder_input = targets[:, :-1]
        ground_truth = targets[:, 1:]

        try:
             with autocast(device_type=device_type, enabled=amp_enabled):
                 logits = model(src_img=images, tgt_tokens=decoder_input)
                 # Prüfe auf NaNs in Logits VOR dem Loss
                 if torch.isnan(logits).any() or torch.isinf(logits).any():
                      logger.error(f"Train Batch {i}: NaN/Inf in model output logits detected! Shape: {logits.shape}")
                      # Versuche fehlerhaften Input zu isolieren? Schwierig.
                      # Überspringe diesen Batch.
                      optimizer.zero_grad(); continue

                 # Prüfe Ground Truth Indizes
                 if ground_truth.max() >= logits.shape[-1]:
                      max_idx = ground_truth.max().item()
                      vocab_size = logits.shape[-1]
                      logger.error(f"Train Batch {i}: Ground truth index ({max_idx}) out of bounds for vocabulary size ({vocab_size})!")
                      # Dies deutet auf ein Problem in Vectorizer oder Datenaufbereitung hin!
                      optimizer.zero_grad(); continue

                 loss = criterion(logits.reshape(-1, logits.shape[-1]), ground_truth.reshape(-1))

             # Prüfe auf NaN/Inf Loss
             if torch.isnan(loss) or torch.isinf(loss):
                 logger.warning(f"Train Batch {i}: NaN/Inf Loss ({loss.item():.4f}) detected. Überspringe Step.")
                 if (i + 1) % configuration.ACCUMULATION_STEPS == 0 or (i + 1) == len(dataloader): optimizer.zero_grad()
                 continue

             loss = loss / configuration.ACCUMULATION_STEPS

             if amp_enabled: scaler.scale(loss).backward()
             else: loss.backward()

             if (i + 1) % configuration.ACCUMULATION_STEPS == 0 or (i + 1) == len(dataloader):
                 if amp_enabled: scaler.unscale_(optimizer)
                 if configuration.GRAD_CLIP > 0: torch.nn.utilities.clip_grad_norm_(model.parameters(), max_norm=configuration.GRAD_CLIP)

                 if amp_enabled: scaler.step(optimizer); scaler.update()
                 else: optimizer.step()

                 scheduler.step() # Step Scheduler nach Optimizer
                 optimizer.zero_grad()

             batch_loss_value = loss.item() * configuration.ACCUMULATION_STEPS
             total_loss += batch_loss_value * B
             train_samples_processed += B
             current_avg_loss = total_loss / train_samples_processed if train_samples_processed > 0 else 0
             current_lr = scheduler.get_last_lr()[0]
             progress_bar.set_postfix({'Avg Loss': f"{current_avg_loss:.4f}", 'LR': f"{current_lr:.3e}"})

        except RuntimeError as rt_error:
            logger.error(f"Runtime Error Training Batch {i}: {rt_error}", exc_info=False)
            logger.debug(traceback.format_exc()) # Logge vollen Traceback im Debug-Modus
            if "CUDA out of memory" in str(rt_error): logger.critical("CUDA OOM! Training abgebrochen."); raise rt_error # OOM ist fatal
            else: optimizer.zero_grad(); continue # Versuche bei anderen Fehlern weiterzumachen
        except Exception as e: # Fange auch andere Fehler ab
             logger.error(f"Allgemeiner Fehler Training Batch {i}: {e}", exc_info=True)
             optimizer.zero_grad(); continue # Sicherheitshalber überspringen


    # Logge Live-Beispiele am Ende der Epoche
    if last_batch_data:
        log_training_samples(model, last_batch_data, vectorizer, device, epoch_num)

    # Stelle sicher, dass wir nicht durch 0 teilen
    if train_samples_processed == 0:
        logger.error(f"Keine Samples erfolgreich in Training Epoche {epoch_num} verarbeitet!")
        return float('inf') # Gib Inf zurück, wenn keine Samples verarbeitet wurden

    avg_epoch_loss = total_loss / train_samples_processed
    return avg_epoch_loss


@torch.no_grad()
def evaluate_one_epoch(model, dataloader, criterion, device, epoch_num, vectorizer):
    """ Evaluiert eine Epoche, berechnet Loss & Metriken (Offline Transformer). """
    if not dataloader or not hasattr(dataloader, 'dataset') or len(dataloader.dataset) == 0:
        logger.warning("Validation dataloader is empty or invalid. Skipping evaluation.")
        return {'loss': float('inf'), 'cer': 1.0, 'wer': 1.0, 'char_f1': 0.0, 'char_precision': 0.0, 'char_recall': 0.0, 'bleu': 0.0}

    model.eval()
    total_loss = 0.0
    eval_samples_processed = 0 # Zählt Samples, für die Loss berechnet wurde
    metric_samples_processed = 0 # Zählt Samples, für die Metriken berechnet wurden
    all_true_strings, all_pred_strings, all_rel_paths = [], [], []
    progress_bar = tqdm(dataloader, desc=f"Eval Ep {epoch_num}", unit="batch", leave=False)
    device_type = device.type
    # AMP nur für Loss-Berechnung aktivieren (falls konfiguriert), NICHT für Generate
    amp_enabled_loss = configuration.USE_MIXED_PRECISION and device_type == 'cuda'
    error_count = 0
    max_errors_to_log = 5

    for batch_idx, batch in enumerate(progress_bar):
        images = batch['images']
        targets = batch['targets'] # Bleibt auf CPU für Dekodierung später
        paths = batch['paths']

        if images.nelement() == 0 or targets.nelement() == 0:
            logger.warning(f"Eval Batch {batch_idx}: Leeres Image/Target Tensor nach Collate übersprungen."); continue

        images = images.to(device, non_blocking=True)
        targets_device = targets.to(device, non_blocking=True) # Kopie auf Device für Loss
        B = images.size(0)

        # --- 1. Loss Berechnung ---
        batch_loss = None
        try:
            decoder_input = targets_device[:, :-1]
            ground_truth = targets_device[:, 1:]

            # Autocast nur für Loss (falls aktiviert)
            with autocast(device_type=device_type, enabled=amp_enabled_loss):
                logits = model(src_img=images, tgt_tokens=decoder_input)
                # Prüfungen wie im Training
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                     logger.error(f"Eval Batch {batch_idx}: NaN/Inf in model output logits! Shape: {logits.shape}")
                     raise ValueError("NaN/Inf in logits during evaluation") # Fehler werfen, um Batch zu überspringen

                if ground_truth.max() >= logits.shape[-1]:
                     max_idx = ground_truth.max().item(); vocab_size = logits.shape[-1]
                     logger.error(f"Eval Batch {batch_idx}: GT index ({max_idx}) >= vocab size ({vocab_size})!")
                     raise ValueError("Ground truth index out of bounds")

                loss = criterion(logits.reshape(-1, logits.shape[-1]), ground_truth.reshape(-1))

            # Prüfe Loss
            if not torch.isnan(loss) and not torch.isinf(loss):
                batch_loss = loss.item() # Gültigen Loss speichern
                total_loss += batch_loss * B
                eval_samples_processed += B # Nur Samples zählen, für die Loss berechnet wurde
            else:
                logger.warning(f"Eval Batch {batch_idx}: NaN/Inf Loss ({loss.item()}) detected.")
                # Kein eval_samples_processed += B hier

        except Exception as loss_e: # Fange Fehler bei Loss-Berechnung ab
            error_count += 1
            if error_count <= max_errors_to_log:
                logger.error(f"Fehler bei Loss-Berechnung Eval Batch {batch_idx}: {loss_e}", exc_info=True if error_count == 1 else False)
            # Gehe trotzdem weiter zur Metrikberechnung, wenn möglich
            # Batch_loss bleibt None

        # --- 2. Vorhersage und Metrikberechnung ---
        try:
            # Generierung OHNE Autocast zur Fehlervermeidung
            pred_ids = model.generate(src_img=images, max_len=configuration.TARGET_MAXLEN, temperature=0.6, top_k=5)
            pred_ids_list = pred_ids.cpu().tolist()
            target_ids_list = targets.cpu().tolist() # Original-Targets von CPU

            # Dekodieren und Metriken sammeln
            batch_true_strs, batch_pred_strs = [], []
            for i in range(B):
                # Verwende Vectorizer.decode für Konsistenz
                true_label_str = vectorizer.decode(target_ids_list[i])
                pred_label_str = vectorizer.decode(pred_ids_list[i])
                batch_true_strs.append(true_label_str)
                batch_pred_strs.append(pred_label_str)

            # Füge zur Gesamtliste hinzu
            all_true_strings.extend(batch_true_strs)
            all_pred_strings.extend(batch_pred_strs)
            all_rel_paths.extend(paths)
            metric_samples_processed += B # Zähle Samples für Metrikberechnung

            # Update Progress Bar (optional mit CER/Loss des aktuellen Batches)
            current_avg_loss = total_loss / eval_samples_processed if eval_samples_processed > 0 else 0
            # Optional: Berechne Batch CER für Postfix
            # batch_cer = np.mean([utilities.cer(t,p) for t,p in zip(batch_true_strs, batch_pred_strs)])
            # progress_bar.set_postfix({'Avg Val Loss': f"{current_avg_loss:.4f}", 'Batch CER': f"{batch_cer:.3f}"})
            progress_bar.set_postfix({'Avg Val Loss': f"{current_avg_loss:.4f}"})

        except Exception as metric_e: # Fange Fehler bei Generierung/Metrik ab
            error_count += 1
            if error_count <= max_errors_to_log:
                logger.error(f"Fehler bei Generierung/Metrik Eval Batch {batch_idx}: {metric_e}", exc_info=True if error_count == 1 else False)
            # Breche diesen Batch für Metriken ab, Loss könnte aber gültig sein

    # --- Nach allen Batches ---
    # Prüfe, ob überhaupt Samples verarbeitet wurden
    if eval_samples_processed == 0 and metric_samples_processed == 0:
         logger.error(f"Keine Samples konnten in Eval Ep {epoch_num} verarbeitet werden (weder Loss noch Metriken).")
         return {'loss': float('inf'), 'cer': 1.0, 'wer': 1.0, 'char_f1': 0.0, 'char_precision': 0.0, 'char_recall': 0.0, 'bleu': 0.0}

    # Berechne finale durchschnittliche Metriken
    avg_loss = total_loss / eval_samples_processed if eval_samples_processed > 0 else float('inf') # WICHTIG: Verwende eval_samples_processed

    if metric_samples_processed > 0:
         metrics = utilities.compute_all_metrics(all_true_strings, all_pred_strings)
         metrics['loss'] = avg_loss # Füge den berechneten Loss hinzu
         # Logge Beispiele nur, wenn Metriken berechnet wurden
         log_validation_samples(all_true_strings, all_pred_strings, all_rel_paths, epoch_num=epoch_num)
    else:
         # Wenn keine Metriken berechnet werden konnten, aber vielleicht Loss
         logger.error(f"Eval Ep {epoch_num}: Loss berechnet ({avg_loss:.4f}), aber keine Metriken (Generierungs-/Dekodierfehler?).")
         # Gib Standard-Fehlermetriken zurück, aber mit dem berechneten Loss
         metrics = {'loss': avg_loss, 'cer': 1.0, 'wer': 1.0, 'char_f1': 0.0, 'char_precision': 0.0, 'char_recall': 0.0, 'bleu': 0.0}

    if error_count > 0:
        logger.warning(f"Eval Ep {epoch_num}: {error_count} Fehler traten während der Batch-Verarbeitung auf.")

    return metrics


# log_validation_samples bleibt konzeptionell gleich, verwendet aber vectorizer.decode
def log_validation_samples(true_strings, pred_strings, rel_paths, num_samples=5, epoch_num=None):
    """ Loggt zufällige Validierungsbeispiele. """
    if not true_strings or not pred_strings or not rel_paths or len(true_strings)==0 : return
    actual_num = min(num_samples, len(true_strings))
    indices = random.sample(range(len(true_strings)), actual_num)
    logger.info(f"--- Validierungsbeispiele Epoche {epoch_num or '?'} ({actual_num} Samples) ---")
    for i in indices:
        true_s = true_strings[i]; pred_s = pred_strings[i]; path = rel_paths[i]
        is_correct = "✅" if true_s == pred_s else "❌"
        logger.info(f"  Sample (Bild: ...{path[-40:]}) {is_correct}")
        logger.info(f"    TARGET: '{true_s}'")
        logger.info(f"    PRED  : '{pred_s}'")
    logger.info("-" * 50)


# save_training_history und plot_training_curves bleiben unverändert...
def save_training_history(history_list, filename="training_history"):
    """ Speichert History als CSV/JSON. """
    if not history_list: logger.warning("Leere History, nichts zu speichern."); return
    try:
        df = pd.DataFrame(history_list)
        # Stelle sicher, dass der Index nicht mitgeschrieben wird
        utilities.ensure_dir(configuration.RESULTS_PATH)
        csv_path = os.path.join(configuration.RESULTS_PATH, f"{filename}.csv")
        json_path = os.path.join(configuration.RESULTS_PATH, f"{filename}.json")
        df.to_csv(csv_path, index=False)
        logger.info(f"History CSV: {csv_path}")
        # Konvertiere zu serialisierbarem Format für JSON
        history_plain = utilities.convert_metrics_to_serializable(df.to_dict(orient='records'))
        with open(json_path, 'w', encoding='utf-8') as f: json.dump(history_plain, f, indent=2, ensure_ascii=False)
        logger.info(f"History JSON: {json_path}")
    except Exception as e: logger.error(f"Fehler Speichern History: {e}", exc_info=True)


def plot_training_curves(history_list):
    """ Plottet Trainingskurven und speichert sie. """
    if not history_list or not isinstance(history_list, list) or len(history_list) == 0 or 'epoch' not in history_list[0]: return
    try:
        df = pd.DataFrame(history_list)
        if df.empty or 'epoch' not in df.columns: logger.warning("History DF leer/ungültig für Plot."); return

        epochs = df['epoch']
        metrics_to_plot = set()
        # Definiere die Reihenfolge und welche Metriken grundsätzlich geplottet werden sollen
        plot_order = ['loss', 'cer', 'wer', 'char_f1', 'bleu', 'char_precision', 'char_recall', 'learning_rate']
        available_metrics = [m for m in plot_order if f'train_{m}' in df.columns or f'val_{m}' in df.columns or m == 'learning_rate']

        if not available_metrics: logger.warning("Keine passenden Metriken zum Plotten in History gefunden."); return
        num_metrics = len(available_metrics)
        n_cols = 3; n_rows = (num_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
        axes = axes.flatten()
        plot_idx = 0

        for metric in available_metrics:
            ax = axes[plot_idx]
            has_data = False
            train_col = f'train_{metric}'
            val_col = f'val_{metric}'
            lr_col = 'learning_rate'

            # Plot Train Data, falls vorhanden und nicht nur NaNs
            if train_col in df.columns and df[train_col].notna().any():
                ax.plot(epochs, df[train_col], marker='.', linestyle='-', label=f"Train {metric.upper()}")
                has_data = True
            # Plot Val Data, falls vorhanden und nicht nur NaNs
            if val_col in df.columns and df[val_col].notna().any():
                ax.plot(epochs, df[val_col], marker='.', linestyle='--', label=f"Val {metric.upper()}")
                has_data = True
            # Plot Learning Rate
            if metric == 'learning_rate' and lr_col in df.columns and df[lr_col].notna().any():
                ax.plot(epochs, df[lr_col], marker='.', linestyle=':', color='orange', label='Learning Rate')
                has_data = True
                # Eigene Y-Achse für LR, falls Werte sehr klein sind
                # ax2 = ax.twinx()
                # ax2.plot(epochs, df[lr_col], marker='.', linestyle=':', color='orange', label='Learning Rate')
                # ax2.set_ylabel('Learning Rate', color='orange')
                # ax2.tick_params(axis='y', labelcolor='orange')
                # ax2.ticklabel_format(style='sci', axis='y', scilimits=(-0,0))
                # Alternative: Gleiche Achse mit wissenschaftlicher Notation
                ax.ticklabel_format(style='sci', axis='y', scilimits=(-0,0))

            if has_data:
                title = metric.replace('_', ' ').upper()
                ax.set_title(f"{title} Verlauf"); ax.set_xlabel("Epoche"); ax.set_ylabel(title)
                ax.legend(); ax.grid(True, linestyle=':', alpha=0.6)
                if metric not in ['loss', 'learning_rate']: ax.set_ylim(bottom=0) # Metriken wie CER nicht negativ
                ax.set_ylim(top=max(1.1, df[train_col].max()*1.1 if train_col in df and df[train_col].notna().any() else 1.1, df[val_col].max()*1.1 if val_col in df and df[val_col].notna().any() else 1.1)) # Obere Grenze etwas über Max

                plot_idx += 1
            # Mache unbenutzte Achsen unsichtbar
        for i in range(plot_idx, len(axes)): axes[i].set_visible(False)

        plt.tight_layout(pad=2.0)
        utilities.ensure_dir(configuration.PLOTS_PATH)
        plot_save_path = os.path.join(configuration.PLOTS_PATH, "training_curves.png")
        plt.savefig(plot_save_path); plt.close(fig)
        logger.info(f"Trainingskurven gespeichert: {plot_save_path}")
    except Exception as e: logger.error(f"Fehler beim Plotten der Trainingskurven: {e}", exc_info=True)
    finally: plt.close('all')


# --- Haupt-Trainingsfunktion ---
def train_offline_transformer_model():
    """ Hauptfunktion für das Offline-Training. """
    training_history = []
    train_start_time = time.time()
    logger.info("="*50); logger.info("=== STARTE OFFLINE TRANSFORMER TRAINING ==="); logger.info("="*50)

    try:
        device = torch.device(configuration.DEVICE)
        logger.info(f"Verwende Gerät: {device}")
        device_type = device.type
        amp_enabled = configuration.USE_MIXED_PRECISION and device_type == 'cuda'

        # --- Vectorizer global verfügbar machen (optional, aber praktisch) ---
        logger.info("Initialisiere Vectorizer...")
        vectorizer = utilities.VectorizeChar()
        configuration.vectorizer = vectorizer # Füge zur Config hinzu für einfachen Zugriff in CollateFn etc.

        # --- Daten Setup ---
        logger.info("Erstelle/Lade Daten-Splits...")
        if not create_offline_splits(force_new=False): logger.error("Fehler Splits. Abbruch."); return
        # Lade Daten NACH Vectorizer Initialisierung
        X_train, y_train = utilities.load_split_data(configuration.TRAIN_LABELS_LINES_FILE, vectorizer)
        X_val, y_val = utilities.load_split_data(configuration.VAL_LABELS_LINES_FILE, vectorizer)
        if not X_train or not y_train: logger.error("Trainingsdaten leer nach Laden/Kodieren. Abbruch."); return

        train_dataset = OfflineHandwritingDataset(X_train, y_train, configuration.LINES_FOLDER, augment=True)
        # Verringere Batch Size für Val Loader testweise, falls OOM ein Problem war
        val_batch_size = configuration.BATCH_SIZE # Gleiche Batch Size wie Training oder * 1 ?
        if X_val and y_val:
             val_dataset = OfflineHandwritingDataset(X_val, y_val, configuration.LINES_FOLDER, augment=False)
             val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=configuration.NUM_WORKERS)
             logger.info(f"Val Dataset/Loader erstellt: {len(val_dataset)} Samples, Batch Size={val_batch_size}")
        else:
             val_dataset = None
             val_loader = None
             logger.info("Keine Validierungsdaten gefunden oder geladen.")

        train_loader = DataLoader(train_dataset, batch_size=configuration.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=configuration.NUM_WORKERS, pin_memory=True, drop_last=True)
        logger.info(f"Train Dataset/Loader erstellt: {len(train_dataset)} Samples, Batch Size={configuration.BATCH_SIZE}")


        # --- Modell, Loss, Optimizer, Scheduler Setup ---
        model = ocr_model.build_offline_transformer_model()
        if model is None: logger.error("Modell Erstellung fehlgeschlagen."); return
        model = model.to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=vectorizer.pad_token_id, label_smoothing=configuration.LABEL_SMOOTHING)
        logger.info(f"Loss: CrossEntropy (ignore={vectorizer.pad_token_id}, smooth={configuration.LABEL_SMOOTHING})")

        optimizer = optim.AdamW(model.parameters(), lr=configuration.LEARNING_RATE, weight_decay=configuration.WEIGHT_DECAY)
        logger.info(f"Optimizer: AdamW (LR={configuration.LEARNING_RATE:.2e}, WD={configuration.WEIGHT_DECAY:.2e})")

        # Scheduler Setup
        # Vorsicht bei drop_last=True im train_loader, len(train_loader) ist dann kleiner
        # steps_per_epoch = len(train_loader)
        # num_training_steps_per_epoch = math.ceil(steps_per_epoch / configuration.ACCUMULATION_STEPS)
        # Korrekter: Berechne Schritte basierend auf Dataset-Größe
        num_train_samples = len(train_dataset)
        num_training_steps_per_epoch = math.ceil(math.ceil(num_train_samples / configuration.BATCH_SIZE) / configuration.ACCUMULATION_STEPS)
        logger.info(f"Berechnete Trainingsschritte pro Epoche: {num_training_steps_per_epoch}")

        total_optim_steps = configuration.EPOCHS * num_training_steps_per_epoch
        total_warmup_steps = configuration.WARMUP_EPOCHS * num_training_steps_per_epoch
        total_decay_steps = total_optim_steps - total_warmup_steps
        scheduler = CustomSchedule(optimizer, init_lr=configuration.WARMUP_INIT_LR, peak_lr=configuration.LEARNING_RATE,
                                   final_lr=configuration.FINAL_LR, warmup_steps=total_warmup_steps,
                                   decay_steps=max(1, total_decay_steps)) # Stelle sicher, dass decay_steps >= 1 ist

        scaler = GradScaler(enabled=amp_enabled)
        logger.info(f"Mixed Precision Scaler initialisiert (enabled={amp_enabled})")


        # --- Training Loop ---
        es_metric_key = configuration.EARLY_STOPPING_METRIC # z.B. 'val_loss'
        # Stelle sicher, dass der Key für History passt ('val_' prefix)
        if not es_metric_key.startswith('val_'): es_metric_key_hist = f'val_{es_metric_key}'
        else: es_metric_key_hist = es_metric_key
        metric_mode = 'min' if 'loss' in es_metric_key or 'cer' in es_metric_key or 'wer' in es_metric_key else 'max'
        best_val_metric = float('inf') if metric_mode == 'min' else float('-inf')
        patience_counter = 0; best_epoch = -1

        logger.info("="*50); logger.info(f"=== STARTE TRAININGSLOOP (Max {configuration.EPOCHS} Epochen) ==="); logger.info("="*50)
        logger.info(f"Early Stopping: Metric='{es_metric_key_hist}', Mode='{metric_mode}', Patience={configuration.EARLY_STOPPING_PATIENCE}")

        for epoch in range(configuration.EPOCHS):
            epoch_num = epoch + 1; epoch_start_time = time.time()
            logger.info(f"--- Epoche {epoch_num}/{configuration.EPOCHS} ---")

            avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler, scaler, vectorizer, epoch_num)
            current_lr = optimizer.param_groups[0]['lr'] # Hol die LR *nach* dem Training der Epoche
            logger.info(f"Epoche {epoch_num}: Avg Train Loss = {avg_train_loss:.5f}")

            # --- Evaluation ---
            current_val_metrics = {}
            if val_loader:
                 current_val_metrics = evaluate_one_epoch(model, val_loader, criterion, device, epoch_num, vectorizer)
                 # Formatierte Ausgabe der Val-Metriken
                 val_log_str = " | ".join([f"Val {k.replace('_',' ').upper()}={v:.4f}" for k, v in current_val_metrics.items()])
                 logger.info(f"Epoche {epoch_num} Eval Result: {val_log_str}")
            else:
                 # Falls kein Val-Loader, fülle mit Dummy-Werten für History
                 current_val_metrics = {'loss': float('inf'), 'cer': 1.0, 'wer': 1.0, 'char_f1': 0.0, 'char_precision': 0.0, 'char_recall': 0.0, 'bleu': 0.0}
                 logger.info(f"Epoche {epoch_num}: Keine Validierungsdaten, Eval übersprungen.")


            # --- History & Early Stopping ---
            epoch_summary={'epoch':epoch_num,'train_loss':avg_train_loss,'learning_rate':current_lr, **{f'val_{k}':v for k,v in current_val_metrics.items()}}
            training_history.append(epoch_summary)

            early_stop_triggered = False
            if val_loader and configuration.EARLY_STOPPING_PATIENCE > 0:
                 current_metric_val = epoch_summary.get(es_metric_key_hist)

                 # Robuste Prüfung auf gültige Metrik
                 if current_metric_val is None or np.isnan(current_metric_val) or np.isinf(current_metric_val):
                     logger.warning(f"ES Metrik '{es_metric_key_hist}' ungültig (Wert={current_metric_val}) in Epoche {epoch_num}. Early Stopping übersprungen.")
                 else:
                      # Logik für Verbesserung
                      improved = (metric_mode == 'min' and current_metric_val < best_val_metric - configuration.MIN_DELTA) or \
                                 (metric_mode == 'max' and current_metric_val > best_val_metric + configuration.MIN_DELTA)
                      if improved:
                           improvement = abs(current_metric_val - best_val_metric); best_val_metric = current_metric_val; patience_counter = 0; best_epoch = epoch_num
                           logger.info(f"📈 Verbesserung '{es_metric_key_hist}': {best_val_metric:.5f} (+{improvement:.5f}). Speichere bestes Modell.")
                           best_model_path = os.path.join(configuration.CHECKPOINT_PATH, "best_offline_transformer_model.pth")
                           try: torch.save(model.state_dict(), best_model_path)
                           except Exception as save_e: logger.error(f"Fehler Speichern Bestes Modell: {save_e}")
                      else:
                           patience_counter += 1
                           logger.info(f"📉 Keine Verbesserung '{es_metric_key_hist}'. Geduld: {patience_counter}/{configuration.EARLY_STOPPING_PATIENCE} (Best={best_val_metric:.5f} @ Ep {best_epoch})")
                           if patience_counter >= configuration.EARLY_STOPPING_PATIENCE:
                                logger.warning(f"⏳ EARLY STOPPING ausgelöst nach {patience_counter} Epochen ohne Verbesserung von '{es_metric_key_hist}'.")
                                early_stop_triggered = True

            # --- Epoch End ---
            epoch_duration = time.time() - epoch_start_time; logger.info(f"Epoche {epoch_num} Dauer: {epoch_duration:.2f} Sek.")
            save_interval = configuration.SAVE_CHECKPOINT_INTERVAL
            # Speichere Checkpoint und History/Plots in Intervallen, am Ende oder bei Early Stopping
            if (save_interval > 0 and epoch_num % save_interval == 0) or (epoch_num == configuration.EPOCHS) or early_stop_triggered:
                 logger.info(f"Speichere Checkpoint & History/Plots (Epoche {epoch_num})...");
                 checkpoint_path = os.path.join(configuration.CHECKPOINT_PATH, f"checkpoint_epoch_{epoch_num}.pth")
                 try: torch.save({ 'epoch': epoch_num, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'best_val_metric': best_val_metric, 'patience_counter': patience_counter, 'training_history': training_history }, checkpoint_path)
                 except Exception as chkpt_e: logger.error(f"Fehler Speichern Checkpoint Ep {epoch_num}: {chkpt_e}")
                 save_training_history(training_history); plot_training_curves(training_history)

            if early_stop_triggered: break # Beende die Trainingsschleife

        logger.info("="*50); logger.info("=== OFFLINE TRANSFORMER TRAINING BEENDET ===");
        reason = "Early Stopping" if early_stop_triggered else f"Max Epochs ({configuration.EPOCHS})"
        logger.info(f"Grund: {reason}.");
        if best_epoch != -1: logger.info(f"Bestes Modell @ Ep {best_epoch} ({es_metric_key_hist}={best_val_metric:.5f}) gespeichert in '{configuration.CHECKPOINT_PATH}'.")
        else: logger.warning("Kein Bestes Modell gespeichert (entweder keine Verbesserung oder keine Val-Daten).")
        # Speichere finale History und Plots auf jeden Fall
        save_training_history(training_history, "training_history_final"); plot_training_curves(training_history)

    except RuntimeError as rt_oom:
        if "CUDA out of memory" in str(rt_oom): logger.critical("TRAINING ABBRUCH: CUDA Out Of Memory!", exc_info=False); save_training_history(training_history, "history_oom")
        else: logger.critical(f"Kritischer Runtime Fehler: {rt_oom}", exc_info=True); save_training_history(training_history, "history_error")
    except Exception as e:
        logger.critical(f"Kritischer Fehler im Trainingsprozess: {e}", exc_info=True)
        save_training_history(training_history, "history_error")
    finally:
        total_training_time = time.time() - train_start_time
        logger.info(f"Gesamte Trainingsdauer: {total_training_time / 60:.2f} Minuten.")
        logger.info("="*50)

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        log_level_str = getattr(config, 'LOGGING_LEVEL', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)-7s] %(name)-25s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logger = logging.getLogger(__name__)
    train_offline_transformer_model()