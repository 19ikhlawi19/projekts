# model_training.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utilities.data import DataLoader # Nur DataLoader importieren
import numpy as np
import logging
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
import time # für Timing

import configuration
import utilities
# ÄNDERUNG: Importiere die verschobenen Komponenten aus utilities
from utilities import OfflineHandwritingDataset, collate_fn, load_data_for_ctc
import ocr_model
# image_augmentation wird von utilities.OfflineHandwritingDataset intern importiert/verwendet

# -----------------------------------------------------------------------------
# model_training.py
# -----------------------------------------------------------------------------
# Training mit ausgelagerten Datenkomponenten.
# -----------------------------------------------------------------------------

def setup_logging():
    """
    Konfiguriert das Logging zentral. Sollte nur einmal aufgerufen werden.
    """
    # Prüfen, ob Handler bereits existieren, um doppeltes Logging zu vermeiden
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
         # logger.debug("Logging bereits konfiguriert. Lösche alte Handler.")
         root_logger.handlers.clear()

    # Basis Logger konfigurieren
    logging.basicConfig(level=configuration.LOGGING_LEVEL,
                        format='%(asctime)s [%(levelname)-8s] %(name)-20s: %(message)s', # Name etwas breiter
                        datefmt='%Y-%m-%d %H:%M:%S',
                        force=True) # Überschreibt evtl. vorhandene Konfigurationen

    # File Handler hinzufügen (wenn Pfad existiert)
    log_dir = configuration.LOGS_PATH
    if log_dir: # Prüft, ob der Pfad nicht leer ist
        try:
             os.makedirs(log_dir, exist_ok=True) # Stelle sicher, dass das Log-Verzeichnis existiert
             log_file = os.path.join(log_dir, f"training_{configuration.MODEL_VERSION}.log")
             # Verwende UTF-8 Encoding
             fh = logging.FileHandler(log_file, encoding='utf-8', mode='a') # 'a' für Append
             fh.setLevel(configuration.LOGGING_LEVEL)
             formatter = logging.Formatter('%(asctime)s [%(levelname)-8s] %(name)-20s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
             fh.setFormatter(formatter)
             root_logger.addHandler(fh)
             # print(f"Logging to file: {log_file}") # Debug
        except Exception as e:
             root_logger.error(f"Fehler beim Erstellen des FileHandlers für {log_dir}: {e}", exc_info=True)
             root_logger.warning(f"Logging nur zur Konsole.")
    else:
        root_logger.warning(f"Kein Log-Verzeichnis (configuration.LOGS_PATH) konfiguriert. Logging nur zur Konsole.")

    # Logging für externe Libraries ggf. reduzieren
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING) # Kann gesprächig sein

    return root_logger # Gib den konfigurierten Root-Logger zurück

# Rufe setup_logging direkt auf, um den Logger für dieses Modul zu initialisieren
logger = setup_logging()


# === DEFINITIONEN ENTFERNT, da nach utilities.py verschoben ===
# class OfflineHandwritingDataset(Dataset): ...
# def collate_fn(batch): ...
# def load_data_for_ctc(csv_file, root_folder): ...
# === DEFINITIONEN ENTFERNT ===


def create_data_splits(paths, labels):
    """
    Erstellt Train/Val/Test-Splits (gemäß config) oder lädt vorhandene.
    Stellt sicher, dass geladene Labels korrekt kodiert sind (Listen von ints).
    """
    train_file = configuration.TRAIN_LABELS_FILE
    val_file = configuration.VAL_LABELS_FILE
    test_file = configuration.TEST_LABELS_FILE

    # Überprüfen, ob alle Split-Dateien existieren
    if all(os.path.exists(f) for f in [train_file, val_file, test_file]):
        logger.info("Versuche, bestehende Train/Val/Test-Splits zu laden.")
        try:
            X_train, y_train = utilities.load_split_data(train_file)
            X_val, y_val = utilities.load_split_data(val_file)
            X_test, y_test = utilities.load_split_data(test_file)

            # Überprüfen, ob die Splits gültig sind (nicht leer und korrekte Label-Typen)
            valid_load = True
            if not X_train or not isinstance(y_train, list) or (y_train and not isinstance(y_train[0], list)):
                logger.warning(f"Trainings-Split {train_file} ist leer oder hat ungültige Labels.")
                valid_load = False
            if not X_val or not isinstance(y_val, list) or (y_val and not isinstance(y_val[0], list)):
                logger.warning(f"Validierungs-Split {val_file} ist leer oder hat ungültige Labels.")
                valid_load = False
            if not X_test or not isinstance(y_test, list) or (y_test and not isinstance(y_test[0], list)):
                logger.warning(f"Test-Split {test_file} ist leer oder hat ungültige Labels.")
                valid_load = False

            if valid_load:
                 logger.info(f"Splits erfolgreich geladen: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
                 return X_train, X_val, X_test, y_train, y_val, y_test
            else:
                 logger.warning("Mindestens ein Split konnte nicht korrekt geladen werden. Erstelle neue Splits.")

        except Exception as e:
             logger.error(f"Fehler beim Laden der Split-Dateien: {e}. Erstelle neue Splits.", exc_info=True)

    logger.info("Erstelle neue Train/Val/Test-Splits.")

    if not paths or not labels:
         logger.error("Keine Pfade oder Labels zum Erstellen von Splits vorhanden.")
         raise ValueError("Leere Daten können nicht gesplittet werden.")

    # Split-Anteile validieren und ggf. normalisieren
    train_p, val_p, test_p = configuration.TRAIN_SPLIT, configuration.VAL_SPLIT, configuration.TEST_SPLIT
    total_p = train_p + val_p + test_p
    if not np.isclose(total_p, 1.0):
        logger.warning(f"Split-Anteile ({train_p:.2f}, {val_p:.2f}, {test_p:.2f}) summieren sich nicht zu 1 (Summe={total_p:.2f}). Normalisiere...")
        if total_p <= 0:
             logger.error("Summe der Split-Anteile ist nicht positiv. Setze auf Standard 80/10/10.")
             train_p, val_p, test_p = 0.8, 0.1, 0.1
        else:
             train_p /= total_p
             val_p /= total_p
             test_p /= total_p
             logger.info(f"Normalisierte Anteile: Train={train_p:.2f}, Val={val_p:.2f}, Test={test_p:.2f}")


    # Berechne Test-Größe und relative Val-Größe
    test_size_abs = test_p
    # Val-Größe relativ zum verbleibenden Teil nach dem Test-Split
    train_val_p = train_p + val_p
    if train_val_p <= 1e-6: # Verhindere Division durch Null, wenn Train+Val fast 0 sind
         val_size_relative = 0.0
         logger.warning("Trainings- und Validierungsanteil sind fast Null. Setze relative Val-Größe auf 0.")
    else:
         val_size_relative = val_p / train_val_p

    try:
        # Stratifizierung ist schwierig mit Listen als Labels, daher None
        # Erstelle Test-Split
        X_temp, X_test, y_temp, y_test = train_test_split(
            paths, labels, test_size=test_size_abs, random_state=configuration.RANDOM_SEED, stratify=None
        )

        # Erstelle Train/Val-Split aus dem Rest (nur wenn Val-Anteil > 0)
        if val_size_relative > 0 and len(X_temp) > 1: # Mindestens 2 Samples für Split nötig
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_relative, random_state=configuration.RANDOM_SEED, stratify=None
            )
        elif len(X_temp) > 0: # Kein Val-Split, alles ist Training
             X_train, y_train = X_temp, y_temp
             X_val, y_val = [], [] # Leere Val-Menge
             logger.info("Kein Validierungs-Split erstellt (Anteil war 0 oder zu wenig Daten).")
        else: # Sollte nicht passieren, wenn paths nicht leer war
             X_train, y_train = [], []
             X_val, y_val = [], []


    except ValueError as ve:
         # Fängt z.B. ab, wenn test_size ungültig ist oder zu wenige Samples
         logger.error(f"ValueError beim Erstellen der Splits: {ve}. Überprüfen Sie die Split-Anteile und die Datenmenge.", exc_info=True)
         raise
    except Exception as e:
        logger.error(f"Allgemeiner Fehler beim Erstellen der Splits: {e}", exc_info=True)
        raise

    # Speichere die neu erstellten Splits
    try:
        utilities.save_split_data(X_train, y_train, 'train')
        utilities.save_split_data(X_val, y_val, 'val')
        utilities.save_split_data(X_test, y_test, 'test')
    except Exception as e:
        logger.error(f"Fehler beim Speichern der erstellten Split-Dateien: {e}", exc_info=True)
        # Training kann trotzdem fortgesetzt werden, aber Splits sind nicht gespeichert

    logger.info(f"Neue Splits erstellt: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(model, data_loader, device, criterion, split_name):
    """
    Bewertet das Modell auf dem angegebenen Datensatz (DataLoader).
    Verwendet utilities.compute_metrics für alle Metriken außer Loss.
    """
    model.eval() # In Evaluationsmodus setzen
    running_loss = 0.0
    all_true_labels = [] # Liste von String-Labels
    all_pred_labels = [] # Liste von String-Vorhersagen
    evaluated_samples = 0

    # Logger-Nachricht abhängig vom Kontext (Epoch-End oder Final)
    eval_context = split_name.replace('_ep',' Ep ') if '_ep' in split_name else split_name
    logger.debug(f"Starte Evaluation für {eval_context}...")

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f"Eval {eval_context[:15]}", leave=False, ncols=100, mininterval=1.0)
        batch_count = 0
        for batch_data in progress_bar:
            # Prüfe, ob collate_fn gültige Daten zurückgegeben hat
            if not batch_data or len(batch_data) != 3:
                logger.error(f"Ungültige Batch-Daten von collate_fn erhalten in Evaluation ({eval_context}). Überspringe Batch.")
                continue
            batch_images, batch_labels, label_lengths = batch_data
            if batch_images.numel() == 0: # Prüfe auf leere Tensoren
                logger.warning(f"Leerer Batch in Evaluation ({eval_context}), Batch {batch_count}. Überspringe.")
                continue

            try:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)
                label_lengths = label_lengths.to(device)
                current_batch_size = batch_images.size(0)

                # Forward Pass
                logits = model(batch_images) # Shape: (T, N, C)
                log_probs = logits.log_softmax(2)

                # Input lengths für CTC
                T = logits.size(0)
                input_lengths = torch.full(size=(current_batch_size,), fill_value=T, dtype=torch.long, device=device)

                # Loss berechnen
                # CTCLoss braucht log_probs, targets, input_lengths, target_lengths
                loss = criterion(log_probs, batch_labels, input_lengths, label_lengths)

                # Handle inf/nan loss
                if torch.isinf(loss) or torch.isnan(loss):
                    logger.warning(f"Ungültiger Loss ({loss.item()}) in Evaluation ({eval_context}), Batch {batch_count}. Überspringe.")
                    continue

                running_loss += loss.item() * current_batch_size # Korrekte Summierung für Durchschnitt
                evaluated_samples += current_batch_size

                # Decoding der Vorhersagen
                preds_np = logits.cpu().numpy()
                recognized = utilities.decode_predictions(preds_np, configuration.IDX_TO_CHAR, configuration.NUM_CLASSES)

                # Decoding der wahren Labels
                start_idx = 0
                true_strs_batch = []
                for length in label_lengths.cpu().numpy():
                    end_idx = start_idx + length
                    label_seq = batch_labels[start_idx:end_idx].cpu().numpy()
                    start_idx = end_idx
                    dec_str = utilities.decode_labels(label_seq, configuration.NUM_CLASSES)[0]
                    true_strs_batch.append(dec_str)

                all_true_labels.extend(true_strs_batch)
                all_pred_labels.extend(recognized)

                # Update Progress Bar
                current_avg_loss = running_loss / evaluated_samples if evaluated_samples > 0 else 0.0
                progress_bar.set_postfix({'Loss': f"{current_avg_loss:.4f}"})
                batch_count += 1

            except Exception as e:
                logger.error(f"Fehler während Evaluation ({eval_context}) in Batch {batch_count}: {e}", exc_info=True)
                continue # Überspringe fehlerhaften Batch

    # Metriken berechnen
    avg_loss = running_loss / evaluated_samples if evaluated_samples > 0 else float('inf')
    # Rufe die zentrale Metrikfunktion auf
    metrics = utilities.compute_metrics(all_true_labels, all_pred_labels)
    metrics['loss'] = avg_loss # Füge Loss hinzu

    # Log-Ausgabe
    log_metrics = {k: v for k, v in metrics.items() if k in configuration.METRICS_TO_COMPUTE or k=='loss'}
    metrics_str = ", ".join([f"{k.upper().replace('_','-')}={v:.4f}" for k, v in sorted(log_metrics.items())])
    logger.info(f"Eval Result ({eval_context}): {metrics_str}")

    return metrics


def save_training_history(history, base_path):
    """
    Speichert die Trainingshistorie in CSV und JSON in einem spezifischen Pfad.
    Behandelt potenzielle Längenunterschiede.
    """
    if not base_path:
        logger.error("Kein Basispfad zum Speichern der History angegeben.")
        return
    os.makedirs(base_path, exist_ok=True) # Sicherstellen, dass Pfad existiert

    try:
        final_history = history.copy() # Kopieren, um Original nicht zu ändern
        # Fülle fehlende Werte mit NaN, falls Listen ungleich lang sind
        if final_history and 'epoch' in final_history:
            epochs = final_history['epoch']
            if not epochs: # Keine Epochen geloggt
                 logger.warning("Leere History, nichts zu speichern.")
                 return

            max_len = len(epochs)
            for key, values in final_history.items():
                 current_len = len(values)
                 if current_len < max_len:
                     logger.warning(f"Fülle fehlende Werte in History für '{key}' ({current_len} vs {max_len}) mit NaN.")
                     # Fülle mit NaN auf die maximale Länge auf
                     padding = [np.nan] * (max_len - current_len)
                     final_history[key] = values + padding
                 elif current_len > max_len and key != 'epoch':
                      logger.warning(f"Mehr Werte als Epochen in History für '{key}' ({current_len} vs {max_len}). Kürze Liste.")
                      final_history[key] = values[:max_len]


        df_history = pd.DataFrame(final_history)

        # CSV speichern
        history_csv_path = os.path.join(base_path, "training_history.csv")
        df_history.to_csv(history_csv_path, index=False, encoding='utf-8')
        logger.debug(f"Trainingshistorie CSV gespeichert: {history_csv_path}")

        # JSON speichern (konvertiere NaN zu None für JSON Kompatibilität)
        history_json = df_history.replace({np.nan: None}).to_dict(orient='list')
        history_json_path = os.path.join(base_path, "training_history.json")
        with open(history_json_path, 'w', encoding='utf-8') as f:
            json.dump(history_json, f, indent=2)
        logger.debug(f"Trainingshistorie JSON gespeichert: {history_json_path}")
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Trainingshistorie in {base_path}: {e}", exc_info=True)


def plot_training_curves(history, base_path):
    """
    Plotet die Trainings- und Validierungsverluste sowie Metriken-Verläufe.
    Speichert den Plot im base_path.
    """
    if not base_path:
        logger.error("Kein Basispfad zum Speichern der Plots angegeben.")
        return
    os.makedirs(base_path, exist_ok=True)

    try:
        if not history or 'epoch' not in history or not history['epoch']:
            logger.warning("Keine Daten in der History zum Plotten.")
            return

        epochs = history['epoch']
        if not epochs: return # Keine Epochen zum Plotten

        # Sammle alle Basis-Metriknamen (z.B. 'loss', 'cer', 'char_f1')
        base_metrics = set()
        for key in history.keys():
            if '_' in key and key != 'epoch':
                 parts = key.split('_', 1)
                 if len(parts) == 2:
                      base_metrics.add(parts[1]) # Der Teil nach dem ersten '_'
        base_metrics = sorted(list(base_metrics))

        if not base_metrics:
             logger.warning("Keine Metriken (außer Epoche) in History gefunden zum Plotten.")
             return

        n_metrics = len(base_metrics)
        n_cols = 2 # Zwei Plots nebeneinander
        n_rows = (n_metrics + n_cols - 1) // n_cols

        plt.style.use('seaborn-v0_8-grid')
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
        axes = axes.flatten() # Zum einfachen Iterieren über Subplots

        plot_idx = 0
        for metric in base_metrics:
            if plot_idx >= len(axes): break # Mehr Metriken als Subplots? Sollte nicht passieren
            ax = axes[plot_idx]
            has_data_for_metric = False
            for split in ['train', 'val', 'test']:
                metric_key = f"{split}_{metric}"
                if metric_key in history and len(history[metric_key]) == len(epochs):
                    # Hole gültige Datenpunkte (nicht None oder NaN)
                    y_values = history[metric_key]
                    valid_indices = [i for i, y in enumerate(y_values) if y is not None and not np.isnan(y)]

                    if valid_indices:
                        x_plot = [epochs[i] for i in valid_indices]
                        y_plot = [y_values[i] for i in valid_indices]
                        ax.plot(x_plot, y_plot, marker='.', linestyle='-', markersize=4, linewidth=1.5,
                                label=f"{split.capitalize()}")
                        has_data_for_metric = True
                # else: logger.debug(f"Keine Daten für {metric_key} oder Längenunterschied.")

            if has_data_for_metric:
                metric_display_name = metric.upper().replace('_', '-')
                ax.set_title(f"Verlauf: {metric_display_name}")
                ax.set_xlabel("Epoche")
                ax.set_ylabel("Wert")
                ax.legend()
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                # Setze Y-Achsen-Limit sinnvoll (z.B. für Raten von 0-1 oder Verlust)
                if 'loss' not in metric.lower():
                     ax.set_ylim(bottom=0, top=max(1.05, ax.get_ylim()[1]*1.05)) # Min 0, max etwas über Daten
                else:
                     ax.set_ylim(bottom=0) # Verlust beginnt bei 0

                plot_idx += 1

        # Übrige leere Achsen entfernen
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        training_plots_path = os.path.join(base_path, "training_plots.png")
        plt.savefig(training_plots_path, dpi=150, bbox_inches='tight')
        plt.close(fig) # Wichtig: Figur schließen, um Speicher freizugeben
        logger.info(f"Trainingskurven gespeichert: {training_plots_path}")

    except Exception as e:
        logger.error(f"Fehler beim Plotten der Trainingskurven: {e}", exc_info=True)


def train_offline_model():
    """
    Trainiert das CRNN-Modell, validiert und testet nach jeder Epoche.
    Speichert das beste Modell und die History.
    """
    start_time = time.time()
    try:
        device = torch.device(configuration.DEVICE)
        logger.info(f"Verwende Gerät: {device}")
        logger.info(f"Starte Training mit Modellversion: {configuration.MODEL_VERSION}")
        # Pfade werden jetzt in setup_logging und beim Speichern verwendet
        logger.info(f"Logs werden in '{configuration.LOGS_PATH}' gespeichert (falls konfiguriert).")
        logger.info(f"Checkpoints -> '{configuration.CHECKPOINT_PATH}'")
        logger.info(f"Results/Plots -> '{configuration.RESULTS_PATH}'")
        logger.info(f"Metrics -> '{configuration.METRICS_PATH}'")

        # Stelle sicher, dass wichtige Ordner existieren
        utilities.create_directory(configuration.CHECKPOINT_PATH)
        utilities.create_directory(configuration.RESULTS_PATH)
        utilities.create_directory(configuration.METRICS_PATH)


        model = ocr_model.build_crnn_model().to(device)
        if model is None:
             logger.critical("Modell konnte nicht erstellt werden. Training abgebrochen.")
             return
        logger.info("CRNN-Modell initialisiert und auf Gerät verschoben.")
        # logger.debug(model) # Optional: Modellarchitektur ausgeben

        # Verlustfunktion
        criterion = nn.CTCLoss(blank=configuration.CHAR_TO_IDX['blank'], reduction='mean', zero_infinity=True)
        logger.info("CTC-Loss-Funktion initialisiert (reduction='mean', zero_infinity=True).")

        # Optimierer
        optimizer_name = configuration.OPTIMIZER.lower()
        lr = configuration.LEARNING_RATE
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
        else:
            logger.warning(f"Unbekannter Optimierer '{configuration.OPTIMIZER}'. Verwende Adam.")
            optimizer = optim.Adam(model.parameters(), lr=lr)
            optimizer_name = 'adam'
        logger.info(f"{optimizer_name.capitalize()}-Optimierer verwendet mit LR={lr}.")

        # Lernraten-Scheduler
        scheduler = None
        scheduler_name = configuration.SCHEDULER.lower()
        if scheduler_name == 'reducelronplateau':
            # Monitor 'val_loss', minimiere Wert. Patience=5 (mehr Geduld)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
            logger.info("Scheduler: ReduceLROnPlateau(val_loss, factor=0.1, patience=5)")
        elif scheduler_name != 'none':
            logger.warning(f"Unbekannter Scheduler '{configuration.SCHEDULER}'. Keiner verwendet.")
        else:
            logger.info("Kein Lernraten-Scheduler verwendet.")

        # Daten laden und splitten
        logger.info("Lade Daten und erstelle/lade Splits...")
        full_paths, full_labels = utilities.load_data_for_ctc(configuration.WORDS_CSV, configuration.WORDS_FOLDER)
        if not full_paths:
             logger.error(f"Keine gültigen Daten konnten aus {configuration.WORDS_CSV} geladen werden. Training abgebrochen.")
             return
        X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(full_paths, full_labels)

        # Prüfen, ob Splits leer sind
        if not X_train:
             logger.error("Trainings-Split ist leer. Training nicht möglich.")
             return
        if not X_val:
             logger.warning("Validierungs-Split ist leer. Early Stopping und Scheduler funktionieren ggf. nicht wie erwartet.")
        if not X_test:
             logger.warning("Test-Split ist leer. Evaluation am Ende wird keine Test-Metriken liefern.")

        # Datasets erstellen
        logger.info("Erstelle PyTorch Datasets...")
        try:
            train_dataset = OfflineHandwritingDataset(X_train, y_train, augment=True)
            val_dataset = OfflineHandwritingDataset(X_val, y_val, augment=False)
            test_dataset = OfflineHandwritingDataset(X_test, y_test, augment=False)
        except ValueError as ve:
             logger.error(f"Fehler bei Dataset-Erstellung: {ve}. Abbruch.")
             return

        # DataLoader erstellen
        logger.info("Erstelle PyTorch DataLoaders...")
        # num_workers > 0 kann unter Windows/macOS Probleme machen, wenn nicht im if __name__ == "__main__": Block
        # Setze auf 0 für höhere Kompatibilität, kann bei Bedarf erhöht werden.
        num_workers = 0 # War 2 if cuda else 0
        pin_memory = (configuration.DEVICE == "cuda") # Nur sinnvoll auf GPU

        train_loader = DataLoader(
            train_dataset, batch_size=configuration.BATCH_SIZE, shuffle=True,
            collate_fn=utilities.collate_fn, num_workers=num_workers, pin_memory=pin_memory,
            drop_last=False # Letzten unvollständigen Batch nicht verwerfen
        )
        # Val und Test Loader nur erstellen, wenn Daten vorhanden sind
        val_loader = DataLoader(
            val_dataset, batch_size=configuration.BATCH_SIZE, shuffle=False,
            collate_fn=utilities.collate_fn, num_workers=num_workers, pin_memory=pin_memory
        ) if X_val else None
        test_loader = DataLoader(
            test_dataset, batch_size=configuration.BATCH_SIZE, shuffle=False,
            collate_fn=utilities.collate_fn, num_workers=num_workers, pin_memory=pin_memory
        ) if X_test else None

        logger.info(f"DataLoaders: Batch={configuration.BATCH_SIZE}, Workers={num_workers}, PinMem={pin_memory}")
        logger.info(f"  Batches: Train={len(train_loader)}, Val={len(val_loader) if val_loader else 0}, Test={len(test_loader) if test_loader else 0}")

        # Training-Vorbereitung
        best_val_metric = float('inf') # Wir überwachen val_loss
        metric_to_monitor = 'val_loss'
        patience_counter = 0
        history = {"epoch": []}
        # Initialisiere History für alle erwarteten Metriken und Splits
        for split in ['train', 'val', 'test']:
            history[f"{split}_loss"] = []
            for metric_base_name in configuration.METRICS_TO_COMPUTE:
                 if metric_base_name != 'loss':
                     history[f"{split}_{metric_base_name}"] = []


        # --- Trainings-Loop ---
        logger.info(f"===== Starte Training für {configuration.EPOCHS} Epochen =====")
        final_epoch = configuration.EPOCHS # Wird ggf. durch Early Stopping reduziert
        for epoch in range(configuration.EPOCHS):
            epoch_start_time = time.time()
            logger.info(f"--- Epoche {epoch+1}/{configuration.EPOCHS} ---")
            model.train() # Modell in Trainingsmodus setzen
            running_train_loss = 0.0
            train_samples = 0

            # Trainings-Batch-Loop
            progress_bar = tqdm(train_loader, desc=f"Ep {epoch+1} Train", leave=False, ncols=100, mininterval=0.5)
            batch_idx = -1 # Für Fehlerlogging
            for batch_idx, batch_data in enumerate(progress_bar):
                # Prüfe, ob collate_fn gültige Daten zurückgegeben hat
                if not batch_data or len(batch_data) != 3:
                    logger.error(f"Ungültige Batch-Daten von collate_fn erhalten in Training Ep {epoch+1}. Überspringe Batch.")
                    continue
                batch_images, batch_labels, label_lengths = batch_data
                if batch_images.numel() == 0:
                    logger.warning(f"Leerer Batch im Training, Ep {epoch+1}, Batch {batch_idx}. Überspringe.")
                    continue

                try:
                    batch_images = batch_images.to(device)
                    batch_labels = batch_labels.to(device)
                    label_lengths = label_lengths.to(device)
                    current_batch_size = batch_images.size(0)

                    # Forward pass
                    logits = model(batch_images) # Shape: (T, N, C)
                    log_probs = logits.log_softmax(2)

                    # Input lengths für CTC
                    T = logits.size(0)
                    input_lengths = torch.full(size=(current_batch_size,), fill_value=T, dtype=torch.long, device=device)

                    # Loss berechnen
                    loss = criterion(log_probs, batch_labels, input_lengths, label_lengths)

                    # Prüfe auf ungültigen Loss
                    if torch.isinf(loss) or torch.isnan(loss):
                        logger.error(f"Ungültiger Loss ({loss.item()}) in Training Ep {epoch+1}, Batch {batch_idx}. Überspringe Optimierungsschritt.")
                        optimizer.zero_grad() # Wichtig: Gradienten zurücksetzen
                        continue

                    # Backward pass und Optimierung
                    optimizer.zero_grad()
                    loss.backward()
                    # Optional: Gradient Clipping
                    # torch.nn.utilities.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    # Update Loss-Tracking
                    running_train_loss += loss.item() * current_batch_size
                    train_samples += current_batch_size

                    # Aktualisieren des Fortschrittsbalkens
                    if train_samples > 0:
                        current_avg_loss = running_train_loss / train_samples
                        progress_bar.set_postfix({'Loss': f"{current_avg_loss:.4f}"})

                except RuntimeError as e:
                     if "CUDA out of memory" in str(e):
                         logger.critical(f"CUDA out of memory in Training Ep {epoch+1}, Batch {batch_idx}! Versuchen Sie, die Batch-Größe zu reduzieren.", exc_info=True)
                         raise e # Training abbrechen
                     else:
                         logger.error(f"RuntimeError in Training Ep {epoch+1}, Batch {batch_idx}: {e}", exc_info=True)
                         # Training fortsetzen oder abbrechen? Wir überspringen den Batch.
                         optimizer.zero_grad() # Sicherstellen, dass keine alten Gradienten bleiben
                         continue
                except Exception as e:
                    logger.error(f"Allg. Fehler in Training Ep {epoch+1}, Batch {batch_idx}: {e}", exc_info=True)
                    continue # Batch überspringen

            # --- Ende des Trainings-Loops für die Epoche ---
            avg_train_loss = running_train_loss / train_samples if train_samples > 0 else float('nan')
            logger.info(f"Ep {epoch+1}: Avg Train Loss = {avg_train_loss:.4f}")
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(avg_train_loss)
            # Fülle andere Train-Metriken mit NaN für diesen Schritt
            for metric_base_name in configuration.METRICS_TO_COMPUTE:
                if metric_base_name != 'loss':
                    history[f"train_{metric_base_name}"].append(np.nan)

            # --- Evaluation nach jeder Epoche ---
            # Val Metrics
            if val_loader:
                 val_metrics = evaluate_model(model, val_loader, device, criterion, f'val_ep{epoch+1}')
                 # Füge Val-Metriken zur History hinzu
                 for metric_name, value in val_metrics.items():
                     history[f"val_{metric_name}"].append(value)
                 current_val_metric_for_stopping = val_metrics.get('loss', float('inf')) # Nimm Loss für Early Stopping
            else: # Kein Val-Loader
                 logger.warning("Kein Validierungs-Loader - Überspringe Val-Evaluation.")
                 val_metrics = {} # Leeres Dict
                 current_val_metric_for_stopping = float('inf') # Kann nicht verbessern
                 # Fülle Val-History mit NaN
                 history["val_loss"].append(np.nan)
                 for metric_base_name in configuration.METRICS_TO_COMPUTE:
                      if metric_base_name != 'loss': history[f"val_{metric_base_name}"].append(np.nan)

            # Test Metrics
            if test_loader:
                 test_metrics = evaluate_model(model, test_loader, device, criterion, f'test_ep{epoch+1}')
                 # Füge Test-Metriken zur History hinzu
                 for metric_name, value in test_metrics.items():
                      history[f"test_{metric_name}"].append(value)
            else: # Kein Test-Loader
                 logger.info("Kein Test-Loader - Überspringe Test-Evaluation während Training.")
                 test_metrics = {}
                 # Fülle Test-History mit NaN
                 history["test_loss"].append(np.nan)
                 for metric_base_name in configuration.METRICS_TO_COMPUTE:
                      if metric_base_name != 'loss': history[f"test_{metric_base_name}"].append(np.nan)


            # Logge die Ergebnisse der Epoche zusammenfassend
            val_log_str = "N/A"
            if val_metrics:
                log_val = {k: v for k, v in val_metrics.items() if k in ['loss', 'cer', 'wer', 'char_f1']}
                val_log_str = ", ".join([f"{k.upper()}={v:.4f}" for k,v in log_val.items()])

            test_log_str = "N/A"
            if test_metrics:
                 log_test = {k: v for k, v in test_metrics.items() if k in ['loss', 'cer', 'wer', 'char_f1']}
                 test_log_str = ", ".join([f"{k.upper()}={v:.4f}" for k,v in log_test.items()])

            logger.info(f"Ep {epoch+1} Summary | Val: [{val_log_str}] | Test: [{test_log_str}]")


            # Lernraten-Scheduler Schritt (nur wenn Val-Loss verfügbar)
            if scheduler and val_loader and not np.isnan(current_val_metric_for_stopping):
                scheduler.step(current_val_metric_for_stopping)

            # Bestes Modell speichern & Early Stopping (nur wenn Val-Daten vorhanden)
            if val_loader:
                 if current_val_metric_for_stopping < best_val_metric:
                     best_val_metric = current_val_metric_for_stopping
                     patience_counter = 0
                     save_path = os.path.join(configuration.CHECKPOINT_PATH, "best_crnn_model.pth")
                     try:
                         torch.save(model.state_dict(), save_path)
                         logger.info(f"Ep {epoch+1}: New best model saved (Val Loss: {best_val_metric:.4f})")
                     except Exception as e:
                         logger.error(f"Fehler beim Speichern des Modells in Ep {epoch+1}: {e}", exc_info=True)
                 else:
                     patience_counter += 1
                     logger.info(f"Ep {epoch+1}: Val loss did not improve ({current_val_metric_for_stopping:.4f} vs best {best_val_metric:.4f}). Patience: {patience_counter}/{configuration.EARLY_STOPPING_PATIENCE}")
                     if patience_counter >= configuration.EARLY_STOPPING_PATIENCE:
                         logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                         final_epoch = epoch + 1 # Letzte abgeschlossene Epoche
                         break # Trainings-Loop beenden
            else:
                 # Wenn kein Val-Set, speichere das Modell jeder Epoche oder nur das letzte
                 save_path = os.path.join(configuration.CHECKPOINT_PATH, f"model_epoch_{epoch+1}.pth")
                 # torch.save(model.state_dict(), save_path) # Optional: Jede Epoche speichern
                 # Speichere stattdessen nur das letzte Modell als "bestes"
                 if epoch == configuration.EPOCHS - 1:
                      last_save_path = os.path.join(configuration.CHECKPOINT_PATH, "best_crnn_model.pth") # Überschreibe "best"
                      torch.save(model.state_dict(), last_save_path)
                      logger.info(f"Ep {epoch+1}: Kein Val-Set, speichere letztes Modell als 'best'.")


            # Speichere Trainingshistorie und Plots nach jeder Epoche
            save_training_history(history, configuration.RESULTS_PATH)
            # Plotte nicht nach jeder Epoche, sondern seltener, um I/O zu reduzieren
            if (epoch + 1) % 5 == 0 or epoch == configuration.EPOCHS - 1 or patience_counter >= configuration.EARLY_STOPPING_PATIENCE:
                plot_training_curves(history, configuration.RESULTS_PATH)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            logger.info(f"Epoche {epoch+1} abgeschlossen in {epoch_duration:.2f} Sekunden.")


        # --- Ende des gesamten Trainings ---
        total_training_time = time.time() - start_time
        logger.info(f"===== Training beendet nach {final_epoch} Epochen. =====")
        logger.info(f"Gesamte Trainingszeit: {total_training_time:.2f} Sekunden.")

        # Lade das *beste* gespeicherte Modell für die finale Evaluation
        best_model_path = os.path.join(configuration.CHECKPOINT_PATH, "best_crnn_model.pth")
        if os.path.exists(best_model_path):
            logger.info(f"Lade bestes Modell von {best_model_path} für finale Evaluation...")
            try:
                # Erstelle ein neues Modell oder lade in das existierende
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                model.eval()

                logger.info("Führe finale Evaluation mit bestem Modell durch...")
                # Verwende die vollen Loader für die finale Evaluation
                final_train_metrics = evaluate_model(model, train_loader, device, criterion, 'train_final') if train_loader else {}
                final_val_metrics = evaluate_model(model, val_loader, device, criterion, 'val_final') if val_loader else {}
                final_test_metrics = evaluate_model(model, test_loader, device, criterion, 'test_final') if test_loader else {}

                # Speichere finale Metriken in einer zusammenfassenden Datei
                final_metrics_summary = {
                    'train': final_train_metrics,
                    'val': final_val_metrics,
                    'test': final_test_metrics,
                    'info': {
                         'best_model_path': best_model_path,
                         'training_epochs_completed': final_epoch,
                         'total_training_time_sec': round(total_training_time, 2)
                    }
                }
                # Speichern der finalen Metriken (verwende utilities.save_metrics_to_file)
                utilities.save_metrics_to_file(final_metrics_summary, "final_summary", configuration.METRICS_PATH)

            except Exception as e:
                 logger.error(f"Fehler bei der finalen Evaluation: {e}", exc_info=True)
        else:
            logger.warning("Bestes Modell nicht gefunden unter {best_model_path}. Finale Evaluation übersprungen.")

    except Exception as e:
        logger.critical(f"Schwerwiegender Fehler während des Trainingsprozesses: {e}", exc_info=True)
        # Optional: raise e # Fehler weitergeben, um Skript ggf. zu beenden

if __name__ == "__main__":
    train_offline_model()