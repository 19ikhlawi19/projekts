# evaluation_offline.py
import os
import logging
import torch
import torch.nn as nn
from torch.utilities.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import argparse
import time
import base64 # For HTML embedding

# Lokale Imports
import configuration
import utilities
import ocr_model # Offline Modell Importieren
from training_offline import OfflineHandwritingDataset, collate_fn # Reuse Dataset/Collate

# Optional OpenCV Import (aus alter Vorlage)
try: import cv2
except ImportError: cv2 = None; logger.warning("OpenCV (cv2) nicht gefunden. HTML-Report mit Bildern nicht möglich.")

logger = logging.getLogger(__name__)

# --- Plotting Funktionen (aus Online eval + spezifische für Offline) ---
def save_sample_predictions_html(relative_paths, true_labels, pred_labels, image_root, split_name, num_samples=50):
    """ Speichert HTML mit Bildbeispielen (Base64 eingebettet). """
    if not cv2 or not base64: logger.error("cv2/base64 fehlt für HTML-Report."); return
    utilities.ensure_dir(configuration.RESULTS_PATH)
    html_path = os.path.join(configuration.RESULTS_PATH, f"{split_name}_sample_predictions_eval.html")
    count = min(num_samples, len(relative_paths))
    if count == 0: return
    indices = np.random.choice(len(relative_paths), count, replace=False)

    html = f"<html><head><title>Sample Predictions ({split_name})</title><style>..." # Style wie in alter Vorlage
    html += f"</style></head><body><h1>Sample Predictions ({split_name.capitalize()} Split) - {count} Samples</h1>"

    embedded_count = 0
    for i in indices:
        rel_path = relative_paths[i]; true_lbl = true_labels[i]; pred_lbl = pred_labels[i]
        abs_path = os.path.join(image_root, rel_path).replace("\\", "/")
        css_class = "correct" if true_lbl == pred_lbl else "incorrect"; indicator = "✅" if true_lbl == pred_lbl else "❌"

        html += f'<div class="sample">'
        img_html = f'<span>(Bild nicht ladbar: {rel_path})</span>'
        try:
             img = cv2.imread(abs_path)
             if img is not None:
                 max_h = 100; h, w = img.shape[:2]; scale = max_h / h if h > max_h else 1.0
                 res_img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                 is_succ, buffer = cv2.imencode(".png", res_img)
                 if is_succ: data_uri = base64.b64encode(buffer).decode("utf-8"); img_html = f'<img src="data:image/png;base64,{data_uri}"/>'; embedded_count += 1
                 else: logger.warning(f"Encode failed: {abs_path}")
             else: logger.warning(f"Load failed: {abs_path}")
        except Exception as img_e: logger.error(f"HTML Img Error ({abs_path}): {img_e}", exc_info=False)
        html += img_html

        html += '<div class="text-content">'
        html += f'<strong>Pfad:</strong> {rel_path}<br/>'
        html += f'<strong class="label">Wahr:</strong> <span class="{css_class}">{true_lbl or "[LEER]"}</span><br/>'
        html += f'<strong class="label">Vorherges.:</strong> <span class="{css_class}">{pred_lbl or "[LEER]"} {indicator}</span>'
        html += '</div></div>\n'

    html += f"<p>{embedded_count} von {count} Bildern eingebettet.</p></body></html>"
    try:
        with open(html_path, 'w', encoding='utf-8') as f: f.write(html)
        logger.info(f"Sample Predictions HTML gespeichert: {html_path}")
    except Exception as e: logger.error(f"Fehler Speichern HTML: {e}")

def plot_metric_distributions(metrics_df, split_name):
    """ Plottet Histogramme für CER, WER. """
    if metrics_df.empty or not all(c in metrics_df.columns for c in ['CER', 'WER']): return
    utilities.ensure_dir(configuration.PLOTS_PATH)
    plot_path = os.path.join(configuration.PLOTS_PATH, f"{split_name}_eval_metric_distributions.png")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(metrics_df['CER'], bins=30, kde=True, ax=axes[0], color='skyblue').set_title(f'{split_name} CER Distribution')
        sns.histplot(metrics_df['WER'], bins=30, kde=True, ax=axes[1], color='salmon').set_title(f'{split_name} WER Distribution')
        axes[0].set_xlim(left=0); axes[1].set_xlim(left=0)
        plt.tight_layout(); plt.savefig(plot_path); plt.close(fig)
        logger.info(f"Metrikverteilungs-Plot ({split_name}) gespeichert: {plot_path}")
    except Exception as e: logger.error(f"Fehler Plot Metrikverteilung {split_name}: {e}", exc_info=True)
    finally: plt.close('all')

def plot_length_analysis(metrics_df, split_name):
    """ Plottet CER/WER vs. Label-Länge. """
    if metrics_df.empty or not all(c in metrics_df.columns for c in ['True Label', 'CER']): return
    utilities.ensure_dir(configuration.PLOTS_PATH)
    plot_path = os.path.join(configuration.PLOTS_PATH, f"{split_name}_eval_length_analysis.png")
    try:
        metrics_df['True Length'] = metrics_df['True Label'].apply(len)
        # Gruppe nach Länge und berechne Mittelwert (nur wenn >= 5 Samples pro Länge)
        grouped = metrics_df.groupby('True Length')[['CER', 'WER']].agg(['mean', 'count'])
        grouped.columns = ['CER_mean', 'CER_count', 'WER_mean', 'WER_count'] # Flache Spaltennamen
        grouped = grouped[grouped['CER_count'] >= 5].reset_index()

        if grouped.empty: logger.warning(f"Nicht genug Daten für Längenanalyse ({split_name})"); return

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        sns.lineplot(x='True Length', y='CER_mean', data=grouped, ax=axes[0], marker='o').set_title(f'{split_name} CER vs Label Length')
        sns.lineplot(x='True Length', y='WER_mean', data=grouped, ax=axes[1], marker='o', color='salmon').set_title(f'{split_name} WER vs Label Length')
        axes[0].set_ylabel("Avg CER"); axes[1].set_ylabel("Avg WER")
        axes[0].set_ylim(bottom=0); axes[1].set_ylim(bottom=0)
        axes[0].grid(True, linestyle=':'); axes[1].grid(True, linestyle=':')
        plt.tight_layout(); plt.savefig(plot_path); plt.close(fig)
        logger.info(f"Längenanalyse-Plot ({split_name}) gespeichert: {plot_path}")
    except Exception as e: logger.error(f"Fehler Plot Längenanalyse {split_name}: {e}", exc_info=True)
    finally: plt.close('all')

# --- Evaluations-Kernfunktion ---
@torch.no_grad()
def run_evaluation_on_split(model, dataloader, device, vectorizer, split_name):
    """ Führt Evaluation für einen Split durch (Offline Transformer). """
    if not dataloader: logger.warning(f"Leerer DataLoader für {split_name}. Eval übersprungen."); return None, None
    model.eval()
    all_preds, all_trues, all_paths = [], [], []
    results_list = [] # Für detaillierte Metriken pro Sample
    progress_bar = tqdm(dataloader, desc=f"Evaluating {split_name}", unit="batch", leave=False)

    special_ids = (vectorizer.start_token_id, vectorizer.end_token_id, vectorizer.pad_token_id, vectorizer.unk_token_id)
    idx2char = vectorizer.get_idx_to_char_map()

    for batch in progress_bar:
        images = batch['images'].to(device, non_blocking=True)
        targets = batch['targets'] # Auf CPU behalten für Dekodierung
        paths = batch['paths']
        if images.nelement() == 0: continue
        B = images.size(0)

        try:
            pred_ids = model.generate(src_img=images, max_len=configuration.TARGET_MAXLEN)
            pred_ids_list = pred_ids.cpu().tolist()
            target_ids_list = targets.cpu().tolist()

            for i in range(B):
                 true_label_str = "".join([idx2char.get(idx, '') for idx in target_ids_list[i] if idx not in special_ids])
                 pred_label_str = utilities.decode_prediction(pred_ids_list[i], idx2char, special_ids)
                 all_trues.append(true_label_str)
                 all_preds.append(pred_label_str)
                 # Metriken pro Sample berechnen
                 curr_cer = utilities.cer(true_label_str, pred_label_str)
                 curr_wer = utilities.wer(true_label_str, pred_label_str)
                 curr_f1 = utilities.compute_f1(true_label_str, pred_label_str, level='char')
                 curr_bleu = utilities.compute_bleu(true_label_str, pred_label_str)
                 curr_prec, curr_rec, _ = utilities.compute_precision_recall_f1(true_label_str, pred_label_str, level='char')
                 results_list.append({
                      'Relative Path': paths[i],
                      'True Label': true_label_str,
                      'Predicted Label': pred_label_str,
                      'CER': curr_cer, 'WER': curr_wer, 'Char F1': curr_f1,
                      'BLEU': curr_bleu, 'Char Precision': curr_prec, 'Char Recall': curr_rec
                 })
            all_paths.extend(paths)
            # Update Fortschrittsbalken optional mit laufenden Metriken (kann verlangsamen)
            # if results_list: avg_cer_run = np.mean([r['CER'] for r in results_list]); progress_bar.set_postfix({'Avg CER': f'{avg_cer_run:.4f}'})

        except RuntimeError as rt_error:
            logger.error(f"Runtime Error Eval Batch ({split_name}): {rt_error}", exc_info=True)
            # Füge Platzhalter hinzu oder überspringe Batch? Überspringen sicherer.
            continue

    # --- Nach allen Batches ---
    if not results_list: logger.error(f"Keine Ergebnisse gesammelt für {split_name}."); return None, None

    results_df = pd.DataFrame(results_list)
    # Berechne Gesamtmetriken über den DataFrame (sicherer als laufende Mittelung)
    overall_metrics = {
        'cer': results_df['CER'].mean(),
        'wer': results_df['WER'].mean(),
        'char_f1': results_df['Char F1'].mean(),
        'bleu': results_df['BLEU'].mean(),
        'char_precision': results_df['Char Precision'].mean(),
        'char_recall': results_df['Char Recall'].mean(),
        'num_samples': len(results_df)
    }

    metrics_log_str = ", ".join([f"{k.upper()}={v:.4f}" for k, v in overall_metrics.items() if k != 'num_samples'])
    logger.info(f"Gesamtmetriken ({split_name}): {metrics_log_str} ({overall_metrics['num_samples']} Samples)")

    # Detaillierte Ergebnisse speichern
    utilities.ensure_dir(configuration.RESULTS_PATH)
    results_csv_path = os.path.join(configuration.RESULTS_PATH, f"{split_name}_evaluation_details.csv")
    try: results_df.to_csv(results_csv_path, index=False); logger.info(f"Detail-Ergebnisse ({split_name}): {results_csv_path}")
    except Exception as e: logger.error(f"Fehler Speichern Detail-CSV ({split_name}): {e}", exc_info=True)

    # Gesamtmetriken speichern
    metrics_json_path = os.path.join(configuration.METRICS_PATH, f"{split_name}_evaluation_summary.json")
    try:
        serializable_metrics = utilities.convert_metrics_to_serializable(overall_metrics)
        with open(metrics_json_path, 'w') as f: json.dump(serializable_metrics, f, indent=4)
        logger.info(f"Metrik-Zusammenfassung ({split_name}): {metrics_json_path}")
    except Exception as e: logger.error(f"Fehler Speichern Metrik-JSON ({split_name}): {e}", exc_info=True)

    # Plots und HTML generieren
    plot_metric_distributions(results_df, split_name)
    plot_length_analysis(results_df, split_name)
    save_sample_predictions_html(results_df['Relative Path'].tolist(),
                                 results_df['True Label'].tolist(),
                                 results_df['Predicted Label'].tolist(),
                                 configuration.LINES_FOLDER, split_name)

    return overall_metrics, results_df # Gib Metriken-Dict und Detail-DataFrame zurück


def evaluate_offline_transformer(model_path_to_evaluate=None):
    """ Haupt-Evaluationsfunktion für das Offline-Transformer Modell. """
    eval_start_time = time.time()
    logger.info("="*50); logger.info("=== STARTE OFFLINE TRANSFORMER EVALUATION ==="); logger.info("="*50)

    try:
        device = torch.device(configuration.DEVICE)
        logger.info(f"Verwende Gerät: {device}")

        # --- Modellpfad bestimmen ---
        if model_path_to_evaluate and os.path.exists(model_path_to_evaluate):
            model_path = model_path_to_evaluate
            logger.info(f"Evaluiere spezifiziertes Modell: {model_path}")
        else:
            default_model_path = os.path.join(configuration.CHECKPOINT_PATH, "best_offline_transformer_model.pth") # Angepasster Name
            if os.path.exists(default_model_path): model_path = default_model_path; logger.info(f"Evaluiere bestes Modell: {model_path}")
            else: logger.error(f"Kein Modell zum Evaluieren gefunden (weder spezifiziert noch Standard: {default_model_path})."); return

        # --- Lade Modell ---
        logger.info(f"Lade Modell von: {model_path}")
        # WICHTIG: Modell mit korrekten Parametern aus config instanziieren
        model = ocr_model.build_offline_transformer_model()
        if model is None: logger.error("Modellerstellung für Eval fehlgeschlagen."); return
        try: model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e: logger.error(f"Fehler Laden State Dict von {model_path}: {e}", exc_info=True); return
        model.to(device); model.eval()
        logger.info("Modell geladen und in Eval-Modus.")

        # --- Daten laden ---
        vectorizer = utilities.VectorizeChar() # Für Dekodierung benötigt
        all_final_metrics = {}
        all_results_dfs = {}

        for split_name in ['train', 'val', 'test']:
            logger.info(f"\n--- Evaluiere Split: {split_name.upper()} ---")
            split_file_map = {'train': configuration.TRAIN_LABELS_LINES_FILE,
                              'val': configuration.VAL_LABELS_LINES_FILE,
                              'test': configuration.TEST_LABELS_LINES_FILE}
            split_file = split_file_map[split_name]

            if not os.path.exists(split_file): logger.warning(f"Split-Datei {split_file} nicht gefunden. Überspringe {split_name}."); continue

            X_split, y_split = utilities.load_split_data(split_file, vectorizer)
            if not X_split: logger.warning(f"Keine validen Daten im Split {split_name}. Überspringe."); continue

            split_dataset = OfflineHandwritingDataset(X_split, y_split, configuration.LINES_FOLDER, augment=False)
            split_loader = DataLoader(split_dataset, batch_size=configuration.BATCH_SIZE * 2, # Größere Batchsize für Eval
                                       shuffle=False, collate_fn=collate_fn, num_workers=configuration.NUM_WORKERS)

            # Evaluation durchführen
            split_metrics, split_results_df = run_evaluation_on_split(model, split_loader, device, vectorizer, split_name)
            if split_metrics: all_final_metrics[split_name] = split_metrics
            if split_results_df is not None: all_results_dfs[split_name] = split_results_df # Speichere auch None ggf.


        # --- Gesamtzusammenfassung ---
        logger.info("\n--- Gesamte Evaluationsübersicht ---")
        if all_final_metrics:
            # Speichere alle Metriken in einer Datei
            metrics_summary_path = os.path.join(configuration.METRICS_PATH, "evaluation_summary_all_splits.json")
            try:
                 serializable_summary = utilities.convert_metrics_to_serializable(all_final_metrics)
                 with open(metrics_summary_path, 'w') as f: json.dump(serializable_summary, f, indent=4)
                 logger.info(f"Gesamtübersicht Metriken: {metrics_summary_path}")
                 # Logge Test-Ergebnis nochmal explizit, falls vorhanden
                 if 'test' in serializable_summary:
                     test_metrics = serializable_summary['test']
                     test_log = ", ".join([f"{k.upper()}={v:.4f}" for k, v in test_metrics.items() if k != 'num_samples'])
                     logger.info(f"  TEST Split Performance: {test_log}")
                 else: logger.warning("Keine Test-Split Ergebnisse verfügbar.")
            except Exception as e: logger.error(f"Fehler Speichern Gesamt-Metriken: {e}")

             # Vergleichsplot über Splits
             # utilities.plot_metrics_comparison(all_final_metrics) # Funktion existiert (noch) nicht in dieser utilities.py
        else:
            logger.warning("Keine Metriken erfolgreich für irgendeinen Split berechnet.")

        logger.info(f"Ausgabeordner für diese Evaluation: {configuration.RUN_OUTPUT_PATH}")

    except Exception as e: logger.exception(f"Kritischer Fehler während Evaluation: {e}", exc_info=True)
    finally: logger.info(f"Gesamte Evaluationsdauer: {(time.time() - eval_start_time):.2f} Sek."); logger.info("="*50)


if __name__ == "__main__":
    parser_eval = argparse.ArgumentParser(description="Offline Transformer OCR Evaluation")
    parser_eval.add_argument('--model', type=str, default=None, help="Optional: Pfad zum .pth Modell.")
    # Füge Argumente hinzu, um Architekturparameter zu laden (FALLS NICHT MIT MODELL GESPEICHERT!)
    # parser_eval.add_argument('--embed_dim', ... ) # etc. - Besser wäre, Config mit Modell zu speichern.
    args_eval = parser_eval.parse_args()

    # Prüfe, ob config schon geladen wurde (passiert bei Import)
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=logging.INFO) # Minimales Logging

    evaluate_offline_transformer(model_path_to_evaluate=args_eval.model)