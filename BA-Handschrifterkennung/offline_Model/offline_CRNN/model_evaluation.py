# model_evaluation.py

import os
import logging
import torch
import torch.nn as nn
from torch.utilities.data import DataLoader # Nur DataLoader importieren
# from sklearn.model_selection import train_test_split # Wird hier nicht direkt benötigt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime

import configuration
import utilities
# ÄNDERUNG: Importiere aus utilities statt model_training
from utilities import OfflineHandwritingDataset, collate_fn, load_data_for_ctc
import ocr_model
# setup_logging wird bei Bedarf aus model_training importiert (siehe __main__)

# -----------------------------------------------------------------------------
# model_evaluation.py
# -----------------------------------------------------------------------------
# Finale Evaluation auf allen Splits mit ausgelagerten Datenkomponenten.
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__) # Logger holen

def save_sample_predictions_html(sample_paths, true_labels, pred_labels, num_samples=10, split_name='test', base_path=None):
    """
    Speichert eine HTML-Datei mit Beispielbildern und ihren Vorhersagen.
    """
    if base_path is None:
        base_path = configuration.RESULTS_PATH
    os.makedirs(base_path, exist_ok=True) # Sicherstellen, dass Pfad existiert

    if not sample_paths:
        logger.warning(f"Keine Sample-Pfade für HTML-Report ({split_name}) vorhanden.")
        return

    html_content = f"<html><head><title>Sample Predictions ({split_name})</title>"
    # Einfaches CSS für bessere Darstellung
    html_content += """
    <style>
        body { font-family: sans-serif; margin: 20px; }
        h1 { border-bottom: 1px solid #ccc; padding-bottom: 10px; }
        .sample-container {
            margin-bottom: 20px;
            border: 1px solid #ccc;
            padding: 10px;
            display: inline-block;
            vertical-align: top;
            margin-right: 10px;
            max-width: 350px; /* Begrenzt Breite */
            word-wrap: break-word; /* Lange Wörter umbrechen */
        }
        .sample-container img {
            max-width: 300px;
            max-height: 100px; /* Höhe begrenzen */
            border: 1px solid lightgray;
            display: block;
            margin-bottom: 5px;
        }
        .label-text { font-family: monospace; margin-top: 5px; font-size: 1.1em; }
        .correct { color: green; }
        .incorrect { color: red; font-weight: bold; }
    </style>
    </head><body>"""
    html_content += f"<h1>Sample Predictions ({split_name.capitalize()})</h1>"

    num_available = len(sample_paths)
    num_samples = min(num_samples, num_available)
    if num_samples == 0:
        logger.warning(f"Keine Samples zum Anzeigen für {split_name}.")
        html_content += "<p>Keine Samples verfügbar.</p>"
    else:
        # Wähle zufällige Indizes aus den verfügbaren Daten
        # Stelle sicher, dass Indizes gültig sind für alle Listen
        max_idx = min(len(sample_paths), len(true_labels), len(pred_labels))
        if max_idx < num_samples:
             logger.warning(f"Weniger Daten ({max_idx}) als angeforderte Samples ({num_samples}). Zeige alle.")
             num_samples = max_idx
        indices = np.random.choice(max_idx, num_samples, replace=False) if max_idx > 0 else []


        for i in indices:
            try:
                img_path_relative = sample_paths[i]
                true_label = true_labels[i]
                pred_label = pred_labels[i]

                # Konstruiere absoluten Pfad
                img_path_abs = img_path_relative
                if configuration.WORDS_FOLDER and not os.path.isabs(img_path_relative):
                    img_path_abs = os.path.join(configuration.WORDS_FOLDER, img_path_relative)

                if not os.path.exists(img_path_abs):
                     logger.warning(f"Bild nicht gefunden für HTML: {img_path_abs}")
                     html_content += f"""
                     <div class="sample-container">
                         <p>Bild nicht gefunden: {img_path_relative}</p>
                         <p class="label-text">
                           <strong>Wahr:</strong> {true_label}<br/>
                           <strong>Pred:</strong> {pred_label}
                         </p>
                     </div>
                     """
                     continue

                # Füge Bild und Labels zum HTML hinzu
                pred_class = "correct" if true_label == pred_label else "incorrect"
                html_content += f"""
                <div class="sample-container">
                    <img src="{img_path_abs}" alt="Sample Image {i}"/>
                    <p class="label-text">
                        <strong>Wahr:</strong> {true_label}<br/>
                        <strong>Pred:</strong> <span class="{pred_class}">{pred_label}</span>
                    </p>
                </div>
                """
            except IndexError:
                 logger.error(f"Indexfehler beim Zugriff auf Sample {i} für HTML ({split_name}). Überspringe.")
            except Exception as e:
                logger.error(f"Fehler beim Erstellen des HTML-Eintrags für Sample {i} ({split_name}): {e}", exc_info=True)

    html_content += "</body></html>"
    html_path = os.path.join(base_path, f"{split_name}_sample_predictions.html")
    try:
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Beispielvorhersagen HTML für {split_name} gespeichert: {html_path}")
    except Exception as e:
        logger.error(f"Fehler beim Speichern der HTML-Datei {html_path}: {e}", exc_info=True)


def plot_error_distribution(all_true, all_pred, split_name, base_path=None):
    """
    Plottet die Verteilung der Fehler (CER und WER).
    """
    if base_path is None:
        base_path = configuration.RESULTS_PATH
    os.makedirs(base_path, exist_ok=True)

    if not all_true or not all_pred or len(all_true) != len(all_pred):
         logger.warning(f"Ungültige oder leere Daten für Fehlerverteilungsplot ({split_name}).")
         return

    try:
        # Berechne CER und WER, ignoriere Paare mit leerem true_str für CER
        cer_values = [utilities.levenshtein_distance(pr, tr) / max(1, len(tr))
                      for pr, tr in zip(all_pred, all_true) if tr]
        wer_values = [utilities.compute_wer(pr, tr)
                      for pr, tr in zip(all_pred, all_true)]

        plt.style.use('seaborn-v0_8-whitegrid')

        # --- Histogramme ---
        fig_hist, axes_hist = plt.subplots(1, 2, figsize=(14, 6))

        # CER Histogramm
        if cer_values:
            sns.histplot(cer_values, bins=30, kde=True, color='skyblue', ax=axes_hist[0])
            axes_hist[0].set_title(f'CER Verteilung ({split_name.capitalize()})')
            axes_hist[0].set_xlabel('Character Error Rate (CER)')
            axes_hist[0].set_ylabel('Häufigkeit')
            cer_max = max(1.0, np.max(cer_values)*1.1) if cer_values else 1.0
            axes_hist[0].set_xlim(0, cer_max)
        else:
             axes_hist[0].text(0.5, 0.5, 'Keine CER-Daten', ha='center', va='center', transform=axes_hist[0].transAxes)
             axes_hist[0].set_title(f'CER Verteilung ({split_name.capitalize()})')

        # WER Histogramm
        if wer_values:
            sns.histplot(wer_values, bins=30, kde=True, color='salmon', ax=axes_hist[1])
            axes_hist[1].set_title(f'WER Verteilung ({split_name.capitalize()})')
            axes_hist[1].set_xlabel('Word Error Rate (WER)')
            axes_hist[1].set_ylabel('Häufigkeit')
            wer_max = max(1.0, np.max(wer_values)*1.1) if wer_values else 1.0
            axes_hist[1].set_xlim(0, wer_max)
        else:
            axes_hist[1].text(0.5, 0.5, 'Keine WER-Daten', ha='center', va='center', transform=axes_hist[1].transAxes)
            axes_hist[1].set_title(f'WER Verteilung ({split_name.capitalize()})')


        fig_hist.suptitle(f"Fehlerverteilung (Histogramme) - {split_name.capitalize()}", fontsize=16)
        fig_hist.tight_layout(rect=[0, 0.03, 1, 0.95])
        hist_plot_path = os.path.join(base_path, f"{split_name}_error_distribution_hist.png")
        fig_hist.savefig(hist_plot_path, dpi=150)
        plt.close(fig_hist)
        logger.info(f"Histogramme von CER und WER für {split_name} gespeichert: {hist_plot_path}")


        # --- Boxplots ---
        fig_box, axes_box = plt.subplots(1, 2, figsize=(10, 6))

        # CER Boxplot
        if cer_values:
            sns.boxplot(y=cer_values, color='skyblue', showfliers=False, ax=axes_box[0]) # Ohne Ausreißer
            axes_box[0].set_title(f'CER Boxplot ({split_name.capitalize()})')
            axes_box[0].set_ylabel('Character Error Rate (CER)')
        else:
            axes_box[0].text(0.5, 0.5, 'Keine CER-Daten', ha='center', va='center', transform=axes_box[0].transAxes)
            axes_box[0].set_title(f'CER Boxplot ({split_name.capitalize()})')
            axes_box[0].set_ylabel('Character Error Rate (CER)')


        # WER Boxplot
        if wer_values:
            sns.boxplot(y=wer_values, color='salmon', showfliers=False, ax=axes_box[1])
            axes_box[1].set_title(f'WER Boxplot ({split_name.capitalize()})')
            axes_box[1].set_ylabel('Word Error Rate (WER)')
        else:
            axes_box[1].text(0.5, 0.5, 'Keine WER-Daten', ha='center', va='center', transform=axes_box[1].transAxes)
            axes_box[1].set_title(f'WER Boxplot ({split_name.capitalize()})')
            axes_box[1].set_ylabel('Word Error Rate (WER)')


        fig_box.suptitle(f"Fehlerverteilung (Boxplots) - {split_name.capitalize()}", fontsize=16)
        fig_box.tight_layout(rect=[0, 0.03, 1, 0.95])
        boxplot_path = os.path.join(base_path, f"{split_name}_error_distribution_boxplots.png")
        fig_box.savefig(boxplot_path, dpi=150)
        plt.close(fig_box)
        logger.info(f"Boxplots von CER und WER für {split_name} gespeichert: {boxplot_path}")

    except Exception as e:
        logger.error(f"Fehler beim Plotten der Fehlerverteilung für {split_name}: {e}", exc_info=True)
        # Schließe Plots, falls noch offen
        plt.close('all')


def generate_word_confusion_matrix(all_true, all_pred, top_n=30, split_name='test', base_path=None):
    """
    Generiert eine Konfusionsmatrix für die häufigsten `top_n` Wörter.
    """
    if base_path is None:
        base_path = configuration.RESULTS_PATH
    os.makedirs(base_path, exist_ok=True)

    if not all_true or not all_pred or len(all_true) != len(all_pred):
        logger.warning(f"Ungültige oder leere Daten für Konfusionsmatrix ({split_name}).")
        return

    from collections import Counter

    try:
        # Zähle Häufigkeit der wahren Wörter (ignoriere leere)
        true_word_counts = Counter(t for t in all_true if t)
        if not true_word_counts:
             logger.info(f"Keine nicht-leeren wahren Wörter für Konfusionsmatrix ({split_name}) gefunden.")
             return

        # Wähle die Top N häufigsten Wörter
        common_words = [word for word, count in true_word_counts.most_common(top_n)]
        if not common_words:
             logger.info(f"Keine häufigen Wörter für Konfusionsmatrix ({split_name}) nach Filterung gefunden.")
             return
        actual_top_n = len(common_words)

        # Filtern der Vorhersagen und Wahrheiten auf diese häufigen Wörter
        filtered_indices = [i for i, true_word in enumerate(all_true) if true_word in common_words]
        if not filtered_indices:
             logger.info(f"Keine der Top-{actual_top_n} häufigsten Wörter im Datensatz gefunden ({split_name}).")
             return

        filtered_true = [all_true[i] for i in filtered_indices]
        # Für Vorhersagen: Ersetze Wörter, die nicht zu den Top N gehören, durch eine Kategorie 'Other' oder ignoriere?
        # Wir mappen sie auf die vorhandenen Labels, wie confusion_matrix es tut.
        filtered_pred = [all_pred[i] for i in filtered_indices]

        # Erstelle die Konfusionsmatrix mit den Top N Wörtern als Labels
        cm = confusion_matrix(filtered_true, filtered_pred, labels=common_words)

        plt.style.use('default') # Standard-Stil für Heatmap
        # Dynamische Figurengröße
        fig_width = max(10, actual_top_n * 0.6)
        fig_height = max(8, actual_top_n * 0.5)
        plt.figure(figsize=(fig_width, fig_height))

        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                    xticklabels=common_words,
                    yticklabels=common_words,
                    annot_kws={"size": 8}) # Schriftgröße für Annotationen
        plt.title(f"Wort-Konfusionsmatrix (Top {actual_top_n} Wörter) - {split_name.capitalize()}", fontsize=14)
        plt.xlabel("Vorhergesagtes Wort", fontsize=12)
        plt.ylabel("Wahres Wort", fontsize=12)
        plt.xticks(rotation=90, fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        cm_path = os.path.join(base_path, f"{split_name}_word_confusion_matrix_top_{actual_top_n}.png")
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Wort-Konfusionsmatrix der Top {actual_top_n} Wörter für {split_name} gespeichert: {cm_path}")

    except Exception as e:
        logger.error(f"Fehler beim Erstellen der Wort-Konfusionsmatrix für {split_name}: {e}", exc_info=True)
        plt.close('all')


def plot_sample_images_with_predictions(sample_paths, true_labels, pred_labels, num_samples=10, split_name='test', base_path=None):
    """
    Plotet eine Auswahl von Bildern mit wahren und vorhergesagten Labels.
    """
    if base_path is None:
        base_path = configuration.RESULTS_PATH
    os.makedirs(base_path, exist_ok=True)

    if not sample_paths or not true_labels or not pred_labels:
         logger.warning(f"Ungültige oder leere Daten für Bild-Plot ({split_name}).")
         return

    num_available = min(len(sample_paths), len(true_labels), len(pred_labels))
    num_samples = min(num_samples, num_available)
    if num_samples <= 0:
        logger.warning(f"Keine Samples zum Plotten für {split_name}.")
        return

    # Wähle zufällige Indizes
    indices = np.random.choice(num_available, num_samples, replace=False)

    try:
        plt.style.use('default')
        n_cols = 2 if num_samples > 1 else 1
        n_rows = (num_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows), squeeze=False)
        axes = axes.flatten()

        plot_idx = 0
        for i in indices:
            if plot_idx >= len(axes): break # Sicherstellen, dass wir nicht über die Achsen hinausgehen
            ax = axes[plot_idx]
            try:
                img_path_relative = sample_paths[i]
                true_label = true_labels[i]
                pred_label = pred_labels[i]

                # Lade Bild über utilities.load_image
                image = utilities.load_image(img_path_relative)

                if image is None:
                    logger.warning(f"Bild konnte nicht geladen werden für Plot: {img_path_relative}")
                    ax.text(0.5, 0.5, f"Fehler beim Laden:\n{img_path_relative}", ha='center', va='center', wrap=True)
                    ax.set_title(f"Wahr: {true_label}\nPred: {pred_label}", fontsize=10)
                else:
                    # Bild ist (1, H, W), squeeze für Anzeige
                    image_display = image.squeeze(0) # => (H, W)
                    ax.imshow(image_display, cmap='gray')
                    title_color = 'green' if true_label == pred_label else 'red'
                    ax.set_title(f"Wahr: {true_label}\nPred: {pred_label}", fontsize=11, color=title_color)

                ax.axis('off')
                plot_idx += 1

            except IndexError:
                 logger.error(f"Indexfehler beim Zugriff auf Sample {i} für Plot ({split_name}). Überspringe.")
                 ax.set_visible(False) # Verstecke Achse bei Fehler
                 plot_idx += 1 # Stelle sicher, dass der Index weitergeht
            except Exception as e:
                 logger.error(f"Fehler beim Plotten von Sample {i} ({split_name}): {e}", exc_info=True)
                 ax.set_visible(False)
                 plot_idx += 1

        # Übrige leere Achsen entfernen
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle(f"Beispielbilder mit Vorhersagen ({split_name.capitalize()})", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        sample_images_path = os.path.join(base_path, f"{split_name}_sample_images_predictions.png")
        fig.savefig(sample_images_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Beispielbilder mit Vorhersagen für {split_name} gespeichert: {sample_images_path}")

    except Exception as e:
        logger.error(f"Genereller Fehler beim Plotten der Beispielbilder für {split_name}: {e}", exc_info=True)
        plt.close('all')


def plot_metrics_comparison(all_split_metrics, base_path=None):
    """
    Plottet einen Vergleich der Metriken über alle Splits (Train, Val, Test).
    Verwendet die Metriken aus dem all_split_metrics Dict.
    """
    if base_path is None:
        base_path = configuration.RESULTS_PATH
    os.makedirs(base_path, exist_ok=True)

    if not all_split_metrics or len(all_split_metrics) < 2: # Mindestens 2 Splits für Vergleich
         logger.warning("Nicht genügend Split-Metriken für Vergleichsplot vorhanden.")
         return

    try:
        # Sammle Metriknamen, die in *allen* vorhandenen Splits vorkommen
        available_splits = list(all_split_metrics.keys())
        if not available_splits: return

        common_metrics = set(all_split_metrics[available_splits[0]].keys())
        for split in available_splits[1:]:
             common_metrics.intersection_update(all_split_metrics[split].keys())

        # Filtere und sortiere Metriken für den Plot
        metrics_to_plot = [m for m in configuration.METRICS_TO_COMPUTE if m in common_metrics and m != 'loss']
        # Füge CER/WER hinzu, falls nicht schon da und verfügbar
        if 'cer' in common_metrics and 'cer' not in metrics_to_plot: metrics_to_plot.append('cer')
        if 'wer' in common_metrics and 'wer' not in metrics_to_plot: metrics_to_plot.append('wer')
        # Entferne BLEU < 4 für Hauptplot
        metrics_to_plot = [m for m in metrics_to_plot if not (m.startswith('bleu_') and m != 'bleu_4')]
        metrics_to_plot = sorted(list(set(metrics_to_plot)))

        if not metrics_to_plot:
             logger.warning("Keine gemeinsamen Metriken (außer Loss/BLEU<4) für Vergleichsplot gefunden.")
             # Versuche trotzdem BLEU zu plotten
        else:
            # --- Hauptmetriken Plot ---
            plot_data = {split: [all_split_metrics[split][metric] for metric in metrics_to_plot] for split in available_splits}
            valid_metric_names = [m.upper().replace('_', '-') for m in metrics_to_plot]

            plt.style.use('seaborn-v0_8-talk')
            plt.figure(figsize=(max(10, len(valid_metric_names) * 1.5), 7))
            x = np.arange(len(valid_metric_names))
            width = 0.25
            colors = {'train': 'skyblue', 'val': 'salmon', 'test': 'lightgreen'}

            offset = -width * (len(available_splits) - 1) / 2 # Zentriere Balken
            for split in available_splits:
                plt.bar(x + offset, plot_data[split], width, label=split.capitalize(), color=colors.get(split, 'gray'))
                offset += width

            plt.xlabel('Metrik', fontsize=14)
            plt.ylabel('Wert', fontsize=14)
            plt.title('Vergleich der Metriken über die Splits', fontsize=16)
            plt.xticks(x, valid_metric_names, rotation=45, ha='right', fontsize=11)
            plt.yticks(fontsize=11)
            plt.legend(fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            all_values = [v for split_vals in plot_data.values() for v in split_vals if v is not None and not np.isnan(v)]
            max_y = max(1.05, np.max(all_values) * 1.1) if all_values else 1.05
            plt.ylim(0, max_y)
            plt.tight_layout()

            metrics_comparison_path = os.path.join(base_path, "metrics_comparison_bar.png")
            plt.savefig(metrics_comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Metriken-Vergleich (Balken) gespeichert: {metrics_comparison_path}")

        # --- BLEU Scores Plot ---
        bleu_metrics = sorted([m for m in common_metrics if m.startswith('bleu_')])
        if bleu_metrics:
            bleu_names = [m.upper().replace('_', '-') for m in bleu_metrics]
            bleu_data = {split: [all_split_metrics[split][metric] for metric in bleu_metrics] for split in available_splits}

            plt.style.use('seaborn-v0_8-talk')
            plt.figure(figsize=(10, 6))
            x_bleu = np.arange(len(bleu_names))
            width_bleu = 0.25

            offset = -width_bleu * (len(available_splits) - 1) / 2 # Zentrieren
            for split in available_splits:
                plt.bar(x_bleu + offset, bleu_data[split], width_bleu, label=split.capitalize(), color=colors.get(split, 'gray'))
                offset += width_bleu

            plt.xlabel('BLEU Score Typ', fontsize=14)
            plt.ylabel('Wert', fontsize=14)
            plt.title('Vergleich der BLEU Scores über die Splits', fontsize=16)
            plt.xticks(x_bleu, bleu_names, fontsize=11)
            plt.yticks(fontsize=11)
            plt.legend(fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            all_bleu_values = [v for split_vals in bleu_data.values() for v in split_vals if v is not None and not np.isnan(v)]
            max_y_bleu = max(1.0, np.max(all_bleu_values) * 1.1) if all_bleu_values else 1.0
            plt.ylim(0, max_y_bleu)
            plt.tight_layout()

            bleu_comparison_path = os.path.join(base_path, "bleu_scores_comparison_bar.png")
            plt.savefig(bleu_comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"BLEU-Vergleich (Balken) gespeichert: {bleu_comparison_path}")
        else:
             logger.info("Keine gemeinsamen BLEU Metriken für Vergleichsplot gefunden.")

    except Exception as e:
        logger.error(f"Fehler beim Plotten des Metrikvergleichs: {e}", exc_info=True)
        plt.close('all')


def analyze_length_accuracy(all_true, all_pred, split_name='test', base_path=None):
    """
    Analysiert die Wortgenauigkeit (exakter Match) nach Wortlänge.
    """
    if base_path is None:
        base_path = configuration.RESULTS_PATH
    os.makedirs(base_path, exist_ok=True)

    if not all_true or not all_pred or len(all_true) != len(all_pred):
         logger.warning(f"Ungültige oder leere Daten für Längenanalyse ({split_name}).")
         return

    try:
        length_metrics = {} # key: length, value: {'correct': count, 'total': count}
        for true, pred in zip(all_true, all_pred):
            length = len(true)
            if length <= 0: continue # Ignoriere leere wahre Labels

            if length not in length_metrics:
                length_metrics[length] = {'correct': 0, 'total': 0}
            length_metrics[length]['total'] += 1
            if true == pred:
                length_metrics[length]['correct'] += 1

        if not length_metrics:
            logger.warning(f"Keine gültigen Längen für Analyse gefunden ({split_name}).")
            return

        # Aufbereiten für Plot
        lengths = sorted(length_metrics.keys())
        accuracies = [length_metrics[l]['correct'] / length_metrics[l]['total'] for l in lengths]
        counts = [length_metrics[l]['total'] for l in lengths]

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 1, figsize=(max(10, len(lengths)*0.5), 10), sharex=True) # Skalierbare Breite

        # Genauigkeits-Plot
        sns.barplot(x=lengths, y=accuracies, palette='viridis', ax=axes[0])
        axes[0].set_title(f'Wortgenauigkeit nach Länge ({split_name.capitalize()})')
        axes[0].set_ylabel('Genauigkeit (Exakter Match)')
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        # Optional: Werte auf Balken anzeigen
        for i, acc in enumerate(accuracies):
            if counts[i] > 0: # Nur wenn Samples vorhanden
                axes[0].text(i, acc + 0.02, f'{acc:.2f}', ha='center', va='bottom', fontsize=9)

        # Anzahls-Plot
        sns.barplot(x=lengths, y=counts, palette='mako', ax=axes[1]) # Anderes Farbschema
        axes[1].set_title(f'Anzahl Samples nach Länge ({split_name.capitalize()})')
        axes[1].set_xlabel('Wortlänge (Zeichen)')
        axes[1].set_ylabel('Anzahl Samples')
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        # Optional: Werte auf Balken anzeigen
        for i, count in enumerate(counts):
            axes[1].text(i, count + max(counts)*0.01 , f'{count}', ha='center', va='bottom', fontsize=9)

        # X-Achsen-Ticks anpassen, wenn zu viele Längen vorhanden sind
        if len(lengths) > 30:
             step = max(1, len(lengths) // 15) # Zeige ca. 15 Ticks
             axes[1].set_xticks(np.arange(0, len(lengths), step))
             axes[1].set_xticklabels(lengths[::step])


        plt.tight_layout()
        length_analysis_path = os.path.join(base_path, f"{split_name}_length_vs_accuracy.png")
        fig.savefig(length_analysis_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Analyse Wortlänge vs. Genauigkeit für {split_name} gespeichert: {length_analysis_path}")

    except Exception as e:
        logger.error(f"Fehler bei der Längenanalyse für {split_name}: {e}", exc_info=True)
        plt.close('all')


def evaluate_model_on_split(model, data_loader, criterion, device, split_name):
    """
    Evaluiert das Modell auf einem bestimmten Split (DataLoader).
    Verwendet utilities.compute_metrics für die Metrikberechnung.
    Gibt Metriken, True/Pred Labels und eine Auswahl an Pfaden zurück.
    """
    model.eval() # In Evaluationsmodus setzen
    running_loss = 0.0
    all_true_labels = [] # Liste von String-Labels
    all_pred_labels = [] # Liste von String-Vorhersagen
    all_image_paths = [] # Sammelt die Pfade der tatsächlich evaluierten Samples
    evaluated_samples = 0

    # Hole die Pfade aus dem zugrundeliegenden Dataset, falls möglich
    # Annahme: DataLoader wurde mit utilities.OfflineHandwritingDataset erstellt
    original_image_paths = []
    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'image_paths'):
         original_image_paths = data_loader.dataset.image_paths
         # Wenn es ein Subset ist, brauchen wir die Indizes
         if isinstance(data_loader.dataset, torch.utilities.data.Subset):
              original_image_paths = [data_loader.dataset.dataset.image_paths[i] for i in data_loader.dataset.indices]


    logger.info(f"Starte Evaluation auf {split_name}-Split...")

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f"Eval {split_name}", leave=False, ncols=100, mininterval=1.0)
        batch_count = 0
        base_idx = 0 # Um Pfade zuzuordnen, wenn shuffle=False

        for batch_data in progress_bar:
            if not batch_data or len(batch_data) != 3: continue # Überspringe ungültigen Batch
            batch_images, batch_labels, label_lengths = batch_data
            if batch_images.numel() == 0: continue # Überspringe leeren Batch

            current_batch_size = batch_images.size(0)
            # Hole die Pfade für diesen Batch (funktioniert nur, wenn shuffle=False)
            batch_paths = original_image_paths[base_idx : base_idx + current_batch_size] if original_image_paths else ["N/A"] * current_batch_size
            base_idx += current_batch_size


            try:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)
                label_lengths = label_lengths.to(device)

                # Forward Pass
                logits = model(batch_images) # (T, N, C)
                log_probs = logits.log_softmax(2)
                T = logits.size(0)
                input_lengths = torch.full((current_batch_size,), T, dtype=torch.long, device=device)

                # Loss
                loss = criterion(log_probs, batch_labels, input_lengths, label_lengths)
                if torch.isinf(loss) or torch.isnan(loss): continue # Überspringe bei ungültigem Loss

                running_loss += loss.item() * current_batch_size
                evaluated_samples += current_batch_size

                # Decoding (Prediction)
                preds_np = logits.cpu().numpy()
                recognized = utilities.decode_predictions(preds_np, configuration.IDX_TO_CHAR, configuration.NUM_CLASSES)

                # Decoding (True Labels)
                start_idx = 0
                true_strs_batch = []
                labels_cpu = batch_labels.cpu().numpy()
                lengths_cpu = label_lengths.cpu().numpy()
                for length in lengths_cpu:
                    end_idx = start_idx + length
                    label_seq = labels_cpu[start_idx:end_idx]
                    start_idx = end_idx
                    dec_str = utilities.decode_labels(label_seq, configuration.NUM_CLASSES)[0]
                    true_strs_batch.append(dec_str)

                all_true_labels.extend(true_strs_batch)
                all_pred_labels.extend(recognized)
                all_image_paths.extend(batch_paths) # Füge Pfade der erfolgreich verarbeiteten Samples hinzu

                # Update Progress Bar
                if evaluated_samples > 0:
                    progress_bar.set_postfix({'Loss': f"{running_loss / evaluated_samples:.4f}"})
                batch_count += 1

            except Exception as e:
                logger.error(f"Fehler während Evaluation ({split_name}) in Batch {batch_count}: {e}", exc_info=True)
                # Sorge dafür, dass base_idx trotzdem erhöht wird, um Pfade synchron zu halten
                base_idx = min(base_idx + current_batch_size - len(batch_paths), len(original_image_paths)) # Korrigiere base_idx
                continue

    # Finale Metriken für den Split berechnen
    avg_loss = running_loss / evaluated_samples if evaluated_samples > 0 else float('inf')
    metrics = utilities.compute_metrics(all_true_labels, all_pred_labels)
    metrics['loss'] = avg_loss

    log_metrics = {k: v for k, v in metrics.items() if k in configuration.METRICS_TO_COMPUTE or k == 'loss'}
    metrics_str = ", ".join([f"{k.upper().replace('_','-')}={v:.4f}" for k, v in sorted(log_metrics.items())])
    logger.info(f"Evaluation abgeschlossen ({split_name}): Samples={evaluated_samples}, {metrics_str}")

    # Speichere Metriken für diesen Split
    utilities.save_metrics_to_file(metrics, split_name, configuration.METRICS_PATH)

    # Erstelle Visualisierungen für diesen Split im Unterordner
    results_path_split = os.path.join(configuration.RESULTS_PATH, split_name)
    os.makedirs(results_path_split, exist_ok=True)

    if evaluated_samples > 0:
        plot_error_distribution(all_true_labels, all_pred_labels, split_name, results_path_split)
        generate_word_confusion_matrix(all_true_labels, all_pred_labels, top_n=30, split_name=split_name, base_path=results_path_split)
        analyze_length_accuracy(all_true_labels, all_pred_labels, split_name=split_name, base_path=results_path_split)

        # Wähle Samples für Plot und HTML
        num_samples_to_show = min(10, evaluated_samples)
        if num_samples_to_show > 0:
             indices = np.random.choice(evaluated_samples, num_samples_to_show, replace=False)
             sample_paths = [all_image_paths[i] for i in indices]
             sample_true = [all_true_labels[i] for i in indices]
             sample_pred = [all_pred_labels[i] for i in indices]

             plot_sample_images_with_predictions(sample_paths, sample_true, sample_pred, num_samples=num_samples_to_show, split_name=split_name, base_path=results_path_split)
             save_sample_predictions_html(sample_paths, sample_true, sample_pred, num_samples=num_samples_to_show, split_name=split_name, base_path=results_path_split)
        else:
             sample_paths, sample_true, sample_pred = [], [], []


        # Speichere *alle* Vorhersagen in einer CSV
        try:
            predictions_df = pd.DataFrame({
                'ImagePath': all_image_paths, # Pfade der tatsächlich evaluierten Bilder
                'TrueLabel': all_true_labels,
                'PredictedLabel': all_pred_labels
            })
            # Sortiere optional nach ImagePath
            # predictions_df = predictions_df.sort_values(by='ImagePath').reset_index(drop=True)
            predictions_csv_path = os.path.join(results_path_split, f"{split_name}_all_predictions.csv")
            predictions_df.to_csv(predictions_csv_path, index=False, encoding='utf-8')
            logger.info(f"Alle {len(predictions_df)} Vorhersagen für {split_name} gespeichert: {predictions_csv_path}")
        except Exception as e:
             logger.error(f"Fehler beim Speichern der Vorhersagen-CSV für {split_name}: {e}", exc_info=True)
    else:
         logger.warning(f"Keine Samples erfolgreich evaluiert für {split_name}. Keine Plots/Reports erstellt.")
         sample_paths, sample_true, sample_pred = [], [], []

    # Gebe die gesammelten Daten zurück (Pfade nur für Samples)
    return metrics, all_true_labels, all_pred_labels, sample_paths, sample_true, sample_pred


def load_train_val_splits():
    """
    Lädt die Train- und Val-Splits aus den gespeicherten Dateien.
    Falls nicht vorhanden, werden sie neu erstellt.
    Gibt Pfade und *kodierte* Labels zurück.
    """
    train_paths, train_labels = utilities.load_split_data(configuration.TRAIN_LABELS_FILE)
    val_paths, val_labels = utilities.load_split_data(configuration.VAL_LABELS_FILE)

    if not train_paths or not val_paths:
        logger.warning("Train/Val-Splits nicht gefunden oder leer, lade Gesamtdaten und erstelle 80/10/10-Split neu.")
        # Gesamtdaten laden (gibt kodierte Labels zurück)
        all_paths, all_labels_encoded = utilities.load_data_for_ctc(configuration.WORDS_CSV, configuration.WORDS_FOLDER)
        if not all_paths:
             logger.error("Keine Gesamtdaten zum Erstellen der Splits gefunden.")
             return [], [], [], [] # Leere Listen zurückgeben

        # Splits neu erstellen und speichern (gibt kodierte Labels zurück)
        # Annahme: create_data_splits ist in diesem Modul nicht verfügbar, aber das macht nichts,
        # da es nur aufgerufen wird, wenn die Dateien fehlen. Sie *sollten* von main.py erstellt werden.
        # Wenn dieses Skript standalone läuft, fehlt create_data_splits hier.
        # Wir verlassen uns darauf, dass die Dateien existieren oder main.py sie erstellt hat.
        # => Wenn wir hier landen, ist etwas schiefgelaufen. Wir geben leere Listen zurück.
        logger.error("Konnte Train/Val-Splits nicht laden und create_data_splits ist hier nicht verfügbar.")
        return [], [], [], []

    # Sicherstellen, dass Labels Listen sind (redundant, da load_split_data das tun sollte)
    if train_labels and not isinstance(train_labels[0], list):
         logger.error("Trainingslabels sind nicht korrekt kodiert (keine Listen).")
    if val_labels and not isinstance(val_labels[0], list):
         logger.error("Validierungslabels sind nicht korrekt kodiert (keine Listen).")

    return train_paths, train_labels, val_paths, val_labels


def load_test_split():
    """
    Lädt den Testsplit aus der gespeicherten Datei.
    Falls nicht vorhanden, wird er neu erstellt (theoretisch, praktisch fehlt create hier).
    Gibt Pfade und *kodierte* Labels zurück.
    """
    test_paths, test_labels = utilities.load_split_data(configuration.TEST_LABELS_FILE)

    if not test_paths:
        logger.warning("Test-Split nicht gefunden oder leer.")
        # Wie bei load_train_val_splits, können wir hier nicht neu erstellen.
        # Wir verlassen uns darauf, dass die Datei existiert.
        logger.error(f"Test-Split-Datei {configuration.TEST_LABELS_FILE} nicht gefunden oder leer.")
        return [], [] # Leere Listen zurückgeben

    if test_labels and not isinstance(test_labels[0], list):
         logger.error("Testlabels sind nicht korrekt kodiert (keine Listen).")

    return test_paths, test_labels


def evaluate_offline_model():
    """
    Hauptfunktion für die finale Evaluation.
    Lädt das beste Modell und evaluiert es auf allen drei Splits.
    Speichert Metriken, Plots und einen Gesamtbericht.
    """
    eval_start_time = datetime.now()
    logger.info("===== Starte Finale Offline-Modell Evaluation =====")
    device = torch.device(configuration.DEVICE)

    # Finde das beste Modell
    best_model_path = os.path.join(configuration.CHECKPOINT_PATH, "best_crnn_model.pth")

    # Fallback-Suche, falls CHECKPOINT_PATH leer ist oder nicht existiert
    if not os.path.exists(best_model_path):
        logger.warning(f"Standard-Checkpoint {best_model_path} nicht gefunden.")
        # Suche im Basis-Checkpoint-Ordner nach dem neuesten versionierten Ordner
        base_checkpoint_dir = os.path.dirname(configuration.CHECKPOINT_PATH) # z.B. .../checkpoints/
        latest_version_dir = None
        latest_mtime = 0
        if os.path.isdir(base_checkpoint_dir):
             try:
                 for d in os.listdir(base_checkpoint_dir):
                     full_dir_path = os.path.join(base_checkpoint_dir, d)
                     if os.path.isdir(full_dir_path):
                          mtime = os.path.getmtime(full_dir_path)
                          if mtime > latest_mtime:
                               potential_model = os.path.join(full_dir_path, "best_crnn_model.pth")
                               if os.path.exists(potential_model):
                                    latest_mtime = mtime
                                    latest_version_dir = full_dir_path
                                    best_model_path = potential_model
             except Exception as e:
                  logger.error(f"Fehler beim Suchen nach alternativem Checkpoint: {e}")

        if latest_version_dir:
             logger.warning(f"Verwende alternativ neuesten gefundenen Checkpoint: {best_model_path}")
             # Setze Pfade auf diese Version für Konsistenz der Ausgabe
             configuration.MODEL_VERSION = os.path.basename(latest_version_dir)
             configuration.CHECKPOINT_PATH = latest_version_dir
             configuration.RESULTS_PATH = os.path.join(configuration.BASE_PATH, "results", configuration.MODEL_VERSION)
             configuration.METRICS_PATH = os.path.join(configuration.BASE_PATH, "metrics", configuration.MODEL_VERSION)
             logger.info(f"Ausgabe-Pfade auf Version '{configuration.MODEL_VERSION}' gesetzt.")
        else:
             logger.error(f"Kein Modell für Evaluation gefunden. Bitte trainieren oder Checkpoint angeben.")
             return

    logger.info(f"Lade bestes Modell von: {best_model_path}")
    try:
        model = ocr_model.build_crnn_model().to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
    except Exception as e:
        logger.error(f"Fehler beim Laden des Modells {best_model_path}: {e}", exc_info=True)
        return

    # Verlustfunktion für Evaluation (gleiche wie Training)
    criterion = nn.CTCLoss(blank=configuration.CHAR_TO_IDX['blank'], reduction='mean', zero_infinity=True)

    # Lade Daten für alle drei Splits
    logger.info("Lade Datensplits für Evaluation...")
    try:
        train_paths, train_labels, val_paths, val_labels = load_train_val_splits()
        test_paths, test_labels = load_test_split()
    except Exception as e:
         logger.error(f"Fehler beim Laden der Datensplits: {e}", exc_info=True)
         return

    # Prüfen, ob Daten geladen wurden
    if not train_paths and not val_paths and not test_paths:
        logger.error("Keine Daten in den Splits gefunden. Evaluation nicht möglich.")
        return

    logger.info(f"Daten geladen: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")

    # Erstelle Datasets und DataLoader
    num_workers = 0 # Sicherer für Evaluation
    pin_memory = (configuration.DEVICE == "cuda")
    try:
        train_loader = DataLoader(
            OfflineHandwritingDataset(train_paths, train_labels, augment=False),
            batch_size=configuration.BATCH_SIZE, shuffle=False, collate_fn=utilities.collate_fn, num_workers=num_workers, pin_memory=pin_memory
        ) if train_paths else None

        val_loader = DataLoader(
            OfflineHandwritingDataset(val_paths, val_labels, augment=False),
            batch_size=configuration.BATCH_SIZE, shuffle=False, collate_fn=utilities.collate_fn, num_workers=num_workers, pin_memory=pin_memory
        ) if val_paths else None

        test_loader = DataLoader(
            OfflineHandwritingDataset(test_paths, test_labels, augment=False),
            batch_size=configuration.BATCH_SIZE, shuffle=False, collate_fn=utilities.collate_fn, num_workers=num_workers, pin_memory=pin_memory
        ) if test_paths else None

    except Exception as e:
        logger.error(f"Fehler beim Erstellen der DataLoaders für Evaluation: {e}", exc_info=True)
        return

    # --- Evaluiere auf allen drei Splits ---
    all_metrics = {}
    all_split_data = {} # Zum Speichern von Labels und Pfaden für Gesamtplots

    # Train Split Evaluation
    if train_loader:
        logger.info("--- Starte Evaluation auf dem TRAIN-Split ---")
        try:
            train_metrics, train_true, train_pred, train_paths_sample, _, _ = \
                evaluate_model_on_split(model, train_loader, criterion, device, 'train')
            all_metrics['train'] = train_metrics
            # Speichere nur Pfade der Samples für den Report
            all_split_data['train'] = {'sample_paths': train_paths_sample}
        except Exception as e:
            logger.error(f"Fehler bei der Evaluation des Train-Splits: {e}", exc_info=True)
            all_metrics['train'] = None
    else:
        logger.warning("Trainings-Split leer, Evaluation übersprungen.")
        all_metrics['train'] = None

    # Validation Split Evaluation
    if val_loader:
        logger.info("--- Starte Evaluation auf dem VALIDATION-Split ---")
        try:
            val_metrics, val_true, val_pred, val_paths_sample, _, _ = \
                evaluate_model_on_split(model, val_loader, criterion, device, 'val')
            all_metrics['val'] = val_metrics
            all_split_data['val'] = {'sample_paths': val_paths_sample}
        except Exception as e:
            logger.error(f"Fehler bei der Evaluation des Validation-Splits: {e}", exc_info=True)
            all_metrics['val'] = None
    else:
         logger.warning("Validierungs-Split leer, Evaluation übersprungen.")
         all_metrics['val'] = None

    # Test Split Evaluation
    if test_loader:
        logger.info("--- Starte Evaluation auf dem TEST-Split ---")
        try:
            test_metrics, test_true, test_pred, test_paths_sample, _, _ = \
                evaluate_model_on_split(model, test_loader, criterion, device, 'test')
            all_metrics['test'] = test_metrics
            all_split_data['test'] = {'sample_paths': test_paths_sample}
        except Exception as e:
            logger.error(f"Fehler bei der Evaluation des Test-Splits: {e}", exc_info=True)
            all_metrics['test'] = None
    else:
         logger.warning("Test-Split leer, Evaluation übersprungen.")
         all_metrics['test'] = None

    # --- Zusammenfassung und Berichte ---
    valid_metrics = {k: v for k, v in all_metrics.items() if v is not None}
    if not valid_metrics:
        logger.error("Evaluation auf allen verfügbaren Splits fehlgeschlagen. Keine Berichte erstellt.")
        return

    # Speichere zusammenfassende Metriken
    # Verwende utilities.save_metrics_to_file, das Serialisierung handhaben sollte
    utilities.save_metrics_to_file(valid_metrics, "evaluation_summary", configuration.METRICS_PATH)

    # Erstelle Vergleichsplots
    plot_metrics_comparison(valid_metrics, configuration.RESULTS_PATH)

    # Erstelle einen zusammenfassenden JSON-Bericht
    eval_end_time = datetime.now()
    # Versuche, Trainings-History zu laden für mehr Infos
    epochs_trained = 'N/A'
    training_duration = 'N/A'
    history_path = os.path.join(configuration.RESULTS_PATH, "training_history.json")
    if os.path.exists(history_path):
        try:
             with open(history_path, 'r') as f: history_data = json.load(f)
             if 'epoch' in history_data and history_data['epoch']:
                  epochs_trained = history_data['epoch'][-1]
             # Suche nach Trainingszeit in final_summary (falls Training vorher lief)
             final_summary_path = os.path.join(configuration.METRICS_PATH, "final_summary_metrics.json")
             if os.path.exists(final_summary_path):
                   with open(final_summary_path, 'r') as f_sum: final_sum_data = json.load(f_sum)
                   training_duration = final_sum_data.get('info',{}).get('total_training_time_sec', 'N/A')

        except Exception as e:
             logger.warning(f"Konnte Trainingshistorie {history_path} nicht laden/parsen: {e}")


    report = {
        'evaluation_run_info': {
            'timestamp': eval_end_time.strftime("%Y-%m-%d %H:%M:%S"),
            'duration_sec': round((eval_end_time - eval_start_time).total_seconds(), 2),
            'model_version': configuration.MODEL_VERSION,
        },
        'model_info': {
            'checkpoint_path': best_model_path,
            'architecture': "CRNN (Details: model_architecture.txt)",
            'num_classes': configuration.NUM_CLASSES
        },
        'dataset_info': {
            'source_csv': configuration.WORDS_CSV,
            'word_image_folder': configuration.WORDS_FOLDER,
            'split_files': {
                'train': configuration.TRAIN_LABELS_FILE,
                'val': configuration.VAL_LABELS_FILE,
                'test': configuration.TEST_LABELS_FILE,
            },
            'sizes': {
                'train': len(train_paths),
                'val': len(val_paths),
                'test': len(test_paths),
            }
        },
        'hyperparameters_at_eval_time': { # Könnten sich von Trainingszeit unterscheiden
             'learning_rate': configuration.LEARNING_RATE,
             'batch_size': configuration.BATCH_SIZE,
             'img_height': configuration.IMG_HEIGHT,
             'pad_to_width': configuration.PAD_TO_WIDTH,
             'char_list_size': len(configuration.CHAR_LIST),
             'epochs_trained': epochs_trained, # Aus History gelesen
             'training_duration_sec': training_duration, # Aus final_summary gelesen
        },
        'evaluation_metrics': valid_metrics
    }

    report_path = os.path.join(configuration.RESULTS_PATH, "evaluation_report.json")
    try:
        # Speichern mit Fallback für nicht-serialisierbare Typen
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=lambda o: f"<not serializable: {type(o).__name__}>")
        logger.info(f"Evaluationsbericht gespeichert: {report_path}")
    except Exception as e:
         logger.error(f"Fehler beim Speichern des Evaluationsberichts: {e}", exc_info=True)

    logger.info(f"===== Finale Evaluation abgeschlossen. Dauer: {(eval_end_time - eval_start_time).total_seconds():.2f} Sek. =====")


if __name__ == "__main__":
    # Führe die Evaluation aus (Logging wird ggf. hier initialisiert)
    if not logging.getLogger().hasHandlers():
         from model_training import setup_logging
         setup_logging()
         # Alternativ: Einfaches BasicConfig
         # logging.basicConfig(level=configuration.LOGGING_LEVEL,
         #                    format='%(asctime)s [%(levelname)-8s] %(name)-15s: %(message)s',
         #                    datefmt='%Y-%m-%d %H:%M:%S')
    evaluate_offline_model()