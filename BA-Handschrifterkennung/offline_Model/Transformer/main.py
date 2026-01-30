# main.py
import argparse
import logging # HIER importieren
import os
import sys
import shutil
from datetime import datetime

# --- Pfad Setup ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path: sys.path.insert(0, project_root)

# === WICHTIG: Logging Setup VOR allen anderen Projekt-Imports ===
# Versuche, den Ausgabeordner aus einer temporären Config-Lesung zu holen
# oder verwende einen Standard-Fallback.
temp_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
temp_run_output_path = os.path.join(project_root, "outputs", f"temp_log_{temp_timestamp}")
log_file_path = os.path.join(temp_run_output_path, "run_initial.log") # Initiales Log

try:
    # Versuche, RUN_OUTPUT_PATH zu bekommen, wenn configuration.py geladen werden kann
    try:
        # Temporärer Import nur für Pfad (ohne Logging-Nebenwirkungen in config)
        import configuration as temp_configuration
        log_file_path = os.path.join(temp_configuration.RUN_OUTPUT_PATH, "run.log")
        temp_run_output_path = temp_configuration.RUN_OUTPUT_PATH # Update für spätere Nutzung
    except Exception as cfg_read_err:
        print(f"[WARNUNG] Konnte Pfad aus configuration.py nicht lesen ({cfg_read_err}). Verwende temporären Logpfad: {log_file_path}", file=sys.stderr)

    # Stelle Log-Ordner sicher
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Basiskonfiguration
    logging.basicConfig(
        level=logging.INFO, # Standard-Level, wird ggf. durch config überschrieben
        format='%(asctime)s [%(levelname)-7s] %(name)-25s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'), # Überschreibe Log bei jedem Start
            logging.StreamHandler(sys.stdout)
        ]
    )
    initial_logger = logging.getLogger("main_setup") # Logger für den Setup-Prozess
    initial_logger.info("Logging initial konfiguriert in main.py.")

except Exception as log_setup_error:
    print(f"[FATAL] Kritischer Fehler beim Logging-Setup in main.py: {log_setup_error}", file=sys.stderr)
    sys.exit(1)
# === Ende Logging Setup ===


# --- JETZT erst die restlichen Module importieren ---
try:
    import configuration # Lädt NUR die Variablen
    import utilities
    import dataset_preparation
    import image_preprocessing
    import model_training
    import model_evaluation
except ImportError as e:
    initial_logger.critical(f"Kritischer Importfehler NACH Logging-Setup: {e}", exc_info=True)
    sys.exit(1)
except Exception as e:
    initial_logger.critical(f"Kritischer Fehler beim Importieren der Module: {e}", exc_info=True)
    sys.exit(1)

# Holen des Loggers für DIESES Modul (main.py)
# Erbt Konfiguration vom Root, die wir gerade gesetzt haben.
logger = logging.getLogger(__name__) # Standard __name__ für main

# =============================================================================
# main.py - Orchestrierung der Offline Transformer OCR Pipeline
# =============================================================================
def main():
    """ Hauptfunktion, parst Argumente und ruft die Pipeline-Schritte auf. """
    pipeline_start_time = datetime.now()
    logger.info("==========================================================")
    logger.info("=== STARTE OFFLINE TRANSFORMER OCR PIPELINE ORCHESTRIERUNG ===")
    logger.info(f"=== Startzeit: {pipeline_start_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    logger.info("==========================================================")

    # --- Setze Seeds NACH Logging Setup aber VOR Argument Parsing / Config Überschreibung ---
    try:
        logger.info(f"Setze Random Seed auf {configuration.RANDOM_SEED}")
        utilities.set_seeds(configuration.RANDOM_SEED) # Nutze Funktion aus utils
    except Exception as seed_e:
        logger.error(f"Fehler beim Setzen des Seeds: {seed_e}")

    # --- Ordnerstruktur erstellen (jetzt, da config geladen ist) ---
    try:
        logger.info("Stelle Ausgabeordner sicher...")
        # Erstelle nur den Haupt-Run-Ordner, Unterordner nach Bedarf
        utilities.ensure_dir(configuration.RUN_OUTPUT_PATH)
        logger.info(f"Ausgabeordner für diesen Lauf: {configuration.RUN_OUTPUT_PATH}")
    except Exception as dir_e:
        logger.critical(f"Konnte Ausgabeordner nicht erstellen: {dir_e}", exc_info=True)
        sys.exit(1)


    # --- Argument Parser Setup ---
    parser = argparse.ArgumentParser(
        description="Orchestriert die Offline Transformer OCR Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # ... (Argumente bleiben gleich wie vorher) ...
    # --- Hauptaktionen / Pipeline-Schritte ---
    parser.add_argument('--prepare_data', action='store_true',
                        help="[Schritt 1] Prüft/Entpackt Bildarchive (z.B. lines.tgz).")
    parser.add_argument('--preprocess_images', action='store_true',
                        help="[Schritt 2] Optional: Führt Bildvorverarbeitung durch (siehe image_preprocessing.py).")
    parser.add_argument('--create_splits', action='store_true',
                        help="[Schritt 3] Erstellt/lädt Train/Val/Test-Splits aus der Excel-Datei (configuration.LINES_XLSX_PATH).")
    parser.add_argument('--train', action='store_true',
                        help="[Schritt 4] Trainiert das Offline-Transformer-Modell.")
    parser.add_argument('--evaluate', action='store_true',
                        help="[Schritt 5] Evaluiert das beste trainierte Modell auf allen Splits.")
    # --- Optionale Überschreibungen & spezifische Argumente ---
    parser.add_argument('--lr', type=float, default=None, help=f"Überschreibt configuration.LEARNING_RATE")
    parser.add_argument('--batch_size', type=int, default=None, help=f"Überschreibt configuration.BATCH_SIZE")
    parser.add_argument('--accum_steps', type=int, default=None, help=f"Überschreibt configuration.ACCUMULATION_STEPS")
    parser.add_argument('--epochs', type=int, default=None, help=f"Überschreibt configuration.EPOCHS")
    parser.add_argument('--img_height', type=int, default=None, help=f"Überschreibt configuration.IMG_HEIGHT")
    parser.add_argument('--pad_width', type=int, default=None, help=f"Überschreibt configuration.PAD_TO_WIDTH")
    parser.add_argument('--early_stopping', type=int, default=None, help=f"Überschreibt configuration.EARLY_STOPPING_PATIENCE")
    parser.add_argument('--target_maxlen', type=int, default=None, help=f"Überschreibt configuration.TARGET_MAXLEN")
    parser.add_argument('--dropout', type=float, default=None, help=f"Überschreibt configuration.TRANSFORMER_DROPOUT")
    parser.add_argument('--embed_dim', type=int, default=None, help=f"Überschreibt configuration.TRANSFORMER_EMBED_DIM")
    # ... weitere Architektur-Params hinzufügen bei Bedarf ...
    parser.add_argument('--output_dir', type=str, default=None, help="Überschreibt Basis-Ausgabeordner.")
    parser.add_argument('--data_root', type=str, default=None, help="Überschreibt Bildordner (configuration.LINES_FOLDER).")
    parser.add_argument('--excel_path', type=str, default=None, help="Überschreibt Pfad zur Excel-Datei.")
    parser.add_argument('--load_model_for_eval', type=str, default=None, help="Pfad zu .pth Modell für --evaluate.")
    parser.add_argument('--force_new_splits', action='store_true', help="Erzwingt Neuerstellung der Splits.")


    args = parser.parse_args()
    logger.info("--- Kommandozeilenargumente ---"); [logger.info(f"  --{k}: {v}") for k, v in sorted(vars(args).items())]; logger.info("---")

    # --- Konfiguration überschreiben ---
    config_updated = []
    # Setze Attribut direkt am config-Modul
    if args.lr is not None: setattr(config, 'LEARNING_RATE', args.lr); config_updated.append("LEARNING_RATE")
    if args.batch_size is not None: setattr(config, 'BATCH_SIZE', args.batch_size); config_updated.append("BATCH_SIZE")
    if args.accum_steps is not None: setattr(config, 'ACCUMULATION_STEPS', args.accum_steps); config_updated.append("ACCUMULATION_STEPS")
    if args.epochs is not None: setattr(config, 'EPOCHS', args.epochs); config_updated.append("EPOCHS")
    if args.img_height is not None: setattr(config, 'IMG_HEIGHT', args.img_height); config_updated.append("IMG_HEIGHT")
    if args.pad_width is not None: setattr(config, 'PAD_TO_WIDTH', args.pad_width); config_updated.append("PAD_TO_WIDTH")
    if args.early_stopping is not None: setattr(config, 'EARLY_STOPPING_PATIENCE', args.early_stopping); config_updated.append("EARLY_STOPPING_PATIENCE")
    if args.target_maxlen is not None: setattr(config, 'TARGET_MAXLEN', args.target_maxlen); config_updated.append("TARGET_MAXLEN")
    if args.dropout is not None: setattr(config, 'TRANSFORMER_DROPOUT', args.dropout); config_updated.append("TRANSFORMER_DROPOUT")
    if args.embed_dim is not None: setattr(config, 'TRANSFORMER_EMBED_DIM', args.embed_dim); config_updated.append("TRANSFORMER_EMBED_DIM")
    if args.data_root is not None: setattr(config, 'LINES_FOLDER', args.data_root); config_updated.append("LINES_FOLDER")
    if args.excel_path is not None: setattr(config, 'LINES_XLSX_PATH', args.excel_path); config_updated.append("LINES_XLSX_PATH")

    # Output Dir handling MUSS VORHER erfolgen, da es Logpfad etc. beeinflusst,
    # was aber jetzt hier zentral konfiguriert wird -> Schwierig nachträglich änderbar.
    # Besser: Der Timestamp-Ordner bleibt unter "outputs", es sei denn, --output_dir gibt den *vollen* Run-Pfad an.
    if args.output_dir is not None:
        if os.path.isabs(args.output_dir): # Erlaube absolute Pfade
            full_run_path = args.output_dir
        else:
             full_run_path = os.path.join(configuration.PROJECT_ROOT, args.output_dir) # Relativ zum Projekt

        logger.warning(f"Überschreibe Ausgabeordner! Verwende direkt: {full_run_path}")
        # WICHTIG: Alle Pfade in config neu setzen und Ordner neu erstellen!
        setattr(config, 'RUN_OUTPUT_PATH', full_run_path)
        setattr(config, 'MODEL_SAVE_PATH', os.path.join(full_run_path, "models"))
        setattr(config, 'CHECKPOINT_PATH', os.path.join(full_run_path, "checkpoints"))
        setattr(config, 'RESULTS_PATH', os.path.join(full_run_path, "results"))
        setattr(config, 'PLOTS_PATH', os.path.join(full_run_path, "plots"))
        setattr(config, 'METRICS_PATH', os.path.join(full_run_path, "metrics"))
        setattr(config, 'LOGS_PATH', os.path.join(full_run_path, "logs")) # Unbenutzt, aber für Konsistenz
        configuration.PATH_CONFIG = {k: getattr(config, f"{k.upper()}_PATH") for k in ['run_output','model_save','checkpoints','results','plots','metrics']}
        try:
             logger.info(f"Erstelle überschriebenen Ausgabeordner: {full_run_path}")
             utilities.ensure_dir(configuration.RUN_OUTPUT_PATH) # Hauptordner
             for path in configuration.PATH_CONFIG.values(): utilities.ensure_dir(path) # Unterordner
             # Aktualisiere Log-Pfad im Handler, falls möglich (kompliziert, einfacher ist Neustart mit Argument)
             logger.warning("Log-Datei kann nicht dynamisch geändert werden. Ausgabe erfolgt weiterhin nach "
                           f"'{log_file_path}', oder starten Sie den Lauf erneut mit --output_dir.")
        except Exception as e:
             logger.critical(f"Konnte überschriebenen Ausgabeordner '{full_run_path}' nicht erstellen: {e}"); sys.exit(1)
        config_updated.append(f"RUN_OUTPUT_PATH (override) -> {configuration.RUN_OUTPUT_PATH}")

    if config_updated: logger.info(f"Konfiguration aktualisiert: {', '.join(config_updated)}")
    else: logger.info("Keine Konfigurations-Überschreibungen durch Argumente.")


    # --- Logging Level anpassen (falls nötig, nach dem ersten Logging) ---
    try:
        final_log_level_str = configuration.LOGGING_LEVEL.upper()
        final_log_level = getattr(logging, final_log_level_str, logging.INFO)
        logging.getLogger().setLevel(final_log_level) # Setze Level für Root Logger
        logger.info(f"Logging-Level auf {final_log_level_str} gesetzt.")
    except Exception as e:
        logger.error(f"Fehler beim Anpassen des Logging-Levels: {e}")

    # --- Rest der main() Funktion (wie vorher) ---
    # ... (Modellpfad bestimmen, Schritte ausführen) ...
    # Bestimme den Pfad zum Modell für die Evaluation
    evaluation_model_path = None
    best_model_default = os.path.join(configuration.CHECKPOINT_PATH, "best_offline_transformer_model.pth") # Korrekter Name
    if args.load_model_for_eval:
        if os.path.exists(args.load_model_for_eval): evaluation_model_path = args.load_model_for_eval; logger.info(f"Nutze Eval-Modell: {evaluation_model_path}")
        else: logger.error(f"Eval-Modell '{args.load_model_for_eval}' nicht gefunden.")
    elif os.path.exists(best_model_default): evaluation_model_path = best_model_default; logger.info(f"Nutze bestes Modell für Eval: {evaluation_model_path}")
    else: logger.warning(f"Kein Modell für Evaluation gefunden (weder spezifiziert noch Standard: {best_model_default}).")


    # --- Pipeline Schritte ---
    logger.info("\n" + "="*30 + " STARTE PIPELINE SCHRITTE " + "="*30)
    executed_steps = []

    if args.prepare_data:
        logger.info("[Schritt 1/5] Datenvorbereitung (Archive)..."); executed_steps.append("Prepare Data")
        try: dataset_preparation.extract_image_archives(); logger.info("[Schritt 1/5] OK.")
        except Exception as e: logger.exception("[Schritt 1/5] FEHLER!", exc_info=True); return

    if args.preprocess_images:
        logger.info("[Schritt 2/5] Optionale Bild-Vorverarbeitung..."); executed_steps.append("Preprocess Images")
        try: image_preprocessing.preprocess_all_image_data(); logger.info("[Schritt 2/5] OK.")
        except Exception as e: logger.exception("[Schritt 2/5] FEHLER!", exc_info=True); return

    if args.create_splits:
        logger.info("[Schritt 3/5] Erstelle/Lade/Validiere Splits..."); executed_steps.append("Create Splits")
        try:
             success = dataset_preparation.create_offline_splits(force_new=args.force_new_splits)
             if success: logger.info("[Schritt 3/5] OK.")
             else: logger.error("[Schritt 3/5] FEHLER!"); return
        except Exception as e: logger.exception("[Schritt 3/5] FEHLER!", exc_info=True); return

    if args.train:
        logger.info("[Schritt 4/5] Starte Training..."); executed_steps.append("Train Model")
        try: model_training.train_offline_transformer_model(); logger.info("[Schritt 4/5] Training beendet (Details siehe Log).")
        except Exception as e: logger.exception("[Schritt 4/5] FEHLER!", exc_info=True) # Training kann auch bei Fehler beenden

    if args.evaluate:
        logger.info("[Schritt 5/5] Starte Evaluation..."); executed_steps.append("Evaluate Model")
        if evaluation_model_path:
            try: model_evaluation.evaluate_offline_transformer(model_path_to_evaluate=evaluation_model_path); logger.info("[Schritt 5/5] Evaluation beendet.")
            except Exception as e: logger.exception("[Schritt 5/5] FEHLER!", exc_info=True)
        else: logger.error("[Schritt 5/5] Evaluation übersprungen - Kein Modell gefunden/spezifiziert.")


    # --- Abschluss ---
    logger.info("\n" + "="*30 + " PIPELINE BEENDET " + "="*30)
    if executed_steps: logger.info(f"Ausgeführte Schritte: {', '.join(executed_steps)}")
    else: logger.info("Keine Pipeline-Schritte ausgewählt.")
    total_duration = datetime.now() - pipeline_start_time
    logger.info(f"Gesamtdauer: {str(total_duration).split('.')[0]}")
    logger.info(f"Ausgabeordner: {configuration.RUN_OUTPUT_PATH}")
    logger.info("="*50)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0) # Erfolg
    except SystemExit as se:
         # sys.exit() wurde explizit aufgerufen, beende mit Code
         sys.exit(se.code)
    except Exception as global_e:
         # Versuche, den Fehler mit dem konfigurierten Logger zu loggen
         try:
             logger.critical("Pipeline mit unerwartetem Fehler auf oberster Ebene abgebrochen!", exc_info=True)
         except NameError: # Falls logger doch nicht definiert war
             print(f"[FATAL] Pipeline mit unerwartetem Fehler abgebrochen (Logger nicht verfügbar): {global_e}", file=sys.stderr)
         except Exception as log_e: # Falls das Logging selbst fehlschlägt
              print(f"[FATAL] Pipeline mit Fehler abgebrochen (Logging fehlgeschlagen): {global_e}, Logging-Fehler: {log_e}", file=sys.stderr)
         sys.exit(1) # Fehler