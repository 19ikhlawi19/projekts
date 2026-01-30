# dataset_preparation.py
import os
import logging
import tarfile
import configuration
import utilities
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from tqdm import tqdm # <<< HIER IMPORTIEREN

logger = logging.getLogger(__name__)

# Rest der Datei bleibt gleich...

# Funktion extract_image_archives bleibt unverändert...
def extract_image_archives():
    """
    Prüft, ob Bildarchive (wie lines.tgz) vorhanden sind und entpackt sie,
    wenn der Zielordner (configuration.LINES_FOLDER) leer oder nicht vorhanden ist.
    """
    tgz_folder = configuration.IAM_TGZ_FOLDER
    output_folder = configuration.LINES_FOLDER # Ziel ist der Bildordner selbst
    if not tgz_folder or not os.path.isdir(tgz_folder):
         logger.info(f"Kein TGZ-Ordner konfiguriert oder gefunden ({tgz_folder}). Überspringe Archiv-Entpacken.")
         return True
    if os.path.isdir(output_folder) and len(os.listdir(output_folder)) > 0:
        logger.info(f"Bildordner '{output_folder}' existiert bereits und enthält Dateien. Überspringe Entpacken.")
        return True
    logger.info(f"Bildordner '{output_folder}' nicht gefunden oder leer. Suche nach Archiven in '{tgz_folder}'...")
    archives_found = False
    possible_archives = ["lines.tgz", "lines.tar.gz"]
    for arc_name in possible_archives:
        arc_path = os.path.join(tgz_folder, arc_name)
        if os.path.exists(arc_path):
            archives_found = True
            logger.info(f"Entpacke Archiv: {arc_path} nach {output_folder}...")
            try:
                utilities.ensure_dir(output_folder)
                mode = "r:gz" if arc_path.endswith(".gz") else "r:"
                with tarfile.open(arc_path, mode) as tar:
                    tar.extractall(path=output_folder)
                logger.info(f"Archiv {arc_path} erfolgreich entpackt.")
                if not os.listdir(output_folder):
                    logger.warning(f"Entpacken von {arc_path} erfolgreich, aber Zielordner '{output_folder}' ist leer. Prüfen Sie Archivstruktur.")
                    # Hier nicht False zurückgeben, da es vielleicht andere Archive gibt
                # return True # <- Nicht hier return, sondern nach der Schleife prüfen
            except Exception as e:
                logger.error(f"Fehler beim Entpacken von {arc_path}: {e}")
                # Nicht unbedingt abbrechen, vielleicht sind Bilder schon da?

    if not archives_found:
        logger.warning(f"Keine relevanten Bildarchive ({possible_archives}) in '{tgz_folder}' gefunden.")
        # Trotzdem True zurückgeben, da vielleicht Bilder manuell kopiert wurden.
        # Die Existenz wird später geprüft.
    return True


def validate_data_and_filter(all_relative_paths, all_labels_encoded):
    """
    Überprüft Bildexistenz und Label-Länge für die übergebenen Listen.
    Gibt gefilterte Listen zurück.
    """
    valid_paths = []
    valid_labels = []
    img_root = configuration.LINES_FOLDER
    max_len = configuration.TARGET_MAXLEN
    skipped_path = 0
    skipped_len = 0

    logger.info(f"Validiere {len(all_relative_paths)} Samples (Bildexistenz, Ziellänge <= {max_len})...")
    if not all_relative_paths: return [], []

    vectorizer = utilities.VectorizeChar()

    # Die Schleife bleibt unverändert, da der Import oben korrigiert wurde
    for i, rel_path in enumerate(tqdm(all_relative_paths, desc="Validiere Daten", unit=" Sample")):
        abs_path = os.path.join(img_root, rel_path).replace("\\", "/")
        if not os.path.exists(abs_path):
            if skipped_path < 10 or skipped_path % 100 == 0:
                 logger.warning(f"Bild nicht gefunden bei Validierung: {abs_path}. Sample #{i} wird übersprungen (Fehler #{skipped_path+1})")
            skipped_path += 1; continue

        encoded_label = all_labels_encoded[i]
        if not isinstance(encoded_label, list) or len(encoded_label) < 2:
            logger.warning(f"Ungültiges kodiertes Label für {rel_path}. Sample #{i} übersprungen.")
            skipped_len += 1; continue
        if len(encoded_label) > max_len:
             logger.warning(f"Kodiertes Label für {rel_path} überschreitet max. Länge ({len(encoded_label)} > {max_len}). Sample #{i} übersprungen.")
             skipped_len += 1; continue

        valid_paths.append(rel_path)
        valid_labels.append(encoded_label)

    logger.info(f"Validierung abgeschlossen:")
    logger.info(f"  Gültige Samples: {len(valid_paths)}")
    logger.info(f"  Übersprungen (Pfad nicht gefunden): {skipped_path}")
    logger.info(f"  Übersprungen (Label zu lang/ungültig): {skipped_len}")
    if skipped_path > 0: logger.error(f"KRITISCH: {skipped_path} Bilder konnten nicht gefunden werden!")
    if len(valid_paths) == 0: logger.error("Keine validen Samples nach Filterung übrig!")

    return valid_paths, valid_labels


# --- NEUE Struktur für create_offline_splits ---
def create_offline_splits(force_new=False):
    """
    Orchestriert die Erstellung der Train/Val/Test Split CSV-Dateien.
    Liest IMMER die primäre Datenquelle (Excel) und erstellt neue Splits,
    überschreibt alte, wenn force_new=True oder wenn sie nicht existieren.
    """
    train_file = configuration.TRAIN_LABELS_LINES_FILE
    val_file = configuration.VAL_LABELS_LINES_FILE
    test_file = configuration.TEST_LABELS_LINES_FILE
    xlsx_file = configuration.LINES_XLSX_PATH
    image_root = configuration.LINES_FOLDER

    # --- Prüfen, ob Splits neu erstellt werden MÜSSEN ---
    splits_exist = all(os.path.exists(f) for f in [train_file, val_file, test_file])
    if splits_exist and not force_new:
        logger.info(f"Split-Dateien existieren bereits ({train_file} etc.). Überspringe Neuerstellung.")
        logger.info("Verwenden Sie --force_new_splits (in main.py) oder löschen Sie die Dateien manuell, um sie neu zu erstellen.")
        return True

    # --- Lösche alte Splits, falls force_new ---
    if force_new:
        logger.warning("Erzwinge Neuerstellung der Splits. Lösche alte Dateien (falls vorhanden)...")
        for f_path in [train_file, val_file, test_file]:
            if os.path.exists(f_path):
                try: os.remove(f_path); logger.info(f"  - Gelöscht: {f_path}")
                except OSError as del_e: logger.error(f"  - Fehler Löschen {f_path}: {del_e}")

    # --- Lade Primärquelle (Excel) ---
    logger.info(f"Lade Primärdaten aus Excel: {xlsx_file}...")
    try:
        try: df = pd.read_excel(xlsx_file, engine='openpyxl')
        except ImportError: # Falls openpyxl nicht installiert ist
             logger.warning("openpyxl nicht gefunden, versuche xlrd (pip install xlrd).")
             df = pd.read_excel(xlsx_file, engine='xlrd')
        except Exception as pd_err:
            logger.error(f"Fehler beim Lesen der Excel-Datei {xlsx_file} mit pandas: {pd_err}")
            return False
        logger.info(f"Excel geladen: {len(df)} Zeilen gefunden.")
    except ImportError:
         logger.error("Kein pandas Excel-Reader (weder openpyxl noch xlrd) gefunden. Installieren Sie z.B. 'pip install openpyxl'.")
         return False
    except FileNotFoundError:
         logger.error(f"Excel-Datei nicht gefunden: {xlsx_file}"); return False


    # Finde Spaltennamen flexibel
    path_col, label_col = None, None
    possible_path_cols = ['file_path', 'filename', 'path', 'image']
    possible_label_cols = ['transcript', 'label', 'text', 'ground_truth']
    df_cols_lower = {col.lower(): col for col in df.columns}
    for pcol in possible_path_cols:
         if pcol in df_cols_lower: path_col = df_cols_lower[pcol]; break
    for lcol in possible_label_cols:
         if lcol in df_cols_lower: label_col = df_cols_lower[lcol]; break
    if not path_col or not label_col:
        logger.error(f"Benötigte Spalten (Pfad, Label) nicht in Excel gefunden: {list(df.columns)}"); return False
    logger.info(f"Verwende Spalten: Path='{path_col}', Label='{label_col}'.")

    # --- Filtere, Kodiere und Validiere Daten ---
    all_relative_paths_raw = df[path_col].tolist()
    all_transcripts_raw = df[label_col].astype(str).tolist()
    vectorizer = utilities.VectorizeChar()

    logger.info(f"Bereinige Pfade, kodiere {len(all_transcripts_raw)} Transkripte und validiere...")
    all_relative_paths_enc = []
    all_labels_encoded = []
    skipped_char_errors = 0
    skipped_encoding_errors = 0
    unknown_chars_found = set() # Sammle alle unbekannten Zeichen

    for rel_path_raw, text_label in zip(all_relative_paths_raw, all_transcripts_raw):
         rel_path = utilities.clean_relative_path(rel_path_raw)
         if not rel_path or not text_label: continue

         current_unknowns = set()
         has_unknown = False
         for char in text_label:
             if char not in vectorizer.char_to_idx:
                 has_unknown = True
                 current_unknowns.add(char)
                 unknown_chars_found.add(char)

         if has_unknown:
             if skipped_char_errors < 10 or skipped_char_errors % 100 == 0:
                 logger.warning(f"Label '{text_label}' enthält unbekannte Zeichen: {sorted(list(current_unknowns))}. Sample übersprungen (Fehler #{skipped_char_errors+1}).")
             skipped_char_errors += 1; continue

         try:
             encoded = vectorizer(text_label)
             if not encoded or len(encoded) <= 2: skipped_encoding_errors += 1; continue
             all_relative_paths_enc.append(rel_path)
             all_labels_encoded.append(encoded)
         except: skipped_encoding_errors += 1; continue

    logger.info(f"Nach Kodierung/Zeichenprüfung: {len(all_labels_encoded)} Samples übrig.")
    if skipped_char_errors > 0:
        logger.error(f"KRITISCH: {skipped_char_errors} Samples wegen unbekannter Zeichen übersprungen!")
        logger.error(f"  Gefundene unbekannte Zeichen: {sorted(list(unknown_chars_found))}")
        logger.error(f"  BITTE configuration.py -> CHAR_LIST entsprechend anpassen!")

    # Validiere Bildexistenz und Länge
    valid_paths, valid_labels = validate_data_and_filter(all_relative_paths_enc, all_labels_encoded)

    if not valid_paths:
        logger.error("Nach Validierung (Bildexistenz/Länge) keine Samples übrig."); return False
    if len(valid_paths) < 10:
        logger.error(f"Zu wenige ({len(valid_paths)}) Samples für Split-Erstellung."); return False

    # --- Führe Split durch ---
    logger.info("Führe Train/Val/Test Split durch...")
    X, y = valid_paths, valid_labels
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []
    train_s, val_s, test_s = configuration.TRAIN_SPLIT, configuration.VAL_SPLIT, configuration.TEST_SPLIT

    try:
        # Test Split
        if test_s > 0 and len(X) > 1:
            test_size_abs = max(1, int(test_s * len(X))) # Mindestens 1 Sample
            X_rem, X_test, y_rem, y_test = train_test_split(X, y, test_size=test_size_abs, random_state=configuration.RANDOM_SEED, shuffle=True)
        else: X_rem, y_rem = X, y; X_test, y_test = [], []

        # Val Split vom Rest
        if val_s > 0 and len(X_rem) > 1:
            # Berechne Val-Größe relativ zur *Gesamtmenge* und ziehe sie vom Rest ab
            val_size_abs = max(1, int(val_s * len(X)))
            # Stelle sicher, dass Val nicht größer als der Rest ist
            val_size_abs = min(val_size_abs, len(X_rem) -1 if len(X_rem) > 1 else 0) # Mindestens 1 für Train übrig lassen
            if val_size_abs > 0:
                X_train, X_val, y_train, y_val = train_test_split(X_rem, y_rem, test_size=val_size_abs, random_state=configuration.RANDOM_SEED, shuffle=True)
            else: # Nicht genug für Val-Split
                 X_train, y_train = X_rem, y_rem; X_val, y_val = [], []
        else: X_train, y_train = X_rem, y_rem; X_val, y_val = [], []

        logger.info(f"Finale Split-Größen: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        total_split_size = len(X_train) + len(X_val) + len(X_test)
        if total_split_size != len(valid_paths):
            logger.warning(f"Summe der Splits ({total_split_size}) != Anzahl validierter Samples ({len(valid_paths)}). Möglicher Rundungsfehler?")

        # --- Speichere die neuen Splits als CSV ---
        success = True
        if X_train: success &= utilities.save_split_data(X_train, y_train, 'train', vectorizer)
        else: logger.warning("Trainings-Split ist leer! Nichts zu speichern.")
        if X_val: success &= utilities.save_split_data(X_val, y_val, 'val', vectorizer)
        else: logger.info("Validierungs-Split ist leer (oder wurde nicht angefordert).")
        if X_test: success &= utilities.save_split_data(X_test, y_test, 'test', vectorizer)
        else: logger.info("Test-Split ist leer (oder wurde nicht angefordert).")

        if success: logger.info("Neue Splits erfolgreich gespeichert.")
        else: logger.error("Fehler beim Speichern der neuen Splits!"); return False

        return True

    except Exception as split_e:
        logger.exception(f"Fehler bei Split-Erstellung: {split_e}", exc_info=True)
        return False


def prepare_offline_data(force_splits=False):
    """ Hauptfunktion: Entpackt Archive und erstellt/validiert Splits. """
    logger.info("Starte Datenvorbereitung (Offline Transformer)...")
    # Schritt 1: Bilder sicherstellen
    if not extract_image_archives(): logger.warning("Problem beim Entpacken/Finden der Bilder.")
    # Schritt 2: Excel prüfen
    if not os.path.exists(configuration.LINES_XLSX_PATH):
        logger.error(f"Primäre Datenquelle {configuration.LINES_XLSX_PATH} nicht gefunden."); return False
    # Schritt 3: Splits erstellen/validieren
    if not create_offline_splits(force_new=force_splits):
         logger.error("Fehler bei der Erstellung der Datensplits."); return False
    logger.info("Datenvorbereitung (Splits) erfolgreich abgeschlossen.")
    return True

if __name__ == "__main__":
    import argparse
    # Sicherstellen, dass das Logging konfiguriert ist, bevor es verwendet wird
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=logging.INFO) # Minimal-Konfiguration
         logger = logging.getLogger(__name__) # Logger für dieses Skript holen

    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help="Erzwinge Neuerstellung der Splits.")
    args = parser.parse_args()

    logger.info("data_preparation.py wird direkt ausgeführt.")
    prepare_offline_data(force_splits=args.force)