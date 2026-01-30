import os
import logging
import tarfile
import csv
import configuration
import utilities

# -----------------------------------------------------------------------------
# dataset_preparation.py
# -----------------------------------------------------------------------------

logging.basicConfig(level=configuration.LOGGING_LEVEL, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def extract_iam_data():
    """
    Entpackt die IAM-Archive ins Verzeichnis configuration.DATASET_PATH_OFFLINE.
    Nur relevant wenn IAM 
    """
    if not os.path.exists(configuration.IAM_TGZ_FOLDER):
        logger.warning("IAM_TGZ_FOLDER existiert nicht. Überspringe Entpacken.")
        return

    archives = [f for f in os.listdir(configuration.IAM_TGZ_FOLDER) if f.endswith('.tgz')]
    for arc in archives:
        arc_path = os.path.join(configuration.IAM_TGZ_FOLDER, arc)
        try:
            with tarfile.open(arc_path, 'r:gz') as tar:
                tar.extractall(path=configuration.DATASET_PATH_OFFLINE)
            logger.info(f"Entpackt: {arc} in {configuration.DATASET_PATH_OFFLINE}")
        except Exception as e:
            logger.error(f"Fehler beim Entpacken von {arc}: {e}")

def parse_iam_words(ascii_dir, output_csv):
    """
    Beispiel-Funktion:
    Parst ASCII-Dateien (words.txt) und erzeugt eine CSV (filename, label).
    => IAM Words-Struktur:
       z.B. a01-000w-01-00.png
    Dies kann je nach IAM-Version abweichen.
    """
    utilities.create_directory(os.path.dirname(output_csv))

    with open(output_csv, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['filename', 'label'])

        for root, _, files in os.walk(ascii_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as txt_f:
                        for line in txt_f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                # Wort-Annotation extrahieren => anpassen
                                # ...
                                # Dementsprechend Pfad bilden.
                                pass
        logger.info(f"IAM-Words als CSV gespeichert: {output_csv}")

def prepare_data():
    """
    Hauptfunktion zur Datenvorbereitung – hier auf Wort-Ebene.
    - Entpacken
    - ASCII lesen
    - CSV erstellen
    """
    logger.info("Starte wortbasierte Datenvorbereitung (z.B. IAM Words)...")
    # 1) Entpacken
    extract_iam_data()
    # 2) Falls ASCII-Files 
    if os.path.exists(configuration.IAM_ASCII_FOLDER):
        # parse_iam_words(configuration.IAM_ASCII_FOLDER, configuration.WORDS_CSV)
        logger.info("ASCII-Folder gefunden, um Words-CSV zu erzeugen.")
    else:
        logger.warning("IAM_ASCII_FOLDER nicht vorhanden, kann keine ASCII->Words-CSV erstellen.")
    logger.info("Datenvorbereitung (Words) abgeschlossen.")
