# Offline Handschrifterkennung (OCR)

Zwei Varianten einer Offline-OCR-Pipeline für Handschrift (Wort- bzw. Zeilenebene): **CRNN+CTC** und **Transformer**. Beide nutzen PyTorch, CTC bzw. Encoder-Decoder und unterstützen IAM-Datensätze.

---

## Projektstruktur

| Verzeichnis | Inhalt |
|-------------|--------|
| **offline_CRNN/** | CRNN + CTC für wortbasierte Offline-OCR (Wortbild → Text). |
| **Transformer/** | Transformer-basiertes Modell (Encoder-Decoder) für Zeilen-OCR. |

Beide Ordner haben dieselbe modulare Struktur mit folgenden **Dateinamen**:

### Dateinamen und Rolle

| Dateiname | Rolle |
|-----------|--------|
| `configuration.py` | Pfade, Hyperparameter, Zeichenvorrat, Seeds, Gerät. |
| `image_augmentation.py` | On-the-fly Augmentierung (Affine, Perspektive, Strichstärke). |
| `dataset_preparation.py` | IAM entpacken, ASCII→CSV (Words/Lines). |
| `image_preprocessing.py` | Bildvorverarbeitung: Deskew, Hintergrund, Denoise, Resize. |
| `ocr_model.py` | Netzarchitektur (CRNN oder Transformer) und `build_*_model()`. |
| `model_training.py` | Training-Loop, Splits, Early Stopping, History, Checkpoints. |
| `model_evaluation.py` | Finale Evaluation, Metriken, Plots, Konfusionsmatrix, HTML/CSV-Reports. |
| `model_inference.py` | Einzelbild-Inferenz: Bild + Modell → erkanntes Wort/Zeile. (nur offline_CRNN) |
| `utilities.py` | Dataset, collate_fn, Laden/Speichern, Metriken (CER, WER, BLEU, F1), CTC-Decoding. |
| `main.py` | Einstieg: CLI, Pipeline-Steuerung (prepare → preprocess → splits → train → evaluate → infer). |

---

## Voraussetzungen

- Python 3.8+
- PyTorch (mit CUDA optional)
- OpenCV (`cv2`), NumPy, Pandas, scikit-learn, tqdm
- NLTK (BLEU), editdistance
- Optional: scikit-image (`skimage.restoration` für Denoise)

```bash
pip install torch torchvision opencv-python numpy pandas scikit-learn tqdm nltk editdistance scikit-image
```

---

## Benutzung (offline_CRNN)

Alle Befehle im Ordner **offline_CRNN** ausführen (dort liegen `configuration.py`, `main.py` usw.).

### 1. Daten vorbereiten

- **CSV** mit Spalten `filename` und `label` (ein Wort pro Zeile) in `configuration.WORDS_CSV`.
- **Bilder** in `configuration.WORDS_FOLDER` (Dateinamen wie in der CSV).

Optional: IAM-Archive entpacken und ASCII→Words-CSV erzeugen:

```bash
cd offline_CRNN
python main.py --prepare_data
```

### 2. Train/Val/Test-Splits erstellen

Splits (z. B. 80/10/10) aus der Words-CSV erzeugen und als CSV speichern:

```bash
python main.py --create_splits
```

### 3. Training

```bash
python main.py --train
```

Optionale Parameter: `--lr`, `--batch_size`, `--epochs`, `--train_split`, `--val_split`, `--test_split`, `--model_version`, `--checkpoint`.

- Bestes Modell (nach Val-Loss): `checkpoints/<MODEL_VERSION>/best_crnn_model.pth`
- Logs: `logs/`, Ergebnisse/Plots: `results/`, Metriken: `metrics/`

### 4. Evaluation

Beste Modellversion auf Train/Val/Test auswerten, Metriken und Plots speichern:

```bash
python main.py --evaluate
```

Mit fester Modellversion:

```bash
python main.py --evaluate --model_version 20250130-120000
```

Optional: `--checkpoint /pfad/zu/modell.pth` (wird nach `checkpoints/<VERSION>/best_crnn_model.pth` kopiert).

### 5. Inferenz auf einem Bild

```bash
python main.py --infer /pfad/zum/wortbild.png
```

Verwendet automatisch `checkpoints/<MODEL_VERSION>/best_crnn_model.pth`. Bei Bedarf zuerst `--model_version` oder `--checkpoint` setzen.

---

## Benutzung (Transformer)

Struktur analog zu **offline_CRNN**, aber:

- **Config**: andere Pfade (z. B. `offline_prepared_train/val/test.csv`, Zeilenbilder).
- **Modell**: Transformer-Encoder-Decoder statt CRNN+CTC.

Typischer Ablauf:

```bash
cd Transformer
python main.py --create_splits   # falls CSV/Zeilen vorhanden
python main.py --train
python main.py --evaluate
python main.py --infer /pfad/zu/zeilenbild.png
```

Konfiguration und genaue Datenformate in `Transformer/configuration.py` anpassen.

---

## Konfiguration (offline_CRNN)

In **offline_CRNN/configuration.py** anpassen:

- **WORDS_CSV** / **WORDS_FOLDER**: Pfad zur Labels-CSV und zum Bildordner.
- **TRAIN_SPLIT**, **VAL_SPLIT**, **TEST_SPLIT**: Anteile (Summe = 1.0).
- **IMG_HEIGHT**, **PAD_TO_WIDTH**, **MAX_IMG_WIDTH**: Bildgröße für das Netz.
- **LEARNING_RATE**, **BATCH_SIZE**, **EPOCHS**, **OPTIMIZER**, **SCHEDULER**.
- **CHAR_LIST** / **NUM_CLASSES**: Zeichenvorrat (inkl. CTC-Blank).
- **EARLY_STOPPING_PATIENCE**: Epochen ohne Verbesserung vor Abbruch.

---

## Ausgaben (offline_CRNN)

- **checkpoints/\<VERSION>/**: `best_crnn_model.pth`
- **results/\<VERSION>/**: Trainingskurven, pro Split CER/WER-Plots, Konfusionsmatrix, Beispielvorhersagen (HTML/PNG), `*_all_predictions.csv`
- **metrics/\<VERSION>/**: JSON-Metriken (train/val/test, evaluation_summary, final_summary)
- **logs/**: Training-Logdateien

---

## Kurzreferenz CLI (offline_CRNN)

| Aktion | Befehl |
|--------|--------|
| Daten vorbereiten | `python main.py --prepare_data` |
| Splits erstellen | `python main.py --create_splits` |
| Trainieren | `python main.py --train` |
| Evaluieren | `python main.py --evaluate` |
| Einzelbild-Inferenz | `python main.py --infer <bild.png>` |
| Modellversion wählen | `--model_version YYYYMMDD-HHMMSS` |
| Checkpoint verwenden | `--checkpoint /pfad/zu/modell.pth` |
| Hilfe | `python main.py --help` |

---

## Lizenz & Kontext

Projekt im Kontext BA Handschrifterkennung; IAM-Datensatz ggf. separat lizenzieren.
