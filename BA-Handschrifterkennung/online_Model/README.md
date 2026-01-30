# Online Handschrifterkennung (BA-Handschrifterkennung)

Projekt zur **Online-Handschrifterkennung** auf Basis des IAM-OnDB-Datensatzes. Es stehen zwei Modellvarianten zur Verfügung: **CRNN+CTC** und **Transformer**.

---

## Projektstruktur

### `src_CRNN` – Pipeline CRNN+CTC

| Datei | Beschreibung |
|-------|--------------|
| **crnn_configuration.py** | Pfade, Hyperparameter, Zeichensatz und Geräte-Einstellungen für die CRNN-Pipeline. |
| **iam_dataset_preparation.py** | IAM-Daten: XML/TXT einlesen, Train/Val/Test-Split, Strokes → Preprocessing → Merkmale → `.bin` und Excel-Manifeste. |
| **stroke_preprocessing.py** | Vorverarbeitung der Striche (Slope, Origin, Slant, Höhe, Resampling, Glättung). |
| **stroke_feature_extraction.py** | Merkmalsextraktion aus Ink (Koordinaten, Richtung, Krümmung, Vicinity, Bitmap). |
| **crnn_handwriting_model.py** | CRNN-Architektur (CNN + BiLSTM + CTC-Ausgabe). |
| **crnn_model_training.py** | Training, Dataset, Collate, Early Stopping, Checkpoints. |
| **crnn_model_evaluation.py** | Evaluation auf Train/Val/Test, CER/WER/BLEU, Plots und Berichte. |
| **handwriting_utilities.py** | Verzeichnisse, Laden von `.bin`, Encode/Decode, CER/WER, Metriken, Manifeste. |
| **run_online_pipeline.py** | Einstieg: Schritte „Daten vorbereiten“, „Trainieren“, „Evaluieren“ mit CLI-Argumenten. |
| **main.py** | Einstiegspunkt (ruft `run_online_pipeline.main()` auf). |

### `src_Transformer` – Pipeline Transformer

| Datei | Beschreibung |
|-------|--------------|
| **iam_dataset_preparation.py** | IAM-Daten vorbereiten (wie CRNN, eigene Konfiguration), Excel-Manifeste. |
| **stroke_preprocessing.py** | Strokes vorverarbeiten (inkl. optional Flip, delayed strokes). |
| **stroke_feature_extraction.py** | Merkmalsextraktion aus Ink (wie CRNN). |
| **transformer_handwriting_model.py** | Transformer Encoder-Decoder für Handschrift. |
| **transformer_model_training.py** | Training mit Warmup/Decay-Scheduler, TensorBoard, Early Stopping. |
| **transformer_model_evaluation.py** | Evaluation (CER, WER, F1, BLEU, Plots). |
| **transformer_utilities.py** | Vectorizer, IAM-Dataset, Collate, Metriken, Projektpfade. |
| **train.py** | Einstieg Training (ruft `transformer_model_training.main()` mit Argumenten). |

---

## Voraussetzungen

- **Python** (z. B. 3.9+)
- **PyTorch** (mit CUDA für GPU-Training)
- Abhängigkeiten: `pandas`, `openpyxl`, `scipy`, `scikit-learn`, `tqdm`, `matplotlib`, `seaborn`, `nltk`, `Levenshtein`, ggf. `tensorboard`

IAM-OnDB-Daten (Ordner `ascii`, `lineStrokes` bzw. Archive) in einen Datenordner legen (Standard: `data/` relativ zum Projekt-Root).

---

## Benutzung

### CRNN+CTC-Pipeline (`src_CRNN`)

**Arbeitsverzeichnis:** Aus dem Projekt-Root ausführen, sodass `src_CRNN` im Pfad liegt, z. B.:

```bash
cd /pfad/zum/online_Model
python src_CRNN/run_online_pipeline.py --prepare_online_data --train_online --evaluate_online
```

Oder aus `src_CRNN` heraus:

```bash
cd src_CRNN
python run_online_pipeline.py --prepare_online_data --train_online --evaluate_online
```

**Schritte:**

1. **Daten vorbereiten** (IAM → Strokes → Preprocessing → Features → `.bin` + Manifeste):
   ```bash
   python src_CRNN/run_online_pipeline.py --prepare_online_data
   ```

2. **Modell trainieren** (CRNN+CTC):
   ```bash
   python src_CRNN/run_online_pipeline.py --train_online
   ```

3. **Modell evaluieren** (beste Checkpoint-Datei):
   ```bash
   python src_CRNN/run_online_pipeline.py --evaluate_online
   ```

Optional: Konfiguration überschreiben, z. B.:

```bash
python src_CRNN/run_online_pipeline.py --train_online --batch_size 16 --epochs 50 --lr 0.0002
```

Evaluation mit festem Modellpfad:

```bash
python src_CRNN/run_online_pipeline.py --evaluate_online --load_model_for_eval outputs_online/.../checkpoints/best_online_crnn_model.pth
```

**Ausgaben (CRNN):** Unter `outputs_online/<lauf>/` (Checkpoints, Logs, Results, Metrics, Plots).

---

### Transformer-Pipeline (`src_Transformer`)

**Arbeitsverzeichnis:** Projekt-Root, damit `src_Transformer` und `data/` gefunden werden.

1. **Daten vorbereiten** (IAM → `.bin` + Excel-Manifeste in `data/`):
   ```bash
   cd /pfad/zum/online_Model
   python src_Transformer/iam_dataset_preparation.py
   ```
   Erzeugt u. a. `data/iam_train.xlsx`, `data/iam_val.xlsx`, `data/iam_test.xlsx`.

2. **Training** (Modellname ist Pflicht):
   ```bash
   python src_Transformer/train.py --model_name my_transformer
   ```
   Oder direkt:
   ```bash
   python src_Transformer/transformer_model_training.py --model_name my_transformer
   ```

   Optionen u. a.: `--batch_size`, `--epochs`, `--learning_rate`, `--embed_dim`, `--num_heads`, `--encoder_layers`, `--decoder_layers`, `--patience`, `--seed`.

3. **Evaluation** (z. B. auf Testdaten):
   ```bash
   python src_Transformer/transformer_model_evaluation.py \
     --model_path Models/<run>/my_transformer_best.pt \
     --data_path data/iam_test.xlsx \
     --output_dir Results/Evaluation
   ```
   Weitere Parameter (z. B. `--feature_dim`, `--embed_dim`, `--num_heads`, `--target_maxlen`) müssen zum trainierten Modell passen.

**Ausgaben (Transformer):** `Models/`, `Results/`, `Plots/`, `runs/` (TensorBoard).

---

## Konfiguration

- **CRNN:** Alle zentralen Einstellungen in `src_CRNN/crnn_configuration.py` (Datenordner, Manifest-Prefix, Feature-Dim, Max-Sequenzlänge, Lernrate, Epochen, Early Stopping, Zeichensatz usw.). Viele Werte lassen sich zusätzlich über die CLI von `run_online_pipeline.py` überschreiben.
- **Transformer:** Datenordner und Dateinamen in `iam_dataset_preparation.py` und in den Skripten (z. B. `DATA_DIR`, `MODEL_DIR`). Modell- und Trainingsparameter über Argumente von `train.py` bzw. `transformer_model_training.py`.

---

## Kurzüberblick Dateinamen

- **crnn_configuration** – Konfiguration CRNN-Pipeline  
- **iam_dataset_preparation** – IAM-Datensatz aufbereiten (beide Pipelines)  
- **stroke_preprocessing** – Strokes vorverarbeiten  
- **stroke_feature_extraction** – Merkmale aus Strokes  
- **crnn_handwriting_model** – CRNN-Modell  
- **crnn_model_training** – CRNN-Training  
- **crnn_model_evaluation** – CRNN-Evaluation  
- **handwriting_utilities** – Hilfsfunktionen CRNN  
- **run_online_pipeline** – CRNN-Pipeline starten  
- **transformer_handwriting_model** – Transformer-Modell  
- **transformer_model_training** – Transformer-Training  
- **transformer_model_evaluation** – Transformer-Evaluation  
- **transformer_utilities** – Hilfsfunktionen Transformer  


