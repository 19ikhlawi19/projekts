# Projekts-Interview

Sammlung von Projektbeispielen aus verschiedenen Domänen: Auftragsverwaltung (Java), Handschrifterkennung (Python/ML), KI-Handels-Framework (Python), PC-Konfigurator (Eclipse/Xtext/Cinco), CableCar & SchimmenSpiel (Kotlin). Einige Projekte stammen aus dem **Universitätskontext** (SoPra TU Dortmund, SCE/Uni Paderborn).

---

## Inhaltsverzeichnis

- [Übersicht der Projekte](#übersicht-der-projekte)
- [Universitätsprojekte (Hinweis)](#universitätsprojekte-hinweis)
- [Schnellstart pro Projekt](#schnellstart-pro-projekt)
- [Voraussetzungen (übergreifend)](#voraussetzungen-übergreifend)
- [Lizenz & Hinweise](#lizenz--hinweise)

---

## Übersicht der Projekte

| Projekt | Technologie | Kurzbeschreibung |
|--------|-------------|------------------|
| **[Auftragsverwaltung](./Auftragsverwaltung/)** | Java (Swing) | Desktop-App: Bauaufträge und Mitarbeiter verwalten, Excel/CSV-Import/Export, Kontextmenüs, Tabs. |
| **[BA-Handschrifterkennung](./BA-Handschrifterkennung/)** | Python, PyTorch | Offline- und Online-OCR (CRNN+CTC, Transformer), IAM-Datensatz, Training & Evaluation. |
| **[KI-Handels-Framework](./KI-Handels-Framework/)** | Python | Binance-Daten laden, RSI-Backtesting, Parameter-Optimierung, LSTM-Marktphasen-Klassifikation. |
| **[PC-Konfigurator](./PC-Konfigurator/)** *(Uni)* | Eclipse, Xtext, Cinco, Xtend | DSL für PC-Komponenten, Generatoren, visueller Konfigurator (PC/Mainboard) mit HTML-Export. |
| **[CableCar](./CableCar/)** *(Uni)* | Kotlin, Gradle | SoPra-Projekt: Spiel mit KI (MCTS), Netzwerk (bgw-net), Gradle-Build. |
| **[SchimmenSpiel](./SchimmenSpiel/)** *(Uni)* | Kotlin, Gradle | SoPra-Projekt: Kartenspiel „Schwimmen“ mit bgw-gui, Gradle, Tests & Dokka. |
| **[Analyse_Indikatoren](./Analyse_Indikatoren/)** | Pine Script v5 (TradingView) | Indikatoren und Strategie (ACSE) für TradingView: Confluence Score, MA-Ribbon, VWAP, Oszillatoren, Volatilität. |

Jedes Projekt hat ein eigenes **README** im jeweiligen Ordner mit Details zu Build, Ausführung und Nutzung.

---

## Universitätsprojekte (Hinweis)

Die folgenden Projekte wurden **im Rahmen der Universität** entwickelt und sind entsprechend dokumentiert:

- **CableCar** und **SchimmenSpiel** – entstanden im **Software-Praktikum (SoPra)** der TU Dortmund; Nutzung von BGW-GUI/BGW-Net und SoPra-Vorgaben (Gradle, Tests, Dokka, Detekt).
- **PC-Konfigurator** – Eclipse/Xtext/Cinco-Projekt mit DSL, Generatoren und visuellen Editoren; typischer Kontext: Lehrveranstaltungen zu modellgetriebener Entwicklung (z. B. SCE/Uni Paderborn).

Bei Weiternutzung oder Referenzierung bitte die jeweiligen Projekt-READMEs und ggf. die angegebenen SoPra-/Hochschul-Links beachten.

---

## Schnellstart pro Projekt

### Auftragsverwaltung

Quellcode liegt unter `Auftragsverwaltung/Auftragsverwaltung/src`. Von dort aus:

```bash
cd Auftragsverwaltung/Auftragsverwaltung/src
javac -encoding UTF-8 Hauptfenster.java
java Hauptfenster
```

Details: [Auftragsverwaltung/README.md](./Auftragsverwaltung/README.md)

---

### BA-Handschrifterkennung

Abhängigkeiten einmalig im Projektroot installieren:

```bash
cd BA-Handschrifterkennung
pip install -r requirements.txt
```

**Offline (CRNN oder Transformer):**

```bash
cd BA-Handschrifterkennung/offline_Model/offline_CRNN
python main.py --prepare_data --create_splits --train --evaluate
```

Transformer-Variante:

```bash
cd BA-Handschrifterkennung/offline_Model/Transformer
python main.py --create_splits --train --evaluate
```

**Online (CRNN):**

```bash
cd BA-Handschrifterkennung/online_Model
python src_CRNN/run_online_pipeline.py --prepare_online_data --train_online --evaluate_online
```

Details: [BA-Handschrifterkennung/offline_Model/README.md](./BA-Handschrifterkennung/offline_Model/README.md), [BA-Handschrifterkennung/online_Model/README.md](./BA-Handschrifterkennung/online_Model/README.md)

---

### KI-Handels-Framework

```bash
cd KI-Handels-Framework
pip install -r requirements.txt
python binance_data_downloader.py          # GUI: Binance-Daten herunterladen
python rsi_backtesting_engine.py           # nutzt CSV (z. B. nach Download)
python rsi_strategy_backtester.py          # Backtest mit RSI 30/70
python rsi_parameter_optimizer.py          # RSI-Parameter-Optimierung
python lstm_market_phase_classifier.py     # LSTM-Klassifikation (Marktphasen)
```

Details: [KI-Handels-Framework/README.md](./KI-Handels-Framework/README.md)

---

### PC-Konfigurator *(Uni)*

- Eclipse mit Xtext und Cinco einrichten.
- Repository klonen, alle Projekte als „Existing Projects“ importieren.
- Xtext-Generierung (MWE2) in `info.scce.cinco.fp.compdsl` ausführen.
- Cinco-Generierung im Produkt-Projekt ausführen.
- `prodserv.jar` in `info.scce.cinco.fp.compgen/lib/` ablegen.

Details: [PC-Konfigurator/README.md](./PC-Konfigurator/README.md)

---

### CableCar & SchimmenSpiel *(Uni)*

Unter Linux/macOS: `./gradlew run`  
Unter Windows: `gradlew.bat run`

**CableCar:**

```bash
cd CableCar
./gradlew run
# bzw. Windows: gradlew.bat run
```

**SchimmenSpiel:**

```bash
cd SchimmenSpiel
./gradlew run
# bzw. Windows: gradlew.bat run
```

Details: [CableCar/README.md](./CableCar/README.md), [SchimmenSpiel/README.md](./SchimmenSpiel/README.md)

---

### Analyse_Indikatoren

1. [TradingView](https://www.tradingview.com/) öffnen und einen Chart laden.
2. Pine Editor öffnen, gewünschte `.pine`-Datei aus dem Ordner [Analyse_Indikatoren](./Analyse_Indikatoren/) öffnen, Inhalt kopieren und einfügen.
3. „Zum Chart hinzufügen“ klicken.

Details: [Analyse_Indikatoren/README.md](./Analyse_Indikatoren/README.md)

---

## Voraussetzungen (übergreifend)

- **Java:** JDK 11+ (Auftragsverwaltung, PC-Konfigurator/Eclipse)
- **Python:** 3.8+ (KI-Handels-Framework, BA-Handschrifterkennung)
- **Kotlin/Gradle:** für CableCar und SchimmenSpiel (Gradle Wrapper vorhanden)
- **Eclipse, Xtext, Cinco:** nur für PC-Konfigurator

Projektspezifische Abhängigkeiten stehen in den jeweiligen `README.md` und ggf. `requirements.txt`.

---

## Lizenz & Hinweise

- Die einzelnen Projekte können unterschiedlich lizenziert sein; siehe jeweiliges README.
- Bei Weitergabe oder Veröffentlichung die Herkunft des Codes angeben.
- IAM-Datensatz (Handschrifterkennung) ggf. separat lizenziert nutzen.
- **Universitätsprojekte:** CableCar, SchimmenSpiel und PC-Konfigurator im Kontext der jeweiligen Lehrveranstaltungen verwenden und dokumentieren.

---

*Zuletzt aktualisiert: Januar 2026*
