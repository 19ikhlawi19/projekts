# Auftragsverwaltung

Eine Java-Swing-Anwendung zur Verwaltung von Bauaufträgen und Mitarbeitern. Das Projekt eignet sich als Beispiel für eine Desktop-Anwendung mit Tabellen, Kontextmenüs, Excel-/CSV-Import und -Export sowie einer übersichtlichen Oberfläche mit Tabs.

---

## Inhaltsverzeichnis

- [Funktionsumfang](#funktionsumfang)
- [Projektstruktur](#projektstruktur)
- [Datenmodell](#datenmodell)
- [Voraussetzungen](#voraussetzungen)
- [Build & Ausführung](#build--ausführung)
- [Bedienung](#bedienung)
- [Import/Export](#importexport)
- [Lizenz](#lizenz)

---

## Funktionsumfang

- **Aufträge verwalten:** Bauaufträge mit Auftraggeber, Ort, Adresse, Beschreibung sowie Start- und Enddatum anlegen, bearbeiten und löschen.
- **Mitarbeiter verwalten:** Mitarbeiter mit Name, Beruf, Einstellungsdatum, Jahresgehalt und Profilbild verwalten.
- **Aufträge zuweisen:** Mitarbeitern können mehrere Bauaufträge zugewiesen werden. Terminüberschneidungen und Einstellungsdatum werden geprüft.
- **Übersicht:** Im Tab „Übersicht“ werden Aufträge nacheinander angezeigt; zugewiesene Mitarbeiter werden pro Auftrag aufgelistet.
- **Suche:** In den Tabs „Aufträge“ und „Mitarbeiter“ kann in den Tabellen gesucht werden.
- **Import/Export:** Daten können als CSV oder Excel (.xlsx) importiert und exportiert werden. Mitarbeiterbilder werden als Base64 in CSV/Excel gespeichert.

---

## Projektstruktur

```
Auftragsverwaltung/
├── Auftragsverwaltung/
│   ├── pom.xml                      # Maven-Build (Java 11, JUnit 5)
│   ├── src/
│   │   ├── Bauauftrag.java          # Datenmodell Bauauftrag
│   │   ├── BauauftragModel.java     # Tabellenmodell für Aufträge
│   │   ├── BauauftragTableMenu.java # Kontextmenü Auftragstabelle
│   │   ├── Mitarbeiter.java         # Datenmodell Mitarbeiter
│   │   ├── MitarbeiterModel.java    # Tabellenmodell für Mitarbeiter
│   │   ├── MitarbeiterTableMenu.java# Kontextmenü Mitarbeitertabelle
│   │   ├── Hauptfenster.java        # Hauptfenster, Tabs, Menü, Einstieg main()
│   │   ├── BildRenderer.java        # Zellen-Renderer für Mitarbeiterbilder
│   │   ├── CheckboxRenderer.java    # Renderer für Auftragsauswahl-Liste
│   │   ├── Hauptfenster.form        # IntelliJ UI-Designer-Form
│   │   ├── test/java/               # Unit-Tests (JUnit 5)
│   │   │   ├── BauauftragTest.java, BauauftragModelTest.java
│   │   │   ├── MitarbeiterTest.java, MitarbeiterModelTest.java
│   │   │   └── ExcelHandler/        # ExcelWerkzeugTest, ZipDateiHelferTest
│   │   └── ExcelHandler/
│   │       ├── ExcelLeser.java      # Lesen von .xlsx (XML-basiert)
│   │       ├── ExcelSchreiber.java  # Schreiben von .xlsx
│   │       ├── ExcelWerkzeug.java   # Hilfsmethoden (Spalten, Typen)
│   │       ├── ZipDateiHelfer.java # ZIP-Verarbeitung für XLSX
│   │       └── excel_vorlage.zip    # Vorlage für neue Excel-Dateien
│   └── UML.jpg                      # Klassendiagramm (Bauauftrag, Mitarbeiter)
├── README.md
└── .gitignore
```

---

## Datenmodell

### Bauauftrag

| Attribut      | Typ    | Beschreibung                    |
|---------------|--------|----------------------------------|
| id            | int    | Eindeutige ID                   |
| auftraggeber  | String | Name des Auftraggebers         |
| ort           | String | Ort des Auftrags               |
| adresse       | String | Adresse                        |
| beschreibung  | String | Beschreibung des Auftrags      |
| startdatum    | String | Format: dd.MM.yyyy             |
| enddatum      | String | Format: dd.MM.yyyy             |

### Mitarbeiter

| Attribut         | Typ    | Beschreibung                          |
|------------------|--------|----------------------------------------|
| id               | int    | Eindeutige ID                          |
| name             | String | Name des Mitarbeiters                  |
| beruf            | String | Berufsbezeichnung                      |
| einstellungsdatum| String | Format: dd.MM.yyyy                     |
| jahresgehalt     | String | Numerisch                              |
| auftraege        | String | IDs zugewiesener Aufträge, getrennt mit „;“ |
| bildBase64       | String | Profilbild als Base64 (optional)       |

Die Zuordnung Mitarbeiter ↔ Aufträge erfolgt über die IDs in `auftraege` (z. B. `"1;3;5"`). Ein Auftrag kann nicht gelöscht werden, solange er noch einem Mitarbeiter zugewiesen ist.

---

## Voraussetzungen

- **Java:** JDK 11 oder höher (getestet mit JDK 17).
- **Maven:** 3.6+ (für Build und Tests).
- **IDE (optional):** IntelliJ IDEA oder eine andere Java-IDE; das Projekt nutzt IntelliJ UI Designer (`.form`).

Für die Anwendung werden keine externen Bibliotheken benötigt; Excel wird über direkte Verarbeitung der XLSX-XML-Struktur (ZIP) gelesen und geschrieben. Für die Unit-Tests wird JUnit 5 (über Maven) verwendet.

---

## Build & Ausführung

### Mit Maven (empfohlen)

```bash
cd Auftragsverwaltung
mvn compile
java -cp target/classes Hauptfenster
```

**Tests ausführen:**

```bash
cd Auftragsverwaltung
mvn test
```

### Mit IDE (IntelliJ IDEA)

1. Projekt öffnen: `File` → `Open` → Ordner `Auftragsverwaltung` (oder das übergeordnete Verzeichnis) wählen.
2. Als Maven-Projekt erkennen lassen (falls nicht automatisch); JDK 11+ einstellen.
3. Hauptklasse ausführen: In `Hauptfenster.java` die Methode `main(String[] args)` ausführen (grüner Pfeil oder Rechtsklick → Run).

---

## Bedienung

- **Tabs:** „Übersicht“, „Aufträge“, „Mitarbeiter“ wechseln.
- **Übersicht:** Mit „<<“ und „>>“ zwischen Aufträgen blättern; links erscheinen die dem aktuellen Auftrag zugewiesenen Mitarbeiter.
- **Aufträge:** Tabelle mit allen Aufträgen. **Rechtsklick** öffnet das Kontextmenü:
  - **Auftrag hinzufügen** / **Auftrag entfernen**
- **Mitarbeiter:** Tabelle mit allen Mitarbeitern. **Rechtsklick** öffnet das Kontextmenü:
  - **Mitarbeiter hinzufügen** / **Mitarbeiter entfernen**
  - **Bild setzen** / **Bild exportieren**
  - **Aufträge zuweisen** (Checkbox-Liste; Terminüberschneidungen und Einstellungsdatum werden geprüft).
- **Menü „Auftragsverwaltung“:** Import/Export von CSV und Excel (siehe unten).

Datumseingaben müssen im Format **dd.MM.yyyy** erfolgen. Leere Pflichtfelder und ungültige Werte werden mit Fehlermeldungen abgefangen.

---

## Import/Export

- **Export:** „Auftragsverwaltung“ → „Exportieren“ → „CSV“ oder „Excel“. Es werden zuerst alle Mitarbeiter (inkl. Bild als Base64), dann eine Leerzeile, dann alle Aufträge exportiert.
- **Import:** „Auftragsverwaltung“ → „Importieren“ → „CSV“ oder „Excel“. Bestehende Daten werden ersetzt. Das erwartete Format entspricht dem Export (Spaltenköpfe siehe Export-Datei bzw. Code in `Hauptfenster.java`).

Die Excel-Funktion nutzt das XLSX-Format (ZIP mit XML). Die Vorlage dafür liegt in `ExcelHandler/excel_vorlage.zip`.

---

## Lizenz

Dieses Projekt dient als Beispiel bzw. Referenz. Nutzung und Anpassung nach eigenen Bedürfnissen sind erlaubt. Es wird keine spezielle Lizenz vorausgesetzt; bei Weitergabe die Herkunft des Codes angeben.

---

