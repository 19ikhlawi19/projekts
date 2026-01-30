# GitHub Actions – Anpassung für Projekte

Diese Workflows laufen nur, wenn Dateien in den jeweiligen Projektordnern geändert werden (**Path-Filter**). Du kannst Branches, Jobs und Schritte anpassen.

---

## Übersicht der Workflows

| Datei | Projekte | Trigger (Paths) |
|-------|----------|------------------|
| **ci-kotlin.yml** | SchimmenSpiel, CableCar | `SchimmenSpiel/**`, `CableCar/**` |
| **ci-java.yml** | Auftragsverwaltung | `Auftragsverwaltung/**` |
| **ci-python.yml** | BA-Handschrifterkennung, KI-Handels-Framework | `BA-Handschrifterkennung/**`, `KI-Handels-Framework/**` |

**Nicht abgedeckt:** PC-Konfigurator (Eclipse/Xtext), Analyse_Indikatoren (Pine Script).

---

## Was du anpassen solltest

### 1. Branch-Namen

Standard: `main` und `master`. Wenn dein Standard-Branch anders heißt:

```yaml
on:
  push:
    branches: [main, dein-branch]
  pull_request:
    branches: [main, dein-branch]
```

### 2. CableCar: Secret für SoPra-Maven

CableCar braucht Abhängigkeiten vom SoPra-GitLab (ntf). Der Job **CableCar** läuft nur, wenn das Secret `SOPRA_GITLAB_TOKEN` gesetzt ist (`if: secrets.SOPRA_GITLAB_TOKEN != ''`). Ohne Secret wird der Job übersprungen (kein Fehler).

So setzt du das Secret:

1. **GitHub Repo** → **Settings** → **Secrets and variables** → **Actions**
2. **New repository secret:** Name `SOPRA_GITLAB_TOKEN`, Wert = dein GitLab Private Token

**Gradlew unter Windows:** Die Workflows konvertieren `gradlew` vor dem Build von CRLF auf LF (`sed -i 's/\r$//'`), damit der Build unter Linux nicht mit „bad interpreter“ fehlschlägt.

### 3. Java-Version (Auftragsverwaltung / Kotlin)

- **ci-java.yml:** aktuell JDK 17 (z. B. auf 11 ändern: `java-version: '11'`).
- **ci-kotlin.yml:** JDK 11 für Gradle (SchimmenSpiel/CableCar nutzen JVM 11).

### 4. Python-Version

In **ci-python.yml** steht `python-version: '3.10'`. Du kannst `3.9` oder `3.11` setzen, je nachdem was deine Projekte brauchen.

### 5. KI-Handels-Framework: TA-Lib

TA-Lib wird im Workflow **aus den Quellen** gebaut (ta-lib-0.4.0), da das Paket unter Ubuntu oft nicht als `libta-lib-dev` verfügbar ist. Wenn du TA-Lib in CI nicht brauchst, kannst du den Schritt „Install TA-Lib (from source)“ entfernen und in `requirements.txt` TA-Lib auskommentieren oder einen separaten `requirements-ci.txt` ohne TA-Lib verwenden.

### 6. Eigene Schritte (Tests, Lint, Deploy)

- **Kotlin:** Nach `./gradlew build` kannst du z. B. `./gradlew test` oder Reports hochladen.
- **Java:** Nach `javac` z. B. JUnit-Tests mit Maven/Gradle, sobald du ein Build-Tool nutzt.
- **Python:** Statt nur Import-Check z. B. `pytest BA-Handschrifterkennung/` oder `pytest KI-Handels-Framework/`, sobald Tests vorhanden sind.

### 7. Path-Filter erweitern

Wenn du z. B. bei Änderungen an der Haupt-README auch alle Projekte bauen willst, kannst du unter `paths:` z. B. `README.md` ergänzen. Standard: Nur die jeweiligen Projektordner, damit CI schnell und gezielt läuft.

---

## Workflow manuell ausführen (optional)

Falls du einen Workflow per Klick starten willst:

```yaml
on:
  workflow_dispatch:
```

in den jeweiligen Workflow unter `on:` ergänzen. Dann erscheint unter **Actions** → Workflow auswählen → **Run workflow**.

---

## Kurz: Nötige Anpassungen für dich

1. **Branch:** `main`/`master` passt lassen oder auf deinen Default-Branch anpassen.
2. **CableCar:** Secret `SOPRA_GITLAB_TOKEN` in den Repo-Secrets anlegen (oder CableCar-Job deaktivieren).
3. **Rest:** Optional JDK-, Python-Version und weitere Schritte (Tests, Lint) nach Bedarf anpassen.
