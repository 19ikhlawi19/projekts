# SchimmenSpiel (SoPra Projekt 1)

Kartenspiel **„Schwimmen“** (31) – entwickelt im Rahmen des **Software-Praktikums (SoPra)** der TU Dortmund. Kotlin-Projekt mit **Gradle**, **BGW-GUI**, JUnit5-Tests, Dokka und Detekt.

## Ausführung

```bash
./gradlew run
# Windows: gradlew.bat run
```

## Projektstruktur (Kurz)

- `src/main/kotlin/` – `Main.kt`, `entity/` (CardSuit, CardValue), `service/` (CardImageLoader), `view/` (SopraApplication, HelloScene)
- `src/test/` – Unit-Tests (z. B. CardImageLoaderTest)
- Gradle: Application-Plugin, mainClass = `MainKt`; Tests mit JUnit5, Jacoco, Detekt, Dokka

## Wichtige Links (Uni)

* Aktuelle Informationen zu diesem SoPra: https://sopra.cs.tu-dortmund.de/wiki/sopra/22d/start
* Beispielprojekt Kartenspiel War: https://sopra-gitlab.cs.tu-dortmund.de/internal/bgw-war
* Weitere Links: https://sopra.cs.tu-dortmund.de/wiki/infos/links/
