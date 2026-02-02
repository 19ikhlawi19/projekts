import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit-Tests für {@link Bauauftrag}.
 * Deckt Getter/Setter, toString und Randfälle ab.
 */
@DisplayName("Bauauftrag")
class BauauftragTest {

    private Bauauftrag auftrag;

    @BeforeEach
    void setUp() {
        auftrag = new Bauauftrag();
    }

    @Nested
    @DisplayName("Getter und Setter")
    class GetterSetter {

        @Test
        @DisplayName("id wird korrekt gesetzt und gelesen")
        void id() {
            auftrag.setId(42);
            assertEquals(42, auftrag.getId());
        }

        @Test
        @DisplayName("auftraggeber wird korrekt gesetzt und gelesen")
        void auftraggeber() {
            auftrag.setAuftraggeber("Firma Müller GmbH");
            assertEquals("Firma Müller GmbH", auftrag.getAuftraggeber());
        }

        @Test
        @DisplayName("ort, adresse, beschreibung Roundtrip")
        void ortAdresseBeschreibung() {
            auftrag.setOrt("München");
            auftrag.setAdresse("Hauptstraße 1");
            auftrag.setBeschreibung("Neubau Einfamilienhaus");
            assertEquals("München", auftrag.getOrt());
            assertEquals("Hauptstraße 1", auftrag.getAdresse());
            assertEquals("Neubau Einfamilienhaus", auftrag.getBeschreibung());
        }

        @Test
        @DisplayName("startdatum und enddatum im Projektformat dd.MM.yyyy")
        void datumsfelder() {
            auftrag.setStartdatum("01.01.2024");
            auftrag.setEnddatum("30.06.2024");
            assertEquals("01.01.2024", auftrag.getStartdatum());
            assertEquals("30.06.2024", auftrag.getEnddatum());
        }
    }

    @Nested
    @DisplayName("toString")
    class ToString {

        @Test
        @DisplayName("Format: Auftraggeber - Beschreibung")
        void format() {
            auftrag.setAuftraggeber("Bauherr Schmidt");
            auftrag.setBeschreibung("Anbau Garage");
            assertEquals("Bauherr Schmidt - Anbau Garage", auftrag.toString());
        }

        @Test
        @DisplayName("Leere Strings werden ausgegeben")
        void leereStrings() {
            auftrag.setAuftraggeber("");
            auftrag.setBeschreibung("");
            assertEquals(" - ", auftrag.toString());
        }
    }

    @Nested
    @DisplayName("Vollständiger Roundtrip")
    class Roundtrip {

        @Test
        @DisplayName("Alle Felder gesetzt und gelesen")
        void alleFelder() {
            auftrag.setId(1);
            auftrag.setAuftraggeber("AG");
            auftrag.setOrt("Berlin");
            auftrag.setAdresse("Test 1");
            auftrag.setBeschreibung("Beschreibung");
            auftrag.setStartdatum("01.02.2024");
            auftrag.setEnddatum("28.02.2024");

            assertEquals(1, auftrag.getId());
            assertEquals("AG", auftrag.getAuftraggeber());
            assertEquals("Berlin", auftrag.getOrt());
            assertEquals("Test 1", auftrag.getAdresse());
            assertEquals("Beschreibung", auftrag.getBeschreibung());
            assertEquals("01.02.2024", auftrag.getStartdatum());
            assertEquals("28.02.2024", auftrag.getEnddatum());
        }
    }
}
