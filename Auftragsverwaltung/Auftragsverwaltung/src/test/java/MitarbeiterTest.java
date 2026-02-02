import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit-Tests für {@link Mitarbeiter}.
 * Deckt Getter/Setter, "null"-String-Verhalten und bildBase64 ab.
 */
@DisplayName("Mitarbeiter")
class MitarbeiterTest {

    private Mitarbeiter mitarbeiter;

    @BeforeEach
    void setUp() {
        mitarbeiter = new Mitarbeiter();
    }

    @Nested
    @DisplayName("Getter und Setter")
    class GetterSetter {

        @Test
        @DisplayName("id wird korrekt gesetzt und gelesen")
        void id() {
            mitarbeiter.setId(10);
            assertEquals(10, mitarbeiter.getId());
        }

        @Test
        @DisplayName("name, beruf, einstellungsdatum, jahresgehalt, auftraege")
        void textfelder() {
            mitarbeiter.setName("Hans Meier");
            mitarbeiter.setBeruf("Maurer");
            mitarbeiter.setEinstellungsdatum("15.03.2020");
            mitarbeiter.setJahresgehalt("45000");
            mitarbeiter.setAuftraege("3");
            assertEquals("Hans Meier", mitarbeiter.getName());
            assertEquals("Maurer", mitarbeiter.getBeruf());
            assertEquals("15.03.2020", mitarbeiter.getEinstellungsdatum());
            assertEquals("45000", mitarbeiter.getJahresgehalt());
            assertEquals("3", mitarbeiter.getAuftraege());
        }

        @Test
        @DisplayName("bildBase64 wird unverändert gespeichert und zurückgegeben")
        void bildBase64() {
            mitarbeiter.setBildBase64("YmFzZTY0");
            assertEquals("YmFzZTY0", mitarbeiter.getBildBase64());
        }
    }

    @Nested
    @DisplayName("Verhalten bei String \"null\" (wie aus Excel/CSV)")
    class NullStringVerhalten {

        @Test
        @DisplayName("name = \"null\" wird als leerer String zurückgegeben")
        void nameNullString() {
            mitarbeiter.setName("null");
            assertEquals("", mitarbeiter.getName());
        }

        @Test
        @DisplayName("beruf = \"null\" wird als leerer String zurückgegeben")
        void berufNullString() {
            mitarbeiter.setBeruf("null");
            assertEquals("", mitarbeiter.getBeruf());
        }

        @Test
        @DisplayName("einstellungsdatum und jahresgehalt = \"null\" werden als leer zurückgegeben")
        void datumGehaltNullString() {
            mitarbeiter.setEinstellungsdatum("null");
            mitarbeiter.setJahresgehalt("null");
            assertEquals("", mitarbeiter.getEinstellungsdatum());
            assertEquals("", mitarbeiter.getJahresgehalt());
        }

        @Test
        @DisplayName("auftraege = \"null\" wird als leerer String zurückgegeben")
        void auftraegeNullString() {
            mitarbeiter.setAuftraege("null");
            assertEquals("", mitarbeiter.getAuftraege());
        }
    }

    @Nested
    @DisplayName("Aufträge-Format (wie in UI: Semikolon-getrennte IDs)")
    class AuftraegeFormat {

        @Test
        @DisplayName("Mehrere Auftrags-IDs werden gespeichert und zurückgegeben")
        void mehrereIds() {
            mitarbeiter.setAuftraege("1;2;3");
            assertEquals("1;2;3", mitarbeiter.getAuftraege());
        }

        @Test
        @DisplayName("Einzelne ID")
        void einzelneId() {
            mitarbeiter.setAuftraege("5");
            assertEquals("5", mitarbeiter.getAuftraege());
        }
    }

    @Nested
    @DisplayName("Jahresgehalt als String (wie in Model validiert)")
    class Jahresgehalt {

        @Test
        @DisplayName("Ganzzahl und Dezimalzahl werden gespeichert")
        void zahlen() {
            mitarbeiter.setJahresgehalt("45000");
            assertEquals("45000", mitarbeiter.getJahresgehalt());
            mitarbeiter.setJahresgehalt("48000.50");
            assertEquals("48000.50", mitarbeiter.getJahresgehalt());
        }
    }
}
