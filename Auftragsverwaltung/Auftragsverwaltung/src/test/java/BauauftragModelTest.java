import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit-Tests für {@link BauauftragModel}.
 * Deckt Tabellen-API, Validierung (leere Felder, Datumsformat, Start vor End) und getValueAt ab.
 */
@DisplayName("BauauftragModel")
class BauauftragModelTest {

    private BauauftragModel model;
    private List<Bauauftrag> liste;

    @BeforeEach
    void setUp() {
        liste = new ArrayList<>();
        liste.add(auftrag(1, "AG1", "Ort1", "Adr1", "Beschreibung1", "01.01.2024", "30.06.2024"));
        liste.add(auftrag(2, "AG2", "Ort2", "Adr2", "Beschreibung2", "01.07.2024", "31.12.2024"));
        model = new BauauftragModel(liste);
    }

    private static Bauauftrag auftrag(int id, String ag, String ort, String adr, String beschr, String start, String end) {
        Bauauftrag a = new Bauauftrag();
        a.setId(id);
        a.setAuftraggeber(ag);
        a.setOrt(ort);
        a.setAdresse(adr);
        a.setBeschreibung(beschr);
        a.setStartdatum(start);
        a.setEnddatum(end);
        return a;
    }

    @Nested
    @DisplayName("Struktur und getItems")
    class Struktur {

        @Test
        @DisplayName("getItems gibt die gleiche Liste zurück")
        void getItems() {
            assertSame(liste, model.getItems());
        }

        @Test
        @DisplayName("getRowCount entspricht Listenlänge")
        void getRowCount() {
            assertEquals(2, model.getRowCount());
            model.getItems().add(new Bauauftrag());
            assertEquals(3, model.getRowCount());
        }

        @Test
        @DisplayName("getColumnCount ist 6 (Auftraggeber, Ort, Adresse, Beschreibung, Startdatum, Enddatum)")
        void getColumnCount() {
            assertEquals(6, model.getColumnCount());
        }

        @Test
        @DisplayName("Spaltennamen in korrekter Reihenfolge")
        void getColumnName() {
            assertEquals("Auftraggeber", model.getColumnName(0));
            assertEquals("Ort", model.getColumnName(1));
            assertEquals("Adresse", model.getColumnName(2));
            assertEquals("Beschreibung", model.getColumnName(3));
            assertEquals("Startdatum", model.getColumnName(4));
            assertEquals("Enddatum", model.getColumnName(5));
        }

        @Test
        @DisplayName("Leeres Modell hat 0 Zeilen")
        void leereListe() {
            BauauftragModel leeresModel = new BauauftragModel(new ArrayList<>());
            assertEquals(0, leeresModel.getRowCount());
            assertEquals(6, leeresModel.getColumnCount());
        }
    }

    @Nested
    @DisplayName("getValueAt")
    class GetValueAt {

        @Test
        @DisplayName("Alle Spalten liefern die richtigen Werte")
        void alleSpalten() {
            assertEquals("AG1", model.getValueAt(0, 0));
            assertEquals("Ort1", model.getValueAt(0, 1));
            assertEquals("Adr1", model.getValueAt(0, 2));
            assertEquals("Beschreibung1", model.getValueAt(0, 3));
            assertEquals("01.01.2024", model.getValueAt(0, 4));
            assertEquals("30.06.2024", model.getValueAt(0, 5));
            assertEquals("AG2", model.getValueAt(1, 0));
        }
    }

    @Nested
    @DisplayName("isCellEditable")
    class IsCellEditable {

        @Test
        @DisplayName("Alle Zellen sind editierbar")
        void alleEditierbar() {
            assertTrue(model.isCellEditable(0, 0));
            assertTrue(model.isCellEditable(0, 5));
        }
    }

    @Nested
    @DisplayName("setValueAt – gültige Werte")
    class SetValueAtGueltig {

        @Test
        @DisplayName("Auftraggeber, Ort, Adresse, Beschreibung werden aktualisiert")
        void textfelder() {
            model.setValueAt("Neuer AG", 0, 0);
            model.setValueAt("Neuer Ort", 0, 1);
            model.setValueAt("Neue Adresse", 0, 2);
            model.setValueAt("Neue Beschreibung", 0, 3);
            assertEquals("Neuer AG", liste.get(0).getAuftraggeber());
            assertEquals("Neuer Ort", liste.get(0).getOrt());
            assertEquals("Neue Adresse", liste.get(0).getAdresse());
            assertEquals("Neue Beschreibung", liste.get(0).getBeschreibung());
        }

        @Test
        @DisplayName("Startdatum und Enddatum im Format dd.MM.yyyy werden übernommen")
        void datumsfelder() {
            model.setValueAt("15.03.2024", 0, 4);
            model.setValueAt("20.05.2024", 0, 5);
            assertEquals("15.03.2024", liste.get(0).getStartdatum());
            assertEquals("20.05.2024", liste.get(0).getEnddatum());
        }
    }

    @Nested
    @DisplayName("setValueAt – Validierung (leere Felder)")
    class SetValueAtLeer {

        @Test
        @DisplayName("Leerer Auftraggeber wird abgelehnt, Wert bleibt unverändert")
        void leererAuftraggeber() {
            model.setValueAt("", 0, 0);
            assertEquals("AG1", liste.get(0).getAuftraggeber());
        }

        @Test
        @DisplayName("Leerer Ort wird abgelehnt")
        void leererOrt() {
            model.setValueAt("", 0, 1);
            assertEquals("Ort1", liste.get(0).getOrt());
        }

        @Test
        @DisplayName("Leere Adresse wird abgelehnt")
        void leereAdresse() {
            model.setValueAt("", 0, 2);
            assertEquals("Adr1", liste.get(0).getAdresse());
        }

        @Test
        @DisplayName("Leere Beschreibung wird abgelehnt")
        void leereBeschreibung() {
            model.setValueAt("", 0, 3);
            assertEquals("Beschreibung1", liste.get(0).getBeschreibung());
        }

        @Test
        @DisplayName("Leeres Startdatum wird abgelehnt")
        void leeresStartdatum() {
            model.setValueAt("", 0, 4);
            assertEquals("01.01.2024", liste.get(0).getStartdatum());
        }

        @Test
        @DisplayName("Leeres Enddatum wird abgelehnt")
        void leeresEnddatum() {
            model.setValueAt("", 0, 5);
            assertEquals("30.06.2024", liste.get(0).getEnddatum());
        }
    }

    @Nested
    @DisplayName("setValueAt – Datumsvalidierung")
    class SetValueAtDatum {

        @Test
        @DisplayName("Ungültiges Startdatum (kein Datum) wird abgelehnt")
        void ungültigesStartdatum() {
            model.setValueAt("kein-datum", 0, 4);
            assertEquals("01.01.2024", liste.get(0).getStartdatum());
        }

        @Test
        @DisplayName("Ungültiges Enddatum wird abgelehnt")
        void ungültigesEnddatum() {
            model.setValueAt("32.13.2024", 0, 5);
            assertEquals("30.06.2024", liste.get(0).getEnddatum());
        }

        @Test
        @DisplayName("Startdatum nach bestehendem Enddatum wird abgelehnt")
        void startNachEnd() {
            model.setValueAt("31.12.2024", 0, 4);
            assertEquals("01.01.2024", liste.get(0).getStartdatum());
        }

        @Test
        @DisplayName("Startdatum gleich Enddatum ist erlaubt")
        void startGleichEnd() {
            model.setValueAt("15.06.2024", 0, 4);
            model.setValueAt("15.06.2024", 0, 5);
            assertEquals("15.06.2024", liste.get(0).getStartdatum());
            assertEquals("15.06.2024", liste.get(0).getEnddatum());
        }
    }
}
