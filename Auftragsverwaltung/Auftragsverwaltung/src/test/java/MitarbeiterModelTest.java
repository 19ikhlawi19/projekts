import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit-Tests für {@link MitarbeiterModel}.
 * Deckt Tabellen-API, editierbare Spalten, Validierung (Name, Beruf, Datum, Jahresgehalt) und Bild-Spalte ab.
 */
@DisplayName("MitarbeiterModel")
class MitarbeiterModelTest {

    private MitarbeiterModel model;
    private List<Mitarbeiter> liste;

    @BeforeEach
    void setUp() {
        liste = new ArrayList<>();
        liste.add(mitarbeiter(1, "Hans", "Maurer", "01.01.2020", "40000", "5"));
        liste.add(mitarbeiter(2, "Anna", "Architektin", "15.06.2021", "55000", "3"));
        model = new MitarbeiterModel(liste);
    }

    private static Mitarbeiter mitarbeiter(int id, String name, String beruf, String datum, String gehalt, String auftraege) {
        Mitarbeiter m = new Mitarbeiter();
        m.setId(id);
        m.setName(name);
        m.setBeruf(beruf);
        m.setEinstellungsdatum(datum);
        m.setJahresgehalt(gehalt);
        m.setAuftraege(auftraege);
        return m;
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
        @DisplayName("getRowCount und getColumnCount")
        void rowColumnCount() {
            assertEquals(2, model.getRowCount());
            assertEquals(6, model.getColumnCount());
        }

        @Test
        @DisplayName("Spaltennamen: Bild, Name, Beruf, Einstellungsdatum, Jahresgehalt, Aufträge")
        void getColumnName() {
            assertEquals("Bild", model.getColumnName(0));
            assertEquals("Name", model.getColumnName(1));
            assertEquals("Beruf", model.getColumnName(2));
            assertEquals("Einstellungsdatum", model.getColumnName(3));
            assertEquals("Jahresgehalt", model.getColumnName(4));
            assertEquals("Aufträge", model.getColumnName(5));
        }

        @Test
        @DisplayName("Leeres Modell hat 0 Zeilen")
        void leereListe() {
            MitarbeiterModel leeresModel = new MitarbeiterModel(new ArrayList<>());
            assertEquals(0, leeresModel.getRowCount());
            assertEquals(6, leeresModel.getColumnCount());
        }
    }

    @Nested
    @DisplayName("getValueAt")
    class GetValueAt {

        @Test
        @DisplayName("Name, Beruf, Datum, Gehalt, Aufträge werden korrekt geliefert")
        void textSpalten() {
            assertEquals("Hans", model.getValueAt(0, 1));
            assertEquals("Maurer", model.getValueAt(0, 2));
            assertEquals("01.01.2020", model.getValueAt(0, 3));
            assertEquals("40000", model.getValueAt(0, 4));
            assertEquals("5", model.getValueAt(0, 5));
        }

        @Test
        @DisplayName("Bild-Spalte: null oder leer → Leerzeichen-String für Anzeige")
        void bildSpalteLeer() {
            liste.get(0).setBildBase64(null);
            Object val = model.getValueAt(0, 0);
            assertEquals(" ", val);
        }

        @Test
        @DisplayName("Bild-Spalte: gültiges Base64 → byte-Array für ImageIcon")
        void bildSpalteMitDaten() {
            String base64 = "iVBORw0KGgo=";
            liste.get(0).setBildBase64(base64);
            Object val = model.getValueAt(0, 0);
            assertTrue(val instanceof byte[]);
            assertTrue(((byte[]) val).length > 0);
        }
    }

    @Nested
    @DisplayName("isCellEditable")
    class IsCellEditable {

        @Test
        @DisplayName("Bild (0) und Aufträge (5) sind nicht editierbar")
        void nichtEditierbar() {
            assertFalse(model.isCellEditable(0, 0));
            assertFalse(model.isCellEditable(0, 5));
        }

        @Test
        @DisplayName("Name, Beruf, Einstellungsdatum, Jahresgehalt sind editierbar")
        void editierbar() {
            assertTrue(model.isCellEditable(0, 1));
            assertTrue(model.isCellEditable(0, 2));
            assertTrue(model.isCellEditable(0, 3));
            assertTrue(model.isCellEditable(0, 4));
        }
    }

    @Nested
    @DisplayName("setValueAt – gültige Werte")
    class SetValueAtGueltig {

        @Test
        @DisplayName("Name wird aktualisiert")
        void name() {
            model.setValueAt("Neuer Name", 0, 1);
            assertEquals("Neuer Name", liste.get(0).getName());
        }

        @Test
        @DisplayName("Beruf und Einstellungsdatum werden aktualisiert")
        void berufUndDatum() {
            model.setValueAt("Polier", 0, 2);
            model.setValueAt("01.03.2022", 0, 3);
            assertEquals("Polier", liste.get(0).getBeruf());
            assertEquals("01.03.2022", liste.get(0).getEinstellungsdatum());
        }

        @Test
        @DisplayName("Jahresgehalt als Ganzzahl wird akzeptiert")
        void jahresgehaltGanzzahl() {
            model.setValueAt("48000", 0, 4);
            assertEquals("48000", liste.get(0).getJahresgehalt());
        }

        @Test
        @DisplayName("Jahresgehalt als Dezimalzahl wird akzeptiert")
        void jahresgehaltDezimal() {
            model.setValueAt("48000.50", 0, 4);
            assertEquals("48000.50", liste.get(0).getJahresgehalt());
        }
    }

    @Nested
    @DisplayName("setValueAt – Validierung (leere Felder)")
    class SetValueAtLeer {

        @Test
        @DisplayName("Leerer Name wird abgelehnt")
        void leererName() {
            model.setValueAt("", 0, 1);
            assertEquals("Hans", liste.get(0).getName());
        }

        @Test
        @DisplayName("Leerer Beruf wird abgelehnt")
        void leererBeruf() {
            model.setValueAt("", 0, 2);
            assertEquals("Maurer", liste.get(0).getBeruf());
        }

        @Test
        @DisplayName("Leeres Einstellungsdatum wird abgelehnt")
        void leeresDatum() {
            model.setValueAt("", 0, 3);
            assertEquals("01.01.2020", liste.get(0).getEinstellungsdatum());
        }

        @Test
        @DisplayName("Leeres Jahresgehalt wird abgelehnt")
        void leeresGehalt() {
            model.setValueAt("", 0, 4);
            assertEquals("40000", liste.get(0).getJahresgehalt());
        }
    }

    @Nested
    @DisplayName("setValueAt – Datums- und Zahlenvalidierung")
    class SetValueAtValidierung {

        @Test
        @DisplayName("Ungültiges Einstellungsdatum wird abgelehnt")
        void ungültigesDatum() {
            model.setValueAt("kein-datum", 0, 3);
            assertEquals("01.01.2020", liste.get(0).getEinstellungsdatum());
        }

        @Test
        @DisplayName("Jahresgehalt nicht numerisch wird abgelehnt")
        void jahresgehaltNichtZahl() {
            model.setValueAt("nicht-zahl", 0, 4);
            assertEquals("40000", liste.get(0).getJahresgehalt());
        }

        @Test
        @DisplayName("Jahresgehalt mit Buchstaben wird abgelehnt")
        void jahresgehaltMitBuchstaben() {
            model.setValueAt("45abc", 0, 4);
            assertEquals("40000", liste.get(0).getJahresgehalt());
        }
    }
}
