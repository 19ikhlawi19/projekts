package ExcelHandler;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit-Tests für {@link ExcelWerkzeug}.
 * Deckt istEsInt/istEsDouble, Excel-Spaltenumrechnung und sucheTreffer ab – zentral für ExcelLeser/ExcelSchreiber.
 */
@DisplayName("ExcelWerkzeug")
class ExcelWerkzeugTest {

    @Nested
    @DisplayName("istEsInt")
    class IstEsInt {

        @Test
        @DisplayName("null gibt false")
        void nullWert() {
            assertFalse(ExcelWerkzeug.istEsInt(null));
        }

        @ParameterizedTest
        @ValueSource(strings = { "0", "1", "42", "-10", "999999" })
        @DisplayName("Gültige Integer-Strings")
        void gueltig(String s) {
            assertTrue(ExcelWerkzeug.istEsInt(s));
        }

        @ParameterizedTest
        @ValueSource(strings = { "", "abc", "12.5", "1,5", "1a", " 1 ", "1 " })
        @DisplayName("Ungültige Strings")
        void ungueltig(String s) {
            assertFalse(ExcelWerkzeug.istEsInt(s));
        }
    }

    @Nested
    @DisplayName("istEsDouble")
    class IstEsDouble {

        @Test
        @DisplayName("null gibt false")
        void nullWert() {
            assertFalse(ExcelWerkzeug.istEsDouble(null));
        }

        @ParameterizedTest
        @ValueSource(strings = { "0", "1", "1.5", "42.0", "-10.25", "0.1" })
        @DisplayName("Gültige Double-Strings")
        void gueltig(String s) {
            assertTrue(ExcelWerkzeug.istEsDouble(s));
        }

        @ParameterizedTest
        @ValueSource(strings = { "", "abc", "1,5", "1a" })
        @DisplayName("Ungültige Strings")
        void ungueltig(String s) {
            assertFalse(ExcelWerkzeug.istEsDouble(s));
        }
    }

    @Nested
    @DisplayName("excelIntZuSpalte (Zahl → Excel-Spaltenbuchstabe)")
    class ExcelIntZuSpalte {

        @Test
        @DisplayName("0 gibt leeren String")
        void nullGibtLeer() {
            assertEquals("", ExcelWerkzeug.excelIntZuSpalte(0));
        }

        @ParameterizedTest
        @CsvSource({ "1, A", "2, B", "26, Z", "27, AA", "52, AZ", "702, ZZ" })
        @DisplayName("Standard-Spalten wie in Excel")
        void standardSpalten(int num, String erwartet) {
            assertEquals(erwartet, ExcelWerkzeug.excelIntZuSpalte(num));
        }
    }

    @Nested
    @DisplayName("excelSpalteZuInt (Excel-Spaltenbuchstabe → Zahl)")
    class ExcelSpalteZuInt {

        @ParameterizedTest
        @CsvSource({ "A, 1", "B, 2", "Z, 26", "AA, 27", "AZ, 52", "ZZ, 702" })
        @DisplayName("Standard-Spalten")
        void standardSpalten(String col, int erwartet) {
            assertEquals(erwartet, ExcelWerkzeug.excelSpalteZuInt(col));
        }

        @Test
        @DisplayName("Ungültiger Buchstabe oder leer liefert -1 bzw. 0")
        void ungueltig() {
            assertTrue(ExcelWerkzeug.excelSpalteZuInt("") <= 0);
            assertEquals(-1, ExcelWerkzeug.excelSpalteZuInt("1"));
            assertEquals(-1, ExcelWerkzeug.excelSpalteZuInt("?"));
        }
    }

    @Nested
    @DisplayName("Excel-Spalten Roundtrip (wie in ExcelSchreiber/ExcelLeser)")
    class Roundtrip {

        @Test
        @DisplayName("1 bis 100: Int → Spalte → Int")
        void roundtrip1Bis100() {
            for (int n = 1; n <= 100; n++) {
                String col = ExcelWerkzeug.excelIntZuSpalte(n);
                int back = ExcelWerkzeug.excelSpalteZuInt(col);
                assertEquals(n, back, "Roundtrip für n=" + n + " Spalte=" + col);
            }
        }

        @Test
        @DisplayName("Typische Spalten A, B, …, Z, AA, AB")
        void typischeSpalten() {
            for (int n = 1; n <= 28; n++) {
                String col = ExcelWerkzeug.excelIntZuSpalte(n);
                assertEquals(n, ExcelWerkzeug.excelSpalteZuInt(col));
            }
        }
    }

    @Nested
    @DisplayName("sucheTreffer (Regex-Gruppen, wie in ExcelLeser)")
    class SucheTreffer {

        @Test
        @DisplayName("Datum-Pattern dd.MM.yyyy findet Gruppen")
        void datumPattern() {
            Pattern p = Pattern.compile("(\\d+)\\.(\\d+)\\.(\\d+)");
            String[] treffer = ExcelWerkzeug.sucheTreffer(p, "Datum: 01.02.2024 und 15.12.2023");
            assertNotNull(treffer);
            assertTrue(treffer.length >= 3);
            assertEquals("01", treffer[0]);
            assertEquals("02", treffer[1]);
            assertEquals("2024", treffer[2]);
        }

        @Test
        @DisplayName("Kein Treffer gibt leeres Array")
        void keinTreffer() {
            Pattern p = Pattern.compile("(xyz)");
            String[] treffer = ExcelWerkzeug.sucheTreffer(p, "abc def");
            assertNotNull(treffer);
            assertEquals(0, treffer.length);
        }

        @Test
        @DisplayName("Einzelner Treffer mit Gruppe")
        void einzelnerTreffer() {
            Pattern p = Pattern.compile("id=\"(\\d+)\"");
            String[] treffer = ExcelWerkzeug.sucheTreffer(p, "id=\"42\"");
            assertNotNull(treffer);
            assertEquals(1, treffer.length);
            assertEquals("42", treffer[0]);
        }
    }
}
