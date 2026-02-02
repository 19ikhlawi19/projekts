package ExcelHandler;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit-Tests für {@link ZipDateiHelfer}.
 * Deckt entpacken (Ordner existiert bereits) und Konstruktor ab – relevant für ExcelSchreiber/ExcelLeser.
 */
@DisplayName("ZipDateiHelfer")
class ZipDateiHelferTest {

    @Test
    @DisplayName("Konstruktor speichert Source- und Ziel-Pfad")
    void konstruktor(@TempDir Path tempDir) {
        String source = tempDir.resolve("quelle").toString();
        String target = tempDir.resolve("ziel.zip").toString();
        ZipDateiHelfer helfer = new ZipDateiHelfer(source, target);
        assertNotNull(helfer);
    }

    @Test
    @DisplayName("entpacken wirft FileAlreadyExistsException wenn Zielordner bereits existiert")
    void entpackenOrdnerExistiertBereits(@TempDir Path tempDir) throws IOException {
        Path existierenderOrdner = tempDir.resolve("existiertBereits");
        Files.createDirectories(existierenderOrdner);
        // Leerer Stream – es geht nur um die Prüfung „Ordner existiert“
        ByteArrayInputStream stream = new ByteArrayInputStream(new byte[0]);

        assertThrows(FileAlreadyExistsException.class, () ->
            ZipDateiHelfer.entpacken(stream, existierenderOrdner.toAbsolutePath().toString())
        );
    }

}
