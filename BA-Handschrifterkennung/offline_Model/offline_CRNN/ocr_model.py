# ocr_model.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import configuration

# Entfernen von logging.basicConfig, um Konflikte zu vermeiden
# logging.basicConfig(level=configuration.LOGGING_LEVEL, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

logger = logging.getLogger(__name__)

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        output = self.linear(rnn_output)
        return output

class CRNN(nn.Module):
    def __init__(self, num_classes=configuration.NUM_CLASSES, in_channels=1):
        super(CRNN, self).__init__()

        # Feature-Extraktion (CNN)
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # => H/2, W/2

            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # => H/4, W/4

            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1)),  # => H/8, W/4

            # Layer 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1))  # => H/16, W/4
        )

        # Berechne Feature-Dimensionen
        self.feature_height = configuration.IMG_HEIGHT // 16  # Nach 4x MaxPooling in Höhe
        self.feature_width = configuration.PAD_TO_WIDTH // 4    # Nach 2x MaxPooling in Breite
        self.lstm_input_size = 512 * self.feature_height

        # Bidirektionale LSTM (2 Schichten)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(self.lstm_input_size, 256, 512),  # Ausgabegröße auf 512 erhöhen
            BidirectionalLSTM(512, 256, 512)  # Konsistent mit der erwarteten Eingabegröße
        )

        # Fully Connected -> Anzahl Klassen
        self.fc = nn.Linear(512, num_classes)  # 512 wegen bidirektional (256*2)

    def forward(self, x):
        # CNN Feature-Extraktion
        conv = self.cnn(x)  # (batch, channels, height, width)
        logger.debug(f"After CNN: {conv.size()}")  # Debug-Ausgabe

        # Reshape für LSTM
        batch, channels, height, width = conv.size()
        conv = conv.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        conv = conv.reshape(batch, width, channels * height)  # (batch, width, 512 * feature_height)
        logger.debug(f"After Reshape: {conv.size()}")  # Debug-Ausgabe

        # RNN
        rnn_output = self.rnn(conv)  # (batch, width, 512)
        logger.debug(f"After RNN: {rnn_output.size()}")  # Debug-Ausgabe

        # Linear
        output = self.fc(rnn_output)  # (batch, width, num_classes)
        logger.debug(f"After FC: {output.size()}")  # Debug-Ausgabe

        # Reshape für CTC
        output = output.permute(1, 0, 2)  # (width, batch, num_classes)
        logger.debug(f"Final Output: {output.size()}")  # Debug-Ausgabe

        return output


def build_crnn_model():
    try:
        model = CRNN(
            num_classes=configuration.NUM_CLASSES,
            in_channels=1  # Direkt 1 für Graustufenbilder
        )

        if not os.path.exists(configuration.MODEL_SAVE_PATH):
            os.makedirs(configuration.MODEL_SAVE_PATH)

        arch_path = os.path.join(configuration.MODEL_SAVE_PATH, "model_architecture.txt")
        with open(arch_path, 'w') as f:
            f.write(str(model))

        logger.info("CRNN-Modell (wortbasiert) erstellt und Architektur gespeichert.")
        return model

    except Exception as e:
        logger.error(f"Fehler beim Erstellen des CRNN-Modells: {e}")
        return None
