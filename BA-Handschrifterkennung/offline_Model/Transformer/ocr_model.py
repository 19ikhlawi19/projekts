# model_offline_ocr.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import sys # <<< HINZUGEFÜGT

import configuration
import utilities # Für ensure_dir, falls Architektur gespeichert wird

logger = logging.getLogger(__name__)

# ============================================================================
# === EINGEFÜGTE TRANSFORMER KOMPONENTEN (aus alter model.py) ===
# ============================================================================
class PositionalEncoding(nn.Module):
    """ Fixed sinusoidal positional encoding. """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Adds positional encoding. Input: (Batch, SeqLen, Dim) """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    """ Combines Token Embedding and Positional Encoding for Decoder. """
    def __init__(self, num_vocab: int, embed_dim: int, maxlen: int, dropout_rate: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(num_vocab, embed_dim, padding_idx=0) # PAD=0
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout_rate, max_len=maxlen)
        self.embed_dim = embed_dim
        logger.info(f"TokenEmbedding: vocab={num_vocab}, embed={embed_dim}, maxlen={maxlen}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """ Input: token IDs (Batch, SeqLen). Output: (Batch, SeqLen, Dim) """
        x = self.token_emb(tokens) * math.sqrt(self.embed_dim)
        x = self.positional_encoding(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """ Single Pre-Norm Transformer Encoder layer. """
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(feed_forward_dim, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-Norm
        residual = x
        x_norm = self.layer_norm1(x)
        # Self-Attention
        attn_output, _ = self.self_attn(query=x_norm, key=x_norm, value=x_norm, key_padding_mask=key_padding_mask)
        # Add & Norm (Implicit Norm comes from next layer's pre-norm) + Dropout
        x = residual + self.dropout1(attn_output)

        # Pre-Norm
        residual = x
        x_norm = self.layer_norm2(x)
        # FFN
        ffn_output = self.ffn(x_norm)
        # Add & Norm + Dropout
        x = residual + self.dropout2(ffn_output)
        return x

class TransformerDecoderLayer(nn.Module):
    """ Single Pre-Norm Transformer Decoder layer. """
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(feed_forward_dim, embed_dim)
        )
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: torch.Tensor = None, memory_key_padding_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Masked Self-Attention (Pre-Norm)
        residual = tgt; tgt_norm = self.layer_norm1(tgt)
        self_attn_output, _ = self.self_attn(query=tgt_norm, key=tgt_norm, value=tgt_norm, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = residual + self.dropout1(self_attn_output)

        # 2. Cross-Attention (Pre-Norm)
        residual = tgt; tgt_norm = self.layer_norm2(tgt)
        cross_attn_output, _ = self.cross_attn(query=tgt_norm, key=memory, value=memory, key_padding_mask=memory_key_padding_mask)
        tgt = residual + self.dropout2(cross_attn_output)

        # 3. Feed-Forward (Pre-Norm)
        residual = tgt; tgt_norm = self.layer_norm3(tgt)
        ffn_output = self.ffn(tgt_norm)
        tgt = residual + self.dropout3(ffn_output)
        return tgt
# ============================================================================
# === ENDE EINGEFÜGTE TRANSFORMER KOMPONENTEN ===
# ============================================================================


class CNNFeatureExtractor(nn.Module):
    """
    CNN Frontend für die Offline-HTR. Extrahiert eine Merkmalssequenz aus dem Bild.
    ANGEPASST: Verwendet AdaptiveAvgPool2d für garantierte Ausgabehöhe von 1.
    """
    def __init__(self, input_channels=configuration.IMG_CHANNELS, output_feature_dim=configuration.CNN_OUTPUT_CHANNELS):
        super().__init__()
        self.final_cnn_channels = -1

        layers = []
        in_c = input_channels
        # Angepasste Konfiguration für robustere Höhenreduktion
        layer_cfg = [
            ('conv', 64, 3, 1, 1), ('bn',), ('relu',), ('pool', 2, 2),         # H/2, W/2
            ('conv', 128, 3, 1, 1), ('bn',), ('relu',), ('pool', 2, 2),        # H/4, W/4
            ('conv', 256, 3, 1, 1), ('bn',), ('relu',),
            ('conv', 256, 3, 1, 1), ('bn',), ('relu',), ('pool', (2, 1), (2, 1)), # H/8, W/4
            ('conv', 512, 3, 1, 1), ('bn',), ('relu',), ('dropout', 0.2),
            ('conv', 512, 3, 1, 1), ('bn',), ('relu',), ('pool', (2, 1), (2, 1)), # H/16, W/4
            # Letzter Conv Block behält die Feature-Dimension bei (oder leicht angepasst)
            ('conv', output_feature_dim, 3, 1, 1), ('bn',), ('relu',)
        ]
        # Korrigiere Out-Channels im letzten Conv Layer
        layer_cfg[-3] = ('conv', output_feature_dim, 3, 1, 1)

        current_h, current_w = configuration.IMG_HEIGHT, configuration.PAD_TO_WIDTH
        logger.info("--- CNN Feature Extractor Aufbau (Angepasst) ---")
        logger.info(f"Input H={current_h}, W={current_w}, C={in_c}")

        for i, cfg in enumerate(layer_cfg):
            l_type = cfg[0]
            if l_type == 'conv':
                _, out_c, k, s, p = cfg
                k_h, k_w = k if isinstance(k, tuple) else (k, k)
                s_h, s_w = s if isinstance(s, tuple) else (s, s)
                p_h, p_w = p if isinstance(p, tuple) else (p, p)
                layers.append(nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p))
                 # Berechne H/W *nach* hinzufügen
                # Note: Padding muss Tupel sein für get_conv_output_size
                # pad_h = p[0] if isinstance(p, tuple) else p
                # pad_w = p[1] if isinstance(p, tuple) else p
                # kh, kw = k[0] if isinstance(k, tuple) else k, k[1] if isinstance(k, tuple) else k
                # sh, sw = s[0] if isinstance(s, tuple) else s, s[1] if isinstance(s, tuple) else s
                # current_h = math.floor((current_h + 2*pad_h - kh) / sh + 1)
                # current_w = math.floor((current_w + 2*pad_w - kw) / sw + 1)
                # Temporäre Lösung: Annahme, dass die Architektur korrekt ist und die Dims nicht exakt berechnet werden müssen hier
                in_c = out_c
                # logger.info(f"  L{i+1}: Conv2d(k={k}, s={s}, p={p}, out_c={out_c}) -> H~{current_h}, W~{current_w}, C={in_c}")
            elif l_type == 'bn': layers.append(nn.BatchNorm2d(in_c)) #; logger.info(f"  L{i+1}: BatchNorm2d(C={in_c})")
            elif l_type == 'relu': layers.append(nn.ReLU(inplace=True)) #; logger.info(f"  L{i+1}: ReLU")
            elif l_type == 'pool':
                _, k, s = cfg
                layers.append(nn.MaxPool2d(kernel_size=k, stride=s))
                # kh, kw = k[0] if isinstance(k, tuple) else k, k[1] if isinstance(k, tuple) else k
                # sh, sw = s[0] if isinstance(s, tuple) else s, s[1] if isinstance(s, tuple) else s
                # current_h = math.floor((current_h - kh) / sh + 1)
                # current_w = math.floor((current_w - kw) / sw + 1)
                # logger.info(f"  L{i+1}: MaxPool2d(k={k}, s={s}) -> H~{current_h}, W~{current_w}, C={in_c}")
            elif l_type == 'dropout': layers.append(nn.Dropout(cfg[1])) #; logger.info(f"  L{i+1}: Dropout(p={cfg[1]})")

        self.cnn = nn.Sequential(*layers)
        self.final_cnn_channels = in_c # Kanäle nach letztem Conv

        # Bestimme Output-Dimensionen nach CNN dynamisch
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, configuration.IMG_HEIGHT, configuration.PAD_TO_WIDTH)
            cnn_out_dummy = self.cnn(dummy_input)
            _, _, self.height_after_cnn, self.width_after_cnn = cnn_out_dummy.shape
            logger.info(f"Dynamisch ermittelte CNN Output Dimension (vor Pool): C={self.final_cnn_channels}, H={self.height_after_cnn}, W={self.width_after_cnn}")

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None)) # Höhe=1, Breite=Beibehalten
        self.output_seq_len = self.width_after_cnn
        self.output_feature_dim_per_step = self.final_cnn_channels

        logger.info(f"Adaptive Pool erzwingt Höhe=1.")
        logger.info(f"Feature Dim pro Zeitschritt (nach Pool): {self.output_feature_dim_per_step}")
        logger.info(f"Finale Sequenzlänge (Breite): {self.output_seq_len}")


    def forward(self, x):
        features = self.cnn(x)      # (B, C, H_cnn, W_cnn)
        pooled = self.adaptive_pool(features) # (B, C, 1, W_cnn)
        return pooled


class OfflineTransformer(nn.Module):
    """
    Angepasste Transformer Architektur für Offline HTR (Bild -> Text).
    Nutzt CNN Frontend, Transformer Encoder/Decoder.
    """
    def __init__(self,
                 num_classes: int = -1, # Wird unten gesetzt
                 cnn_output_channels: int = configuration.CNN_OUTPUT_CHANNELS,
                 embed_dim: int = configuration.TRANSFORMER_EMBED_DIM,
                 num_heads: int = configuration.TRANSFORMER_NUM_HEADS,
                 ffn_dim: int = configuration.TRANSFORMER_FFN_DIM,
                 num_encoder_layers: int = configuration.TRANSFORMER_ENCODER_LAYERS,
                 num_decoder_layers: int = configuration.TRANSFORMER_DECODER_LAYERS,
                 target_maxlen: int = configuration.TARGET_MAXLEN,
                 dropout_rate: float = configuration.TRANSFORMER_DROPOUT
                 ):
        super().__init__()

        # Initialisiere Vectorizer ZUERST, um num_classes zu bekommen
        self.vectorizer = utilities.VectorizeChar(max_len=target_maxlen)
        self.num_classes = self.vectorizer.get_vocab_size() # Korrekte Anzahl Klassen setzen
        self.pad_token_id = self.vectorizer.pad_token_id
        if self.pad_token_id != 0: raise ValueError("PAD token ID must be 0!")

        self.embed_dim = embed_dim
        self.target_maxlen = target_maxlen

        logger.info("Initialisiere OfflineTransformer Modell...")
        logger.info(f"  Vocab Size (from Vectorizer): {self.num_classes}")
        logger.info(f"  Embed Dim (d_model): {embed_dim}")
        # ... Restliches Parameter-Logging ...
        logger.info(f"  Num Heads: {num_heads}")
        logger.info(f"  FFN Dim: {ffn_dim}")
        logger.info(f"  Encoder Layers: {num_encoder_layers}")
        logger.info(f"  Decoder Layers: {num_decoder_layers}")
        logger.info(f"  Dropout Rate: {dropout_rate}")
        logger.info(f"  Target Max Length: {target_maxlen}")

        # 1. CNN Feature Extractor
        self.cnn_frontend = CNNFeatureExtractor(output_feature_dim=cnn_output_channels)
        self.encoder_input_feature_dim = self.cnn_frontend.output_feature_dim_per_step
        self.encoder_seq_len = self.cnn_frontend.output_seq_len
        logger.info(f"  CNN Frontend Output: Features/Step={self.encoder_input_feature_dim}, SeqLen={self.encoder_seq_len}")

        # 2. Projektion
        if self.encoder_input_feature_dim != embed_dim:
            self.encoder_input_projection = nn.Linear(self.encoder_input_feature_dim, embed_dim)
            logger.info(f"  Encoder Input Projektion: Linear({self.encoder_input_feature_dim} -> {embed_dim})")
        else:
            self.encoder_input_projection = nn.Identity()
            logger.info("  Encoder Input Projektion: Identity")

        # 3. Positional Encoding Encoder
        # *** KORREKTUR: max_len muss groß genug sein für die CNN Sequenzlänge ***
        self.encoder_pos_encoding = PositionalEncoding(embed_dim, dropout_rate, max_len=max(2000, self.encoder_seq_len + 10))

        # 4. Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout_rate)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # 5. Transformer Decoder Components
        self.decoder_embedding = TokenEmbedding(self.num_classes, embed_dim, target_maxlen, dropout_rate)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ffn_dim, dropout_rate)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # 6. Output Classifier
        self.classifier = nn.Linear(embed_dim, self.num_classes) # Verwende self.num_classes

        self._initialize_weights()
        logger.info("OfflineTransformer Modellinitialisierung komplett.")

    # _initialize_weights, _generate_causal_mask, _create_padding_mask bleiben gleich...
    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight);
                if module.bias is not None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, self.embed_dim**-0.5);
                if module.padding_idx is not None: nn.init.constant_(module.weight[module.padding_idx], 0)
            elif isinstance(module, nn.LayerNorm): nn.init.constant_(module.bias, 0); nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.Conv2d): nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu');
            elif isinstance(module, nn.BatchNorm2d): nn.init.constant_(module.weight, 1); nn.init.constant_(module.bias, 0)
        logger.info("Modellgewichte initialisiert.")

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)

    def _create_padding_mask(self, sequence: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        return (sequence == pad_token_id)

    def encode(self, src_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_features = self.cnn_frontend(src_img) # (B, C, 1, W_out)
        encoder_input = pooled_features.squeeze(2).permute(0, 2, 1).contiguous() # (B, W_out, C)
        encoder_input = self.encoder_input_projection(encoder_input) # (B, W_out, D)
        x = self.encoder_pos_encoding(encoder_input) # (B, W_out, D)
        memory_key_padding_mask = None # No padding from CNN fixed width output
        for layer in self.encoder_layers: x = layer(x, key_padding_mask=memory_key_padding_mask)
        memory = self.encoder_norm(x) # (B, W_out, D)
        return memory, memory_key_padding_mask

    def decode(self, tgt_tokens: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor,
               memory_key_padding_mask: torch.Tensor = None, tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        y = self.decoder_embedding(tgt_tokens) # (B, T_dec, D)
        for layer in self.decoder_layers: y = layer(y, memory, tgt_mask, memory_key_padding_mask, tgt_key_padding_mask)
        y = self.decoder_norm(y)
        logits = self.classifier(y) # (B, T_dec, V)
        return logits

    def forward(self, src_img: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        tgt_seq_len = tgt_tokens.size(1)
        causal_mask = self._generate_causal_mask(tgt_seq_len, tgt_tokens.device)
        tgt_padding_mask = self._create_padding_mask(tgt_tokens, self.pad_token_id)
        memory, memory_key_padding_mask = self.encode(src_img)
        logits = self.decode(tgt_tokens, memory, causal_mask, memory_key_padding_mask, tgt_padding_mask)
        return logits

    @torch.no_grad()
    def generate(self, src_img: torch.Tensor, max_len: int = None, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
        self.eval(); device = src_img.device; B = src_img.size(0)
        start_idx=self.vectorizer.start_token_id; end_idx=self.vectorizer.end_token_id; pad_idx=self.pad_token_id
        if max_len is None: max_len = self.target_maxlen

        memory, mem_key_pad_mask = self.encode(src_img)
        decoded_ids = torch.full((B, 1), start_idx, dtype=torch.long, device=device)
        finished_seq = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            if finished_seq.all(): break
            current_len = decoded_ids.size(1)
            causal_mask = self._generate_causal_mask(current_len, device)
            tgt_pad_mask = self._create_padding_mask(decoded_ids, pad_idx)
            logits = self.decode(decoded_ids, memory, causal_mask, mem_key_pad_mask, tgt_pad_mask)
            next_token_logits = logits[:, -1, :]

            # Sampling / Greedy
            if temperature != 1.0: next_token_logits /= temperature
            if finished_seq.any(): next_token_logits[finished_seq] = -float('inf'); next_token_logits[finished_seq, pad_idx] = 0
            if top_k > 0:
                k = min(top_k, next_token_logits.size(-1)); kth_vals, _ = torch.topk(next_token_logits, k, dim=-1)
                indices_to_remove = next_token_logits < kth_vals[:, [-1]]; next_token_logits.masked_fill_(indices_to_remove, -float('inf'))
            if temperature != 1.0 or top_k > 0:
                 probs=F.softmax(next_token_logits, dim=-1); probs=torch.nan_to_num(probs); probs/=probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                 next_token = torch.multinomial(probs, 1).squeeze(1)
            else: next_token = torch.argmax(next_token_logits, dim=-1)

            newly_finished = (next_token == end_idx) & (~finished_seq); finished_seq |= newly_finished
            next_token = torch.where(finished_seq & ~newly_finished, torch.tensor(pad_idx, device=device), next_token)
            decoded_ids = torch.cat([decoded_ids, next_token.unsqueeze(1)], dim=1)

        # Final padding/cleanup
        curr_len = decoded_ids.size(1)
        if curr_len < max_len: decoded_ids = torch.cat([decoded_ids, torch.full((B, max_len - curr_len), pad_idx, dtype=torch.long, device=device)], dim=1)
        final_sequences = []
        for i in range(B):
            seq = decoded_ids[i]; eos_pos = (seq == end_idx).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                clean_seq = seq[:eos_pos[0] + 1]
                pad_len = max_len - len(clean_seq)
                if pad_len > 0: clean_seq = torch.cat([clean_seq, torch.full((pad_len,), pad_idx, dtype=torch.long, device=device)])
                final_sequences.append(clean_seq)
            else: final_sequences.append(seq[:,:max_len][0] if seq.ndim > 1 else seq[:max_len]) # Truncate safely

        return torch.stack(final_sequences)


def build_offline_transformer_model():
    """ Instantiates the OfflineTransformer model using config parameters. """
    logger.info("Baue OfflineTransformer Modell...")
    try:
        model = OfflineTransformer() # Holt Parameter jetzt korrekt aus config
        utilities.ensure_dir(configuration.CHECKPOINT_PATH)
        arch_path = os.path.join(configuration.CHECKPOINT_PATH, "model_architecture.txt")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        with open(arch_path, 'w') as f:
            f.write(str(model) + "\n\n" + "="*30 + " Parameter Count " + "="*30 + "\n")
            f.write(f"Gesamt: {num_params:,}\nTrainierbar: {num_trainable:,}\n")
        logger.info(f"Modellarchitektur -> {arch_path}")
        logger.info(f"Trainierbare Parameter: {num_trainable:,}")
        return model
    except Exception as e:
        logger.exception("Fehler Erstellen OfflineTransformer", exc_info=True)
        return None