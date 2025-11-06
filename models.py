"""
Autoregressive Language Model with Visual Scene Generation
Multi-component architecture for learning consistent text-to-scene mappings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class AutoregressiveLanguageModel(nn.Module):
    """
    Transformer-based autoregressive language model trained from scratch.
    Produces embeddings that capture scene semantics.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embeddings and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Output projection for next token prediction
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Special embedding extractor for scene generation
        self.embedding_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask for autoregressive generation"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the language model.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_embeddings: Whether to return sentence embeddings

        Returns:
            Dictionary containing logits and optionally embeddings
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings with positional encoding
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Create causal mask for autoregressive prediction
        causal_mask = self.create_causal_mask(seq_len, device)

        # Apply padding mask if provided
        if attention_mask is not None:
            # Convert padding mask to attention mask format
            padding_mask = (1.0 - attention_mask) * -10000.0
        else:
            padding_mask = None

        # Transformer encoding
        hidden_states = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        # Next token prediction logits
        logits = self.output_projection(hidden_states)

        output = {'logits': logits}

        # Extract sentence embedding if requested
        if return_embeddings:
            # Use the last token's hidden state (or mean pool)
            if attention_mask is not None:
                # Mean pooling over non-padded tokens
                masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
                summed = masked_hidden.sum(dim=1)
                counts = attention_mask.sum(dim=1, keepdim=True)
                sentence_embedding = summed / counts
            else:
                # Use last token
                sentence_embedding = hidden_states[:, -1, :]

            # Project to embedding space
            sentence_embedding = self.embedding_projection(sentence_embedding)
            output['embeddings'] = sentence_embedding

        return output

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Starting token IDs [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            eos_token_id: End-of-sequence token ID

        Returns:
            Generated token IDs
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]

        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_length - generated.shape[1]):
                # Get logits for next token
                outputs = self.forward(generated)
                next_token_logits = outputs['logits'][:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float('inf')

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)

                # Check for EOS token
                if eos_token_id is not None and (next_token == eos_token_id).any():
                    break

        return generated


class SceneDecoder(nn.Module):
    """
    Decoder network that transforms text embeddings into visual scenes.
    Uses transposed convolutions to generate 64x64 pixel grids.
    Includes architectural constraints to prevent trivial solutions.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        output_channels: int = 3,
        output_size: int = 64,
        use_vae: bool = True,
        z_dim: int = 128
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.use_vae = use_vae
        self.z_dim = z_dim

        # Information bottleneck (optional VAE-style)
        if use_vae:
            self.fc_mu = nn.Linear(embedding_dim, z_dim)
            self.fc_logvar = nn.Linear(embedding_dim, z_dim)
            self.bottleneck_projection = nn.Linear(z_dim, hidden_dim * 4 * 4)
        else:
            # Direct bottleneck to prevent trivial solutions
            self.bottleneck = nn.Sequential(
                nn.Linear(embedding_dim, z_dim),
                nn.BatchNorm1d(z_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(z_dim, hidden_dim * 4 * 4)
            )

        # Decoder architecture: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.decoder = nn.Sequential(
            # Reshape will happen in forward pass
            nn.BatchNorm2d(hidden_dim),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(),

            # Final convolution to output channels
            nn.Conv2d(hidden_dim // 8, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )

        # Spatial coherence network (for imposing structure)
        self.spatial_prior = nn.Sequential(
            nn.Conv2d(output_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, output_channels, kernel_size=3, padding=1)
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        text_embedding: torch.Tensor,
        return_latent: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate visual scene from text embedding.

        Args:
            text_embedding: Text embeddings [batch_size, embedding_dim]
            return_latent: Whether to return latent codes

        Returns:
            Dictionary containing generated scene and optional latent info
        """
        batch_size = text_embedding.shape[0]
        output = {}

        # Pass through bottleneck
        if self.use_vae:
            mu = self.fc_mu(text_embedding)
            logvar = self.fc_logvar(text_embedding)
            z = self.reparameterize(mu, logvar)
            features = self.bottleneck_projection(z)
            output['mu'] = mu
            output['logvar'] = logvar
            if return_latent:
                output['latent'] = z
        else:
            features = self.bottleneck(text_embedding)

        # Reshape to spatial dimensions
        features = features.view(batch_size, self.hidden_dim, 4, 4)

        # Generate scene through decoder
        scene = self.decoder(features)

        # Apply spatial coherence
        scene_refined = scene + 0.1 * self.spatial_prior(scene)
        scene_refined = torch.clamp(scene_refined, 0, 1)

        output['scene'] = scene_refined
        output['scene_raw'] = scene

        return output


class CaptionNetwork(nn.Module):
    """
    Network that generates captions from visual scenes.
    Uses a lightweight CNN-RNN architecture for 64x64 images.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        n_layers: int = 2,
        input_channels: int = 3,
        input_size: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Convolutional encoder for visual features
        self.visual_encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Global average pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # Project visual features to embedding dimension
        self.visual_projection = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # Token embedding for caption generation
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM for caption generation
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        # Embedding extractor (matches AR model's embedding space)
        self.embedding_extractor = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def encode_image(self, scene: torch.Tensor) -> torch.Tensor:
        """
        Encode visual scene into embedding.

        Args:
            scene: Visual scene [batch_size, 3, 64, 64]

        Returns:
            Visual embedding [batch_size, embedding_dim]
        """
        visual_features = self.visual_encoder(scene)
        visual_embedding = self.visual_projection(visual_features)
        return visual_embedding

    def forward(
        self,
        scene: torch.Tensor,
        target_captions: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for caption generation.

        Args:
            scene: Visual scene [batch_size, 3, 64, 64]
            target_captions: Target caption tokens for teacher forcing [batch_size, seq_len]
            return_embeddings: Whether to return caption embeddings

        Returns:
            Dictionary containing logits and optionally embeddings
        """
        batch_size = scene.shape[0]
        device = scene.device
        output = {}

        # Encode visual scene
        visual_embedding = self.encode_image(scene)

        if target_captions is not None:
            # Teacher forcing mode
            seq_len = target_captions.shape[1]

            # Prepare inputs: [visual_embedding, caption_tokens]
            caption_embeddings = self.token_embedding(target_captions)

            # Prepend visual embedding as first "token"
            visual_token = visual_embedding.unsqueeze(1)
            inputs = torch.cat([visual_token, caption_embeddings[:, :-1, :]], dim=1)

            # Generate through LSTM
            lstm_out, (h_n, c_n) = self.lstm(inputs)

            # Project to vocabulary
            logits = self.output_projection(lstm_out)
            output['logits'] = logits

            # Extract embedding if needed
            if return_embeddings:
                # Use final hidden state
                caption_embedding = self.embedding_extractor(h_n[-1])
                output['embeddings'] = caption_embedding
        else:
            # Inference mode - would implement beam search here
            output['visual_embedding'] = visual_embedding

        return output

    def generate_caption(
        self,
        scene: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        sos_token_id: int = 1,
        eos_token_id: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate caption for a visual scene.

        Args:
            scene: Visual scene [batch_size, 3, 64, 64]
            max_length: Maximum caption length
            temperature: Sampling temperature
            sos_token_id: Start-of-sequence token ID
            eos_token_id: End-of-sequence token ID

        Returns:
            Tuple of (generated_tokens, caption_embedding)
        """
        self.eval()
        batch_size = scene.shape[0]
        device = scene.device

        # Encode visual scene
        visual_embedding = self.encode_image(scene)

        # Initialize with SOS token
        generated = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)

        # Initialize LSTM hidden state
        h = torch.zeros((self.lstm.num_layers, batch_size, self.hidden_dim), device=device)
        c = torch.zeros_like(h)

        with torch.no_grad():
            for i in range(max_length - 1):
                if i == 0:
                    # First step: use visual embedding as input
                    lstm_input = visual_embedding.unsqueeze(1)  # [batch, 1, embedding_dim]
                else:
                    # Subsequent steps: embed the LAST GENERATED token
                    last_token = generated[:, -1:]  # Get the most recent token
                    lstm_input = self.token_embedding(last_token)  # [batch, 1, embedding_dim]

                # LSTM step - process one token at a time
                lstm_out, (h, c) = self.lstm(lstm_input, (h, c))

                # Generate next token from the LSTM output
                logits = self.output_projection(lstm_out.squeeze(1)) / temperature  # [batch, vocab_size]
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]

                # Append to sequence
                generated = torch.cat([generated, next_token], dim=-1)

                # Check for EOS (only break if ALL sequences in batch hit EOS)
                if (next_token == eos_token_id).all():
                    break

        # Extract final embedding from last hidden state
        caption_embedding = self.embedding_extractor(h[-1])

        return generated, caption_embedding