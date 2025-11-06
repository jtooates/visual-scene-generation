"""
Custom loss functions for the visual scene generation system.
Includes losses to prevent trivial solutions and encourage plausible scenes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class SceneGenerationLoss(nn.Module):
    """
    Composite loss for training the scene generation system.
    Combines multiple objectives to ensure meaningful visual generation.
    """

    def __init__(
        self,
        lambda_reconstruction: float = 1.0,
        lambda_kl: float = 0.001,
        lambda_spatial: float = 0.1,
        lambda_diversity: float = 0.01,
        lambda_perceptual: float = 0.1,
        lambda_consistency: float = 1.0
    ):
        super().__init__()
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_kl = lambda_kl
        self.lambda_spatial = lambda_spatial
        self.lambda_diversity = lambda_diversity
        self.lambda_perceptual = lambda_perceptual
        self.lambda_consistency = lambda_consistency

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL divergence loss for VAE regularization.
        Prevents the decoder from encoding information trivially.
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()

    def total_variation_loss(self, images: torch.Tensor) -> torch.Tensor:
        """
        Total variation loss to encourage spatial smoothness.
        Helps generate coherent scenes rather than noise.
        """
        batch_size = images.size(0)
        tv_h = torch.pow(images[:, :, 1:, :] - images[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(images[:, :, :, 1:] - images[:, :, :, :-1], 2).sum()
        return (tv_h + tv_w) / (batch_size * images.size(1) * images.size(2) * images.size(3))

    def diversity_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encourages diversity in generated scenes.
        Prevents mode collapse where all scenes look the same.
        """
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Compute pairwise distances
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.t())

        # Mask diagonal (self-similarity)
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, 0)

        # Penalize high similarity (encourage diversity)
        diversity = similarity_matrix.sum() / (batch_size * (batch_size - 1))
        return diversity

    def perceptual_loss(self, generated: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Simple perceptual loss based on edge detection.
        Encourages structure in generated images.
        """
        # Sobel edge detection kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        sobel_x = sobel_x.view(1, 1, 3, 3).to(generated.device)
        sobel_y = sobel_y.view(1, 1, 3, 3).to(generated.device)

        # Convert to grayscale if needed
        if generated.size(1) == 3:
            gray = 0.299 * generated[:, 0:1, :, :] + 0.587 * generated[:, 1:2, :, :] + 0.114 * generated[:, 2:3, :, :]
        else:
            gray = generated

        # Compute edges
        edges_x = F.conv2d(gray, sobel_x, padding=1)
        edges_y = F.conv2d(gray, sobel_y, padding=1)
        edges = torch.sqrt(edges_x.pow(2) + edges_y.pow(2))

        # Encourage some edge content (not too much, not too little)
        edge_mean = edges.mean()
        target_edge_level = 0.2  # Target average edge strength
        return (edge_mean - target_edge_level).pow(2)

    def embedding_consistency_loss(
        self,
        original_embedding: torch.Tensor,
        reconstructed_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensures the caption network can reconstruct the original text embedding.
        This is the core constraint that ensures semantic consistency.
        """
        # Cosine similarity loss
        cos_sim = F.cosine_similarity(original_embedding, reconstructed_embedding, dim=1)
        loss = 1 - cos_sim.mean()

        # Optional: Add L2 distance as well
        l2_dist = F.mse_loss(original_embedding, reconstructed_embedding)

        return loss + 0.1 * l2_dist

    def forward(
        self,
        scene_outputs: Dict[str, torch.Tensor],
        original_embeddings: torch.Tensor,
        reconstructed_embeddings: torch.Tensor,
        ar_logits: Optional[torch.Tensor] = None,
        ar_targets: Optional[torch.Tensor] = None,
        caption_logits: Optional[torch.Tensor] = None,
        caption_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for the scene generation system.

        Args:
            scene_outputs: Dictionary from SceneDecoder forward pass
            original_embeddings: Original text embeddings from AR model
            reconstructed_embeddings: Reconstructed embeddings from caption network
            ar_logits: Language model logits for AR loss
            ar_targets: Target tokens for AR loss
            caption_logits: Caption network logits
            caption_targets: Target tokens for caption loss

        Returns:
            Dictionary of individual and total losses
        """
        losses = {}
        total_loss = 0.0

        # 1. Embedding consistency loss (main objective)
        consistency_loss = self.embedding_consistency_loss(original_embeddings, reconstructed_embeddings)
        losses['consistency'] = consistency_loss
        total_loss += self.lambda_consistency * consistency_loss

        # 2. KL divergence loss (if using VAE)
        if 'mu' in scene_outputs and 'logvar' in scene_outputs:
            kl_loss = self.kl_divergence(scene_outputs['mu'], scene_outputs['logvar'])
            losses['kl'] = kl_loss
            total_loss += self.lambda_kl * kl_loss

        # 3. Spatial coherence loss
        tv_loss = self.total_variation_loss(scene_outputs['scene'])
        losses['spatial'] = tv_loss
        total_loss += self.lambda_spatial * tv_loss

        # 4. Diversity loss
        div_loss = self.diversity_loss(original_embeddings)
        losses['diversity'] = div_loss
        total_loss += self.lambda_diversity * div_loss

        # 5. Perceptual loss
        perc_loss = self.perceptual_loss(scene_outputs['scene'])
        losses['perceptual'] = perc_loss
        total_loss += self.lambda_perceptual * perc_loss

        # 6. Language modeling loss (if provided)
        if ar_logits is not None and ar_targets is not None:
            ar_loss = F.cross_entropy(
                ar_logits.reshape(-1, ar_logits.size(-1)),
                ar_targets.reshape(-1),
                ignore_index=-100
            )
            losses['ar_loss'] = ar_loss
            total_loss += ar_loss

        # 7. Caption generation loss (if provided)
        if caption_logits is not None and caption_targets is not None:
            caption_loss = F.cross_entropy(
                caption_logits.reshape(-1, caption_logits.size(-1)),
                caption_targets.reshape(-1),
                ignore_index=-100
            )
            losses['caption_loss'] = caption_loss
            total_loss += caption_loss

        losses['total'] = total_loss
        return losses


class ContrastiveLoss(nn.Module):
    """
    Optional contrastive loss to ensure different texts produce different scenes.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        text_embeddings: torch.Tensor,
        scene_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss between text and scene embeddings.

        Args:
            text_embeddings: Text embeddings [batch_size, embedding_dim]
            scene_embeddings: Scene embeddings [batch_size, embedding_dim]

        Returns:
            Contrastive loss value
        """
        batch_size = text_embeddings.size(0)

        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        scene_embeddings = F.normalize(scene_embeddings, p=2, dim=1)

        # Compute similarity matrix
        logits = torch.matmul(text_embeddings, scene_embeddings.t()) / self.temperature

        # Labels are diagonal (matching pairs)
        labels = torch.arange(batch_size, device=logits.device)

        # Compute cross-entropy loss in both directions
        loss_text_to_scene = F.cross_entropy(logits, labels)
        loss_scene_to_text = F.cross_entropy(logits.t(), labels)

        return (loss_text_to_scene + loss_scene_to_text) / 2


class ColorConsistencyLoss(nn.Module):
    """
    Ensures that specific words (like colors) map to consistent visual patterns.
    """

    def __init__(self, color_words: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self.color_words = color_words or {}

    def register_color_pattern(self, word: str, pattern: torch.Tensor):
        """Register expected color pattern for a word"""
        self.color_words[word] = pattern

    def forward(
        self,
        scenes: torch.Tensor,
        text_tokens: torch.Tensor,
        vocab: Dict[str, int]
    ) -> torch.Tensor:
        """
        Compute consistency loss for color words in text.

        Args:
            scenes: Generated scenes [batch_size, 3, 64, 64]
            text_tokens: Token IDs from text [batch_size, seq_len]
            vocab: Vocabulary mapping

        Returns:
            Color consistency loss
        """
        loss = torch.tensor(0.0, device=scenes.device)
        count = 0

        for color_word, expected_pattern in self.color_words.items():
            if color_word in vocab:
                color_token_id = vocab[color_word]

                # Find scenes that contain this color word
                mask = (text_tokens == color_token_id).any(dim=1)

                if mask.any():
                    relevant_scenes = scenes[mask]

                    # Compute average color in scenes
                    avg_color = relevant_scenes.mean(dim=[2, 3])  # [n_scenes, 3]

                    # Compare with expected pattern
                    color_diff = F.mse_loss(avg_color, expected_pattern.expand_as(avg_color))
                    loss = loss + color_diff
                    count += 1

        if count > 0:
            loss = loss / count

        return loss