"""
Main training script for the autoregressive visual scene generation system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import argparse
import os
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

from models import AutoregressiveLanguageModel, SceneDecoder, CaptionNetwork
from losses import SceneGenerationLoss, ContrastiveLoss
from data_utils import create_data_loaders, SceneDescriptionDataset
from visualization import TrainingMonitor, visualize_scenes, save_checkpoint, load_checkpoint


class SceneGenerationTrainer:
    """
    Main trainer class for the scene generation system.
    """

    def __init__(
        self,
        config: argparse.Namespace,
        device: torch.device
    ):
        self.config = config
        self.device = device

        # Ensure checkpoint and log directories exist
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        print(f"Checkpoint directory: {os.path.abspath(config.checkpoint_dir)}")
        print(f"Log directory: {os.path.abspath(config.log_dir)}")

        # Create data loaders
        print("\nCreating dataset...")
        self.train_loader, self.val_loader, self.dataset = create_data_loaders(
            batch_size=config.batch_size,
            num_samples=config.num_samples,
            train_split=config.train_split,
            seed=config.seed
        )
        print(f"Dataset created with {len(self.dataset)} samples")
        print(f"Vocabulary size: {self.dataset.vocab_size}")

        # Initialize models
        print("\nInitializing models...")
        self.ar_model = AutoregressiveLanguageModel(
            vocab_size=self.dataset.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout
        ).to(device)

        self.scene_decoder = SceneDecoder(
            embedding_dim=config.d_model,
            hidden_dim=config.hidden_dim,
            use_vae=config.use_vae,
            z_dim=config.z_dim
        ).to(device)

        self.caption_network = CaptionNetwork(
            vocab_size=self.dataset.vocab_size,
            embedding_dim=config.d_model,
            hidden_dim=config.hidden_dim,
            n_layers=2,
            dropout=config.dropout
        ).to(device)

        # Initialize optimizers
        self.ar_optimizer = optim.Adam(
            self.ar_model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999)
        )

        self.decoder_optimizer = optim.Adam(
            self.scene_decoder.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999)
        )

        self.caption_optimizer = optim.Adam(
            self.caption_network.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999)
        )

        # Learning rate schedulers
        self.ar_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.ar_optimizer,
            T_max=config.epochs
        )
        self.decoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.decoder_optimizer,
            T_max=config.epochs
        )
        self.caption_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.caption_optimizer,
            T_max=config.epochs
        )

        # Initialize loss functions
        self.scene_loss_fn = SceneGenerationLoss(
            lambda_reconstruction=config.lambda_reconstruction,
            lambda_kl=config.lambda_kl,
            lambda_spatial=config.lambda_spatial,
            lambda_diversity=config.lambda_diversity,
            lambda_perceptual=config.lambda_perceptual,
            lambda_consistency=config.lambda_consistency
        )

        self.contrastive_loss_fn = ContrastiveLoss()

        # Mixed precision training
        self.scaler = GradScaler() if config.use_amp else None

        # Training monitor
        self.monitor = TrainingMonitor(log_dir=config.log_dir)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.warmup_steps = 100  # Gradual warmup for stability

        # Scheduled sampling parameters
        self.scheduled_sampling_start_epoch = 5  # Start after model stabilizes
        self.scheduled_sampling_end_epoch = 30   # Full autoregressive by epoch 30

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.ar_model.train()
        self.scene_decoder.train()
        self.caption_network.train()

        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}')

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Stage 1: Train AR model on language modeling
            self.ar_optimizer.zero_grad()

            if self.config.use_amp and self.scaler is not None:
                with autocast():
                    ar_outputs = self.ar_model(
                        input_ids,
                        attention_mask,
                        return_embeddings=True
                    )
                    ar_loss = nn.functional.cross_entropy(
                        ar_outputs['logits'].reshape(-1, ar_outputs['logits'].size(-1)),
                        labels.reshape(-1),
                        ignore_index=-100
                    )

                self.scaler.scale(ar_loss).backward()
                self.scaler.unscale_(self.ar_optimizer)
                torch.nn.utils.clip_grad_norm_(self.ar_model.parameters(), max_norm=1.0)
                self.scaler.step(self.ar_optimizer)
                self.scaler.update()
            else:
                ar_outputs = self.ar_model(
                    input_ids,
                    attention_mask,
                    return_embeddings=True
                )
                ar_loss = nn.functional.cross_entropy(
                    ar_outputs['logits'].reshape(-1, ar_outputs['logits'].size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100
                )
                ar_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ar_model.parameters(), max_norm=1.0)
                self.ar_optimizer.step()

            # Check for NaN in AR loss
            if torch.isnan(ar_loss) or torch.isinf(ar_loss):
                print(f"Warning: NaN/Inf in AR loss at step {self.global_step}, skipping batch")
                continue

            # Stage 2: Generate scenes from embeddings
            with torch.no_grad():
                text_embeddings = ar_outputs['embeddings'].detach()

                # Check embeddings for NaN
                if torch.isnan(text_embeddings).any() or torch.isinf(text_embeddings).any():
                    print(f"Warning: NaN/Inf in embeddings at step {self.global_step}, skipping batch")
                    continue

            self.decoder_optimizer.zero_grad()
            self.caption_optimizer.zero_grad()

            # Calculate scheduled sampling probability
            # Gradually transition from teacher forcing (prob=0) to autoregressive (prob=1)
            if self.epoch < self.scheduled_sampling_start_epoch:
                use_autoregressive_prob = 0.0
            elif self.epoch >= self.scheduled_sampling_end_epoch:
                use_autoregressive_prob = 1.0
            else:
                # Linear ramp from 0 to 1
                progress = (self.epoch - self.scheduled_sampling_start_epoch) / \
                          (self.scheduled_sampling_end_epoch - self.scheduled_sampling_start_epoch)
                use_autoregressive_prob = progress

            # Decide whether to use autoregressive generation or teacher forcing
            use_autoregressive = (torch.rand(1).item() < use_autoregressive_prob)

            if self.config.use_amp and self.scaler is not None:
                with autocast():
                    # Generate scenes
                    scene_outputs = self.scene_decoder(text_embeddings)
                    generated_scenes = scene_outputs['scene']

                    # Generate captions from scenes - use scheduled sampling
                    if use_autoregressive:
                        # Autoregressive generation mode - detach to avoid backprop through generation
                        with torch.no_grad():
                            generated_tokens, _ = self.caption_network.generate_caption(
                                generated_scenes.detach(),  # Detach scene to avoid backprop issues
                                max_length=input_ids.shape[1],
                                sos_token_id=self.dataset.vocab['<SOS>'],
                                eos_token_id=self.dataset.vocab['<EOS>']
                            )

                        # Now get embeddings in training mode for gradient flow
                        caption_outputs = self.caption_network(
                            generated_scenes,
                            generated_tokens,
                            return_embeddings=True
                        )
                        # For autoregressive mode, we only care about embedding consistency
                        caption_logits = None
                        caption_targets = None
                    else:
                        # Teacher forcing mode (original behavior)
                        caption_outputs = self.caption_network(
                            generated_scenes,
                            input_ids,  # Use original as target for teacher forcing
                            return_embeddings=True
                        )
                        caption_logits = caption_outputs['logits']
                        caption_targets = labels

                    # Compute composite loss
                    losses = self.scene_loss_fn(
                        scene_outputs,
                        text_embeddings,
                        caption_outputs['embeddings'],
                        caption_logits=caption_logits,
                        caption_targets=caption_targets
                    )

                    # Apply warmup scaling
                    warmup_scale = min(1.0, self.global_step / self.warmup_steps)
                    total_batch_loss = losses['total'] * warmup_scale

                self.scaler.scale(total_batch_loss).backward()
                # Gradient clipping to prevent NaN
                self.scaler.unscale_(self.decoder_optimizer)
                self.scaler.unscale_(self.caption_optimizer)
                torch.nn.utils.clip_grad_norm_(self.scene_decoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.caption_network.parameters(), max_norm=1.0)
                self.scaler.step(self.decoder_optimizer)
                self.scaler.step(self.caption_optimizer)
                self.scaler.update()
            else:
                # Generate scenes
                scene_outputs = self.scene_decoder(text_embeddings)
                generated_scenes = scene_outputs['scene']

                # Generate captions from scenes - use scheduled sampling
                if use_autoregressive:
                    # Autoregressive generation mode - detach to avoid backprop through generation
                    with torch.no_grad():
                        generated_tokens, _ = self.caption_network.generate_caption(
                            generated_scenes.detach(),  # Detach scene to avoid backprop issues
                            max_length=input_ids.shape[1],
                            sos_token_id=self.dataset.vocab['<SOS>'],
                            eos_token_id=self.dataset.vocab['<EOS>']
                        )

                    # Now get embeddings in training mode for gradient flow
                    caption_outputs = self.caption_network(
                        generated_scenes,
                        generated_tokens,
                        return_embeddings=True
                    )
                    # For autoregressive mode, we only care about embedding consistency
                    caption_logits = None
                    caption_targets = None
                else:
                    # Teacher forcing mode (original behavior)
                    caption_outputs = self.caption_network(
                        generated_scenes,
                        input_ids,
                        return_embeddings=True
                    )
                    caption_logits = caption_outputs['logits']
                    caption_targets = labels

                # Compute composite loss
                losses = self.scene_loss_fn(
                    scene_outputs,
                    text_embeddings,
                    caption_outputs['embeddings'],
                    caption_logits=caption_logits,
                    caption_targets=caption_targets
                )

                # Apply warmup scaling to prevent early instability
                warmup_scale = min(1.0, self.global_step / self.warmup_steps)
                total_batch_loss = losses['total'] * warmup_scale
                total_batch_loss.backward()

                # Gradient clipping to prevent NaN
                torch.nn.utils.clip_grad_norm_(self.scene_decoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.caption_network.parameters(), max_norm=1.0)

                self.decoder_optimizer.step()
                self.caption_optimizer.step()

            # Check for NaN and skip update if found
            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                print(f"Warning: NaN/Inf detected at step {self.global_step}, skipping batch")
                self.decoder_optimizer.zero_grad()
                self.caption_optimizer.zero_grad()
                continue

            # Update metrics
            total_loss += total_batch_loss.item()
            progress_bar.set_postfix({
                'ar_loss': ar_loss.item(),
                'consistency': losses['consistency'].item(),
                'kl_loss': losses.get('kl', torch.tensor(0)).item(),
                'total': total_batch_loss.item(),
                'autoreg_prob': f'{use_autoregressive_prob:.2f}'
            })

            # Log intermediate results
            if batch_idx % self.config.log_interval == 0:
                self.monitor.update({
                    'train_loss': total_batch_loss.item(),
                    'consistency_loss': losses['consistency'].item(),
                    'spatial_loss': losses.get('spatial', torch.tensor(0)).item(),
                    'diversity_loss': losses.get('diversity', torch.tensor(0)).item(),
                    'perceptual_loss': losses.get('perceptual', torch.tensor(0)).item()
                })

            self.global_step += 1

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self) -> float:
        """Validate the model"""
        self.ar_model.eval()
        self.scene_decoder.eval()
        self.caption_network.eval()

        total_loss = 0
        all_scenes = []
        all_texts = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                texts = batch['texts']

                # Forward pass through all models
                ar_outputs = self.ar_model(
                    input_ids,
                    attention_mask,
                    return_embeddings=True
                )

                text_embeddings = ar_outputs['embeddings']
                scene_outputs = self.scene_decoder(text_embeddings)
                generated_scenes = scene_outputs['scene']

                caption_outputs = self.caption_network(
                    generated_scenes,
                    input_ids,
                    return_embeddings=True
                )

                # Compute loss
                losses = self.scene_loss_fn(
                    scene_outputs,
                    text_embeddings,
                    caption_outputs['embeddings']
                )

                total_loss += losses['total'].item()

                # Collect samples for visualization and diagnostics
                if batch_idx == 0:
                    all_scenes = generated_scenes[:8]
                    all_texts = texts[:8]

                    # Generate captions for round-trip diagnostics
                    all_reconstructed = []
                    for i in range(min(4, len(all_scenes))):  # Show 4 examples
                        scene = all_scenes[i:i+1]
                        generated_caption, _ = self.caption_network.generate_caption(scene)
                        reconstructed_text = self.dataset.decode_tokens(generated_caption[0])
                        all_reconstructed.append(reconstructed_text)

        avg_loss = total_loss / len(self.val_loader)

        # Print round-trip diagnostics
        if len(all_scenes) > 0 and len(all_reconstructed) > 0:
            print("\n" + "="*70)
            print("ROUND-TRIP EXAMPLES (Text → Scene → Text)")
            print("="*70)
            for i in range(len(all_reconstructed)):
                print(f"\n{i+1}. Original:      {all_texts[i]}")
                print(f"   Reconstructed: {all_reconstructed[i]}")
            print("="*70 + "\n")

            # Save to file
            self.monitor.log_roundtrip(all_texts[:len(all_reconstructed)], all_reconstructed, self.epoch)

        # Visualize some results
        if len(all_scenes) > 0:
            self.monitor.log_images(all_scenes, all_texts, self.epoch)

        return avg_loss

    def train(self):
        """Main training loop"""
        print("\nStarting training...")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            # Training
            train_loss = self.train_epoch()

            # Calculate current scheduled sampling probability for logging
            if self.epoch < self.scheduled_sampling_start_epoch:
                ss_prob = 0.0
            elif self.epoch >= self.scheduled_sampling_end_epoch:
                ss_prob = 1.0
            else:
                progress = (self.epoch - self.scheduled_sampling_start_epoch) / \
                          (self.scheduled_sampling_end_epoch - self.scheduled_sampling_start_epoch)
                ss_prob = progress

            print(f"\nEpoch {epoch + 1}/{self.config.epochs} - Train Loss: {train_loss:.4f} - Autoregressive prob: {ss_prob:.2f}")

            # Validation
            val_loss = self.validate()
            print(f"Validation Loss: {val_loss:.4f}")

            # Update learning rates
            self.ar_scheduler.step()
            self.decoder_scheduler.step()
            self.caption_scheduler.step()

            # Save checkpoint if best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = save_checkpoint(
                    models={
                        'ar_model': self.ar_model,
                        'scene_decoder': self.scene_decoder,
                        'caption_network': self.caption_network
                    },
                    optimizers={
                        'ar_optimizer': self.ar_optimizer,
                        'decoder_optimizer': self.decoder_optimizer,
                        'caption_optimizer': self.caption_optimizer
                    },
                    epoch=epoch,
                    loss=val_loss,
                    save_dir=self.config.checkpoint_dir
                )
                print(f"Saved best model checkpoint: {checkpoint_path}")

            # Update monitor
            self.monitor.update({
                'train_loss': train_loss,
                'val_loss': val_loss
            })

            # Plot training curves periodically
            if (epoch + 1) % 5 == 0:
                self.monitor.plot(save=True)

        # Save final summary
        self.monitor.save_summary()
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

    def generate_samples(self, num_samples: int = 4):
        """Generate sample scenes from text"""
        self.ar_model.eval()
        self.scene_decoder.eval()
        self.caption_network.eval()

        # Sample texts
        sample_texts = [
            "a large red ball in the center",
            "small blue cube on the left",
            "yellow sphere floating on striped background",
            "multiple green triangles arranged in a pattern"
        ]

        print("\nGenerating sample scenes...")
        all_scenes = []
        all_reconstructed = []

        with torch.no_grad():
            for text in sample_texts[:num_samples]:
                # Tokenize text
                tokens = [self.dataset.vocab.get(word, self.dataset.vocab['<UNK>'])
                         for word in text.split()]
                tokens = [self.dataset.vocab['<SOS>']] + tokens + [self.dataset.vocab['<EOS>']]
                input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)

                # Generate embedding
                ar_outputs = self.ar_model(input_ids, return_embeddings=True)
                text_embedding = ar_outputs['embeddings']

                # Generate scene
                scene_outputs = self.scene_decoder(text_embedding)
                scene = scene_outputs['scene']

                # Generate caption
                generated_caption, _ = self.caption_network.generate_caption(scene)
                reconstructed_text = self.dataset.decode_tokens(generated_caption[0])

                all_scenes.append(scene)
                all_reconstructed.append(reconstructed_text)

        # Visualize results
        all_scenes = torch.cat(all_scenes, dim=0)
        visualize_scenes(
            all_scenes,
            [f"Orig: {t}\nRecon: {r}" for t, r in zip(sample_texts, all_reconstructed)],
            save_path='sample_generations.png',
            title='Sample Scene Generations'
        )


def main():
    parser = argparse.ArgumentParser(description='Train Visual Scene Generation Model')

    # Model architecture
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for decoder')
    parser.add_argument('--z_dim', type=int, default=128, help='Latent dimension for VAE')
    parser.add_argument('--use_vae', action='store_true', help='Use VAE in scene decoder')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (8 recommended for stability)')
    parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train/val split ratio')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training (may cause instability)')

    # Loss weights
    parser.add_argument('--lambda_reconstruction', type=float, default=1.0)
    parser.add_argument('--lambda_kl', type=float, default=0.001, help='KL divergence weight (increased from 0.00001 to enforce compression)')
    parser.add_argument('--lambda_spatial', type=float, default=0.1)
    parser.add_argument('--lambda_diversity', type=float, default=0.01)
    parser.add_argument('--lambda_perceptual', type=float, default=0.1)
    parser.add_argument('--lambda_consistency', type=float, default=1.0)

    # Data parameters
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Logging
    parser.add_argument('--log_dir', type=str, default='logs', help='Logging directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')

    # Resume training
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create trainer
    trainer = SceneGenerationTrainer(args, device)

    # Resume if specified
    if args.resume:
        epoch, loss = load_checkpoint(
            args.resume,
            models={
                'ar_model': trainer.ar_model,
                'scene_decoder': trainer.scene_decoder,
                'caption_network': trainer.caption_network
            },
            optimizers={
                'ar_optimizer': trainer.ar_optimizer,
                'decoder_optimizer': trainer.decoder_optimizer,
                'caption_optimizer': trainer.caption_optimizer
            }
        )
        trainer.epoch = epoch
        print(f"Resumed from epoch {epoch} with loss {loss:.4f}")

    # Train model
    trainer.train()

    # Generate sample outputs
    trainer.generate_samples()


if __name__ == '__main__':
    main()