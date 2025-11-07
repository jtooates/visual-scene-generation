"""
Visualization utilities for monitoring training and results.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple
import os
from PIL import Image
import io


def visualize_scenes(
    scenes: torch.Tensor,
    texts: List[str],
    save_path: Optional[str] = None,
    title: str = "Generated Scenes"
) -> None:
    """
    Visualize a batch of generated scenes with their text descriptions.

    Args:
        scenes: Generated scenes [batch_size, 3, 64, 64]
        texts: Corresponding text descriptions
        save_path: Optional path to save the figure
        title: Figure title
    """
    batch_size = min(scenes.size(0), 8)  # Limit to 8 images
    scenes = scenes[:batch_size].cpu()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(batch_size):
        # Convert tensor to numpy and transpose to HWC format
        scene = scenes[i].permute(1, 2, 0).numpy()
        scene = np.clip(scene, 0, 1)

        axes[i].imshow(scene)
        axes[i].set_title(texts[i][:30] + '...' if len(texts[i]) > 30 else texts[i], fontsize=10)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(batch_size, 8):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training curves for all tracked metrics.

    Args:
        history: Dictionary of metric names to lists of values
        save_path: Optional path to save the figure
    """
    num_metrics = len(history)
    fig, axes = plt.subplots((num_metrics + 1) // 2, 2, figsize=(12, 4 * ((num_metrics + 1) // 2)))

    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (metric_name, values) in enumerate(history.items()):
        axes[idx].plot(values)
        axes[idx].set_title(f'{metric_name.capitalize()} Over Time')
        axes[idx].set_xlabel('Iteration')
        axes[idx].set_ylabel(metric_name.capitalize())
        axes[idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()


def visualize_embedding_space(
    embeddings: torch.Tensor,
    labels: Optional[List[str]] = None,
    method: str = 'tsne',
    save_path: Optional[str] = None,
    random_state: Optional[int] = None
) -> None:
    """
    Visualize high-dimensional embeddings in 2D.

    Args:
        embeddings: Embeddings to visualize [n_samples, embedding_dim]
        labels: Optional labels for coloring
        method: Dimensionality reduction method ('tsne' or 'pca')
        save_path: Optional path to save the figure
        random_state: Random seed for reproducibility (None for non-deterministic)
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    embeddings_np = embeddings.cpu().numpy()

    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=random_state)
    else:
        reducer = PCA(n_components=2)

    embeddings_2d = reducer.fit_transform(embeddings_np)

    # Plot
    plt.figure(figsize=(10, 8))

    if labels:
        # Create color map for unique labels
        unique_labels = list(set(labels))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

        for label in unique_labels:
            mask = [l == label for l in labels]
            points = embeddings_2d[mask]
            plt.scatter(points[:, 0], points[:, 1], label=label, alpha=0.6)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)

    plt.title(f'Embedding Space Visualization ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()


def compare_reconstructions(
    original_texts: List[str],
    generated_scenes: torch.Tensor,
    reconstructed_texts: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Compare original texts with their scene generations and reconstructed captions.

    Args:
        original_texts: Original text descriptions
        generated_scenes: Generated visual scenes
        reconstructed_texts: Captions generated from scenes
        save_path: Optional path to save the figure
    """
    batch_size = min(len(original_texts), 4)

    fig, axes = plt.subplots(batch_size, 1, figsize=(12, 3 * batch_size))

    if batch_size == 1:
        axes = [axes]

    for i in range(batch_size):
        # Create subplot with image and texts
        ax = axes[i]

        # Display scene
        scene = generated_scenes[i].cpu().permute(1, 2, 0).numpy()
        scene = np.clip(scene, 0, 1)

        ax.imshow(scene, extent=[0, 1, 0, 1])
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)

        # Add text annotations
        ax.text(1.1, 0.7, f"Original: {original_texts[i]}", fontsize=10, wrap=True)
        ax.text(1.1, 0.3, f"Reconstructed: {reconstructed_texts[i]}", fontsize=10, wrap=True)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.suptitle('Text → Scene → Text Reconstruction', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()


def visualize_attention_weights(
    attention_weights: torch.Tensor,
    tokens: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize attention weights from the transformer model.

    Args:
        attention_weights: Attention weights [seq_len, seq_len]
        tokens: List of token strings
        save_path: Optional path to save the figure
    """
    weights = attention_weights.cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(weights, cmap='hot', interpolation='nearest')
    plt.colorbar()

    # Add labels
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.yticks(range(len(tokens)), tokens)

    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.title('Attention Weights')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()


def save_checkpoint(
    models: Dict[str, torch.nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    epoch: int,
    loss: float,
    save_dir: str = 'checkpoints'
) -> str:
    """
    Save model checkpoint.

    Args:
        models: Dictionary of model name to model
        optimizers: Dictionary of optimizer name to optimizer
        epoch: Current epoch
        loss: Current loss value
        save_dir: Directory to save checkpoints

    Returns:
        Path to saved checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'models': {name: model.state_dict() for name, model in models.items()},
        'optimizers': {name: opt.state_dict() for name, opt in optimizers.items()}
    }

    save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, save_path)

    return save_path


def load_checkpoint(
    checkpoint_path: str,
    models: Dict[str, torch.nn.Module],
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None
) -> Tuple[int, float]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        models: Dictionary of model name to model
        optimizers: Optional dictionary of optimizer name to optimizer

    Returns:
        Tuple of (epoch, loss)
    """
    checkpoint = torch.load(checkpoint_path)

    # Load model states
    for name, model in models.items():
        if name in checkpoint['models']:
            model.load_state_dict(checkpoint['models'][name])

    # Load optimizer states if provided
    if optimizers:
        for name, opt in optimizers.items():
            if name in checkpoint['optimizers']:
                opt.load_state_dict(checkpoint['optimizers'][name])

    return checkpoint['epoch'], checkpoint['loss']


def create_visualization_grid(
    scenes: torch.Tensor,
    n_rows: int = 4,
    n_cols: int = 8,
    img_size: int = 64
) -> np.ndarray:
    """
    Create a grid of scenes for visualization.

    Args:
        scenes: Batch of scenes [batch_size, 3, height, width]
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        img_size: Size of each image

    Returns:
        Grid as numpy array
    """
    batch_size = min(scenes.size(0), n_rows * n_cols)
    scenes = scenes[:batch_size].cpu()

    # Create empty grid
    grid = np.zeros((n_rows * img_size, n_cols * img_size, 3))

    for idx in range(batch_size):
        row = idx // n_cols
        col = idx % n_cols

        # Extract and convert scene
        scene = scenes[idx].permute(1, 2, 0).numpy()
        scene = np.clip(scene, 0, 1)

        # Place in grid
        grid[row * img_size:(row + 1) * img_size,
             col * img_size:(col + 1) * img_size] = scene

    return grid


class TrainingMonitor:
    """
    Helper class for monitoring training progress.
    """

    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = log_dir
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'consistency_loss': [],
            'kl_loss': [],
            'spatial_loss': [],
            'diversity_loss': [],
            'perceptual_loss': []
        }
        os.makedirs(log_dir, exist_ok=True)

    def update(self, metrics: Dict[str, float]):
        """Update history with new metrics"""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
            else:
                self.history[key] = [value]

    def plot(self, save: bool = True):
        """Plot current training curves"""
        save_path = os.path.join(self.log_dir, 'training_curves.png') if save else None
        plot_training_curves(self.history, save_path)

    def log_images(self, scenes: torch.Tensor, texts: List[str], epoch: int):
        """Log generated images"""
        save_path = os.path.join(self.log_dir, f'scenes_epoch_{epoch}.png')
        visualize_scenes(scenes, texts, save_path)

    def log_roundtrip(self, original_texts: List[str], reconstructed_texts: List[str], epoch: int):
        """Log round-trip examples to file"""
        save_path = os.path.join(self.log_dir, f'roundtrip_epoch_{epoch}.txt')
        with open(save_path, 'w') as f:
            f.write(f"Round-Trip Examples (Epoch {epoch})\n")
            f.write("="*70 + "\n\n")
            for i, (orig, recon) in enumerate(zip(original_texts, reconstructed_texts)):
                f.write(f"{i+1}. Original:      {orig}\n")
                f.write(f"   Reconstructed: {recon}\n\n")

    def save_summary(self):
        """Save training summary"""
        summary_path = os.path.join(self.log_dir, 'training_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Training Summary\n")
            f.write("=" * 50 + "\n\n")

            for metric_name, values in self.history.items():
                if values:
                    f.write(f"{metric_name}:\n")
                    f.write(f"  Final: {values[-1]:.6f}\n")
                    f.write(f"  Best: {min(values):.6f}\n")
                    f.write(f"  Average: {np.mean(values):.6f}\n\n")