# Autoregressive Visual Scene Generation

A novel approach to training an autoregressive language model that generates consistent visual scenes from text descriptions. The system uses a circular consistency constraint to ensure semantic preservation through the text→scene→text loop.

## Architecture Overview

The system consists of three main components:

1. **Autoregressive Language Model** (`models.py:AutoregressiveLanguageModel`)
   - Transformer-based model trained from scratch on scene descriptions
   - Produces text embeddings that capture scene semantics
   - 512-dim embeddings by default

2. **Scene Decoder** (`models.py:SceneDecoder`)
   - Transforms text embeddings into 64x64 pixel visual scenes
   - Uses VAE-style bottleneck to prevent trivial solutions
   - Includes spatial coherence network for structured outputs

3. **Caption Network** (`models.py:CaptionNetwork`)
   - CNN-LSTM architecture for generating captions from scenes
   - Ensures generated scenes preserve semantic information
   - Embeddings must match original text embeddings

## Key Features

- **Self-supervised learning**: No paired text-image data required
- **Consistency enforcement**: Circular text→scene→text constraint
- **Anti-trivial mechanisms**:
  - VAE bottleneck with KL divergence
  - Spatial coherence losses
  - Diversity encouragement
  - Perceptual structure losses
- **Emergent visual language**: System develops its own consistent visual representations

## Installation

```bash
pip install torch torchvision matplotlib numpy scikit-learn pillow tqdm
```

## Quick Start

### Basic Training

```bash
python train.py --epochs 50 --batch_size 32 --use_vae
```

### Advanced Training with Custom Parameters

```bash
python train.py \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --d_model 512 \
    --hidden_dim 256 \
    --use_vae \
    --use_amp \
    --lambda_consistency 2.0 \
    --lambda_spatial 0.2 \
    --num_samples 20000
```

### Resume Training

```bash
python train.py --resume checkpoints/checkpoint_epoch_25.pt
```

## File Structure

- `models.py` - Neural network architectures
- `losses.py` - Custom loss functions for preventing trivial solutions
- `data_utils.py` - Synthetic scene description dataset
- `visualization.py` - Training monitoring and result visualization
- `train.py` - Main training script

## Loss Components

The system uses multiple loss components to ensure meaningful generation:

1. **Consistency Loss**: Ensures text→scene→text embeddings match
2. **KL Divergence**: Regularizes VAE latent space
3. **Spatial Coherence**: Encourages smooth, structured scenes
4. **Diversity Loss**: Prevents mode collapse
5. **Perceptual Loss**: Encourages edge structure
6. **Color Consistency**: Maps color words to consistent patterns

## Hyperparameters

Key hyperparameters to tune:

- `--lambda_consistency`: Weight for embedding consistency (default: 1.0)
- `--lambda_kl`: VAE regularization strength (default: 0.001)
- `--lambda_spatial`: Spatial smoothness weight (default: 0.1)
- `--z_dim`: Bottleneck dimension (default: 128, smaller prevents trivial solutions)
- `--use_vae`: Enable VAE bottleneck (recommended)

## Expected Behavior

The model should learn to:
- Map color words (e.g., "yellow") to consistent visual patterns
- Generate spatially coherent scenes (not random noise)
- Preserve semantic information through the generation cycle
- Create diverse outputs for different text inputs

## Monitoring Training

Training progress is logged to the `logs/` directory:
- Training curves: `logs/training_curves.png`
- Generated scenes: `logs/scenes_epoch_N.png`
- Training summary: `logs/training_summary.txt`

## Sample Generation

After training, the script automatically generates sample scenes for visualization:
- Output saved to `sample_generations.png`
- Shows original text, generated scene, and reconstructed caption

## Tips for Success

1. **Start small**: Begin with 10K samples and 50 epochs to verify the approach
2. **Monitor consistency loss**: This is the key metric for semantic preservation
3. **Tune bottleneck size**: Smaller z_dim (64-128) prevents trivial solutions
4. **Balance loss weights**: Adjust lambdas if one loss dominates
5. **Use VAE mode**: The VAE bottleneck helps prevent information passthrough

## Extending the System

Ideas for enhancement:
- Add hierarchical scene generation (layout → details)
- Implement VQ-VAE for discrete visual vocabulary
- Add contrastive learning between matching/non-matching pairs
- Incorporate real image data for comparison
- Add attention visualization for interpretability

## Troubleshooting

- **Mode collapse**: Increase diversity loss weight
- **Noisy scenes**: Increase spatial coherence weight
- **Poor reconstruction**: Increase consistency loss weight
- **Trivial solution**: Decrease z_dim or increase KL weight

## Citation

This implementation explores the concept of self-supervised visual grounding through circular consistency constraints, where an autoregressive language model learns to generate consistent abstract visual representations without paired supervision.