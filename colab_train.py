"""
Simplified training script optimized for Google Colab.
Run this after cloning the repository in Colab.
"""

import torch
import os
import sys

def setup_colab():
    """Setup Colab environment"""
    # Check if we're in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("Running in Google Colab")
    except:
        IN_COLAB = False
        print("Not in Colab")

    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"GPU Available: {device}")
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {memory:.1f} GB")
    else:
        print("No GPU available. Using CPU.")
        if IN_COLAB:
            print("⚠️ Go to Runtime > Change runtime type > GPU for better performance")

    return IN_COLAB

def quick_train():
    """Quick training configuration for testing"""
    cmd = """python train.py \
        --epochs 10 \
        --batch_size 16 \
        --num_samples 2000 \
        --use_vae \
        --use_amp \
        --log_interval 5 \
        --lr 0.0001 \
        --d_model 256 \
        --hidden_dim 128 \
        --z_dim 64 \
        --lambda_kl 0.0001"""

    print("Starting quick training run...")
    print("This should take ~5-10 minutes on Colab GPU")
    os.system(cmd)

def full_train():
    """Full training configuration"""
    cmd = """python train.py \
        --epochs 50 \
        --batch_size 32 \
        --num_samples 10000 \
        --use_vae \
        --use_amp \
        --lr 0.0001 \
        --d_model 512 \
        --hidden_dim 256 \
        --z_dim 128 \
        --lambda_consistency 1.0 \
        --lambda_spatial 0.1 \
        --lambda_kl 0.0001"""

    print("Starting full training run...")
    print("This should take ~30-60 minutes on Colab GPU")
    os.system(cmd)

def save_to_drive():
    """Save results to Google Drive"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')

        # Create directory if it doesn't exist
        os.system('mkdir -p /content/drive/MyDrive/visual-scene-generation')

        # Copy checkpoints and logs
        os.system('cp -r checkpoints /content/drive/MyDrive/visual-scene-generation/')
        os.system('cp -r logs /content/drive/MyDrive/visual-scene-generation/')
        os.system('cp *.png /content/drive/MyDrive/visual-scene-generation/')

        print("✅ Results saved to Google Drive!")
        return True
    except:
        print("❌ Could not save to Google Drive (not in Colab or Drive not mounted)")
        return False

def display_results():
    """Display training results in Colab"""
    try:
        from IPython.display import Image, display

        # Display training curves
        if os.path.exists('logs/training_curves.png'):
            print("Training Curves:")
            display(Image('logs/training_curves.png'))

        # Display sample generations
        if os.path.exists('sample_generations.png'):
            print("\nSample Generations:")
            display(Image('sample_generations.png'))

        # Display latest epoch scenes
        import glob
        scene_files = sorted(glob.glob('logs/scenes_epoch_*.png'))
        if scene_files:
            print(f"\nLatest Generated Scenes:")
            display(Image(scene_files[-1]))
    except:
        print("Run this in Colab notebook to see images")

if __name__ == "__main__":
    # Setup environment
    in_colab = setup_colab()

    # Parse command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "quick"

    # Run training
    if mode == "quick":
        quick_train()
    elif mode == "full":
        full_train()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python colab_train.py [quick|full]")
        sys.exit(1)

    # Display results if in Colab
    if in_colab:
        display_results()
        save_to_drive()