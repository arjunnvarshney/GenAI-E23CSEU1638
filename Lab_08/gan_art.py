import torch
import torch.hub
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid, save_image
import os

# Create directory for outputs
os.makedirs('outputs', exist_ok=True)

def visualize_samples(samples, title, filename):
    grid = make_grid(samples, nrow=5, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(15, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(title)
    plt.axis('off')
    plt.savefig(f'outputs/{filename}')
    plt.show()
    print(f"Saved {title} to outputs/{filename}")

def interpolate(v1, v2, steps=10):
    """Linear interpolation between two vectors."""
    ratios = np.linspace(0, 1, steps)
    vectors = []
    for ratio in ratios:
        v = (1 - ratio) * v1 + ratio * v2
        vectors.append(v)
    return torch.stack(vectors)

def run_lab_8():
    print("Step 1: Data Preparation & Model Setup")
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # TASK 2: Load Basic GAN (DCGAN)
    # ---------------------------------------------------------
    print("\n--- Basic GAN: DCGAN ---")
    try:
        print("Loading DCGAN model...")
        # Loading pre-trained DCGAN trained on CelebA
        dcgan = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', 
                               dataset_name='celebA', pretrained=True, useGPU=use_gpu)
        print("Model loaded successfully.")
        
        # Generate samples
        print("Generating samples...")
        noise, _ = dcgan.buildNoiseData(10)
        with torch.no_grad():
            samples_dcgan = dcgan.test(noise)
        
        if samples_dcgan is not None:
            visualize_samples(samples_dcgan, "DCGAN Generated Samples (CelebA)", "dcgan_samples.png")

            # TASK 3: Latent Space Exploration (Interpolation)
            print("Performing interpolation for DCGAN...")
            v1, _ = dcgan.buildNoiseData(1)
            v2, _ = dcgan.buildNoiseData(1)
            interp_noise = interpolate(v1[0], v2[0], steps=10)
            with torch.no_grad():
                interp_samples = dcgan.test(interp_noise)
            
            if interp_samples is not None:
                visualize_samples(interp_samples, "DCGAN Interpolation", "dcgan_interpolation.png")
        else:
            print("DCGAN generated None samples.")

    except Exception as e:
        print(f"Error with DCGAN: {e}")
        import traceback
        traceback.print_exc()

    # ---------------------------------------------------------
    # TASK 2: Load Advanced GAN (PGAN)
    # ---------------------------------------------------------
    print("\n--- Advanced GAN: PGAN (Progressive GAN) ---")
    try:
        print("Loading PGAN model...")
        pgan = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN', 
                              model_name='celebAHQ-256', pretrained=True, useGPU=use_gpu)
        print("Model loaded successfully.")

        # Generate samples
        print("Generating samples...")
        noise, _ = pgan.buildNoiseData(10)
        with torch.no_grad():
            samples_pgan = pgan.test(noise)
        
        if samples_pgan is not None:
            visualize_samples(samples_pgan, "PGAN Generated Samples (CelebA-HQ)", "pgan_samples.png")

            # TASK 3: Latent Space Exploration (Interpolation)
            print("Performing interpolation for PGAN...")
            v1, _ = pgan.buildNoiseData(1)
            v2, _ = pgan.buildNoiseData(1)
            interp_noise = interpolate(v1[0], v2[0], steps=10)
            with torch.no_grad():
                interp_samples = pgan.test(interp_noise)
            
            if interp_samples is not None:
                visualize_samples(interp_samples, "PGAN Interpolation", "pgan_interpolation.png")
        else:
            print("PGAN generated None samples.")

    except Exception as e:
        print(f"Error with PGAN: {e}")
        # Not printing full traceback here unless requested, but the beta error is likely here

if __name__ == "__main__":
    run_lab_8()
