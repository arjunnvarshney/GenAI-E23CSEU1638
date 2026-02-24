# Lab 6: Pix2Pix GAN Image-to-Image Translation

This project implements the Pix2Pix GAN architecture using a U-Net Generator and PatchGAN Discriminator. It also includes a baseline CNN Encoder-Decoder for performance comparison.

## Objectives
1. Implement U-Net Generator with skip connections.
2. Implement PatchGAN Discriminator.
3. Train using Adversarial + L1 Reconstruction loss.
4. Compare Pix2Pix results with a simple CNN baseline.

## Dataset
We use the **Facades** dataset (Edges/Labels to Real Images).

## Usage
1. **Download Data**:
   ```bash
   python download_data.py
   ```
2. **Train Models**:
   Modify `train.py` to uncomment the training calls, then run:
   ```bash
   python train.py
   ```

## Architecture Details
- **Generator**: U-Net (Encoder-Decoder with Skip Connections)
- **Discriminator**: PatchGAN (Classifies 70x70 patches)
- **Baseline**: Simple CNN Encoder-Decoder (No skip connections, no GAN loss)

## Results
Results are saved in `results_pix2pix/` and `results_baseline/`.
