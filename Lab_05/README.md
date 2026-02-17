# Week 5: Image Transformation Models

**Date:** 11-02-2026

## Overview

This week focuses on implementing and applying deep learning models for image transformation tasks, from custom CNN architectures for colorization to pre-trained diffusion models for text-guided image editing.

## Notebook: `Week_05_11_02_2026.ipynb`

### Part I: CNN Encoder-Decoder for Image Colorization

#### Task 1: Data Preparation
- Loaded **CIFAR-10 dataset** with image normalization to [-1, 1] range
- Configured data loaders with batch size 64 for training and size 10 for testing
- Converted RGB images to grayscale for encoder-decoder training
- Applied `transforms.ToTensor()` and `transforms.Normalize()` for preprocessing

#### Task 2: Encoder-Decoder Architecture
- **Encoder:** Compresses 32x32 RGB image to latent representation
  - Conv2d layers (1 → 32 → 64 channels) with stride 2 for downsampling
  - Output: 8x8 latent feature maps
- **Decoder:** Expands latent representation back to 32x32 RGB image
  - ConvTranspose2d layers (64 → 32 → 3 channels) with stride 2 for upsampling
  - Output: 3-channel RGB image using Tanh activation [-1, 1] range

#### Task 3: Training
- **Loss Function:** Mean Squared Error (MSE) for pixel-level reconstruction
- **Optimizer:** Adam with learning rate 0.001
- **Epochs:** 10 epochs on training set
- **Input-Target Pair:** Grayscale image (1 channel) → Color image (3 channels)

#### Task 4: Evaluation & Visualization
- Generated colorized images from grayscale inputs
- Compared model output with original color targets
- Visualized side-by-side comparison of input, generated, and real target images

### Part II: InstructPix2Pix for Text-Guided Image Editing

#### Task 1: Model Setup
- Loaded **Stable Diffusion InstructPix2Pix** pre-trained model from Hugging Face
- Selected appropriate hardware device (CUDA if available, CPU otherwise)
- Used `EulerAncestralDiscreteScheduler` for inference
- Configured automatic safety checker disabling for research purposes

#### Task 2: User Settings & Tuning Parameters
- **IMAGE_STABILITY:** Controls how strongly the model preserves original image features (default: 2.0)
- **TEXT_STRENGTH:** Controls how strongly the model follows the text prompt (default: 10.0)
- **Inference Steps:** 40 steps for quality-stability balance

#### Task 3: Multi-Prompt Generation
Demonstrated text-guided image transformation with diverse prompts:
- Environmental changes: "make me sit in open mountains", "turn the background into a cyberpunk city"
- Style transformations: "make it look like a pencil sketch"
- Creative edits: "Make 2 balloons grow on the head", "Make me drive a supercar"
- Scene positioning: "Make me study under the moon", "Make me hospitalized in hospital bed"
- Travel scenarios: "Make 2 balloons grow on the head", "Make me ride a superbike"

#### Task 4: Output Generation & Saving
- Generated 512x512 output images for each prompt
- Saved results with descriptive filenames including prompt text
- Displayed side-by-side comparison of original and edited images

## Key Concepts

### Image Colorization
- **Encoder-Decoder:** Learns compressed intermediate representation of data
- **Pixel-Level Loss:** MSE loss directly optimizes reconstructed pixel values
- **CNN Architecture:** Exploits spatial structure and properties of images

### Text-Guided Image Editing
- **Diffusion Models:** Iterative denoising process for high-quality image generation
- **Guidance Scales:** Control balance between image preservation and prompt adherence
- **Pre-trained Models:** Leverage models trained on large-scale image-text datasets

## Learning Outcomes

1. Understand and implement CNN encoder-decoder architectures for image tasks
2. Train custom models for image colorization using MSE loss
3. Apply pre-trained diffusion models from Hugging Face
4. Master text-guided image editing using InstructPix2Pix
5. Fine-tune generation parameters for optimal results
6. Work with modern deep learning libraries and pipelines

## Technical Stack

- **PyTorch:** Deep learning framework for custom model implementation
- **Torchvision:** Dataset loading and computer vision utilities
- **Diffusers:** Hugging Face library for diffusion models
- **PIL (Pillow):** Image processing and format conversion
- **Matplotlib:** Visualization of results
