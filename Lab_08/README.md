# Lab 08: Create Artistic Outputs using Neural Art Concepts

## Objective
To generate artistic images using Generative Adversarial Networks (GANs) by exploring the latent space of the generator.

## Implementation Details
This lab explores two types of GAN architectures:
1.  **Basic GAN:** Deep Convolutional GAN (DCGAN) trained on the CelebA dataset.
2.  **Advanced GAN:** Progressive GAN (PGAN) trained on CelebA-HQ (256x256), which generates higher-resolution images.

### Tasks Completed
- **Data Preparation:** Used pre-trained models from `pytorch_GAN_zoo`. Normalized image visualization is handled by the model's `test` method.
- **Load Trained GAN Model:**
    - Loaded `DCGAN` (Basic).
    - Loaded `PGAN` (Advanced).
    - Generators were set to evaluation mode to produce deterministic outputs.
- **Latent Space Exploration:**
    - Generated 10 random samples for each model.
    - Performed linear interpolation between two random latent vectors to observe smooth transitions.
- **Artistic Outputs:**
    - Visualized the results in the `outputs/` directory.

## Results
The generated images demonstrate the GAN's ability to synthesize realistic yet new facial features. 
- **DCGAN** provides a basic understanding of facial structure at lower resolutions.
- **PGAN** shows much higher fidelity and smoother transitions during interpolation due to its progressive training nature.

### Generated Samples
- `outputs/dcgan_samples.png`: Random samples from DCGAN.
- `outputs/dcgan_interpolation.png`: Latent space transition for DCGAN.
- `outputs/pgan_samples.png`: Random samples from PGAN (High Quality).
- `outputs/pgan_interpolation.png`: Latent space transition for PGAN.

## Learning Outcomes
- Understood the role of the Generator and the latent space.
- Observed the difference in output quality between basic (DCGAN) and advanced (PGAN) architectures.
- Successfully implemented latent vector interpolation for smooth image transitions.
