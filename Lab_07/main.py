import numpy as np
import torch
import torch.optim as optim
from utils import load_image, im_convert
from model_utils import get_vgg19, get_features, gram_matrix
import matplotlib.pyplot as plt
from PIL import Image
import os

def run_style_transfer(content_path, style_path, iterations=2000, content_weight=1, style_weight=1e6):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load content and style images
    content = load_image(content_path).to(device)
    # Resize style to match content, makes code easier and fits VGG expectations
    style = load_image(style_path, shape=content.shape[-2:]).to(device)

    # Load VGG19 model
    vgg = get_vgg19().to(device)

    # Pre-calculate content and style features
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # Calculate style gram matrices
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Initialize target image as a copy of content image
    # Optimization will happen on target
    target = content.clone().requires_grad_(True).to(device)

    # Style weights for each layer
    style_weights = {'conv1_1': 1.0,
                     'conv2_1': 0.8,
                     'conv3_1': 0.5,
                     'conv4_1': 0.3,
                     'conv5_1': 0.2}

    # Optimizer
    optimizer = optim.Adam([target], lr=0.003)

    print("Starting optimization...")
    for i in range(1, iterations + 1):
        # Get features of target image
        target_features = get_features(target, vgg)

        # 1. Content Loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        # 2. Style Loss
        style_loss = 0
        for layer in style_weights:
            # Get target feature
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            # Get style gram
            style_gram = style_grams[layer]
            # Layer style loss
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            # Add to total style loss
            style_loss += layer_style_loss / (d * h * w)

        # 3. Total Loss
        total_loss = content_weight * content_loss + style_weight * style_loss

        # Update target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f'Iteration: {i}, Total loss: {total_loss.item():.4f}')
            # Save intermediate result
            out_img = im_convert(target)
            plt.imsave(f"Lab_07/output_iter_{i}.png", out_img)

    return target

if __name__ == "__main__":
    content_img = "Lab_07/inputs/content.jpg"
    style_img = "Lab_07/inputs/style.jpg"
    
    if not os.path.exists(content_img) or not os.path.exists(style_img):
        print("Error: Input images not found. Run download_images.py first.")
    else:
        # 300 iterations is usually enough to see the texture transfer start on CPU
        final_img_tensor = run_style_transfer(content_img, style_img, iterations=300)
        
        # Save and show final result
        final_img = im_convert(final_img_tensor)
        plt.imsave("Lab_07/final_stylized_image.png", final_img)
        print("Finished! Final image saved as Lab_07/final_stylized_image.png")
