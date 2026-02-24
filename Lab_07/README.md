# Lab 07: Neural Style Transfer (NST)

## Objective
To implement Neural Style Transfer (NST) using a pretrained CNN (VGG19) and generate a stylized image by combining the content of one image and the style of another image.

## Lab Tasks
1.  **Data Preparation**:
    *   Load content image and style image.
    *   Load pretrained VGG19 and freeze weights.
    *   Extract content layer (`conv4_2`) and style layers (`conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`).
2.  **Define Loss Functions**:
    *   **Content Loss**: Mean squared error between target and content image features.
    *   **Style Loss**: Mean squared error between the Gram Matrices of target and style image features across multiple layers.
    *   **Total Loss**: Weighted sum of content and style losses.
3.  **Optimization**:
    *   Perform gradient descent on the target image itself to minimize total loss.

## Implementation Details
- **`model_utils.py`**: Contains VGG19 loading, feature extraction, and Gram Matrix calculation.
- **`utils.py`**: Contains image preprocessing and post-processing (normalization for VGG).
- **`download_images.py`**: Downloads a default content image (Landscape) and style image (Van Gogh's Starry Night).
- **`main.py`**: The main optimization script.

## How to Run
1.  Install dependencies:
    ```bash
    pip install torch torchvision numpy matplotlib requests pillow
    ```
2.  Download images:
    ```bash
    python Lab_07/download_images.py
    ```
3.  Run the style transfer:
    ```bash
    python Lab_07/main.py
    ```

## Expected Output
- The `final_stylized_image.png` will show the landscape with the painterly texture of "Starry Night".
- Content structure should be preserved while style texture is transferred.
