import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class Pix2PixDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        
        # Pix2Pix datasets usually have input and target side-by-side
        width = image.shape[1]
        width_half = width // 2
        input_image = image[:, :width_half, :]
        target_image = image[:, width_half:, :]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        input_image = transform(input_image)
        target_image = transform(target_image)

        return input_image, target_image

def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to("cuda" if torch.cuda.is_available() else "cpu"), y.to("cuda" if torch.cuda.is_available() else "cpu")
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization
        y = y * 0.5 + 0.5
        x = x * 0.5 + 0.5
        
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(x[0].cpu().permute(1, 2, 0))
        plt.title("Input")
        plt.subplot(1, 3, 2)
        plt.imshow(y_fake[0].cpu().permute(1, 2, 0))
        plt.title("Generated")
        plt.subplot(1, 3, 3)
        plt.imshow(y[0].cpu().permute(1, 2, 0))
        plt.title("Real Target")
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(f"{folder}/epoch_{epoch}.png")
        plt.close()
    gen.train()
