import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import Discriminator
from models.cnn_baseline import CNNBaseline
from utils import Pix2PixDataset, save_some_examples
from tqdm import tqdm
import os

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_EPOCHS = 20  # Reduced for simulation
L1_LAMBDA = 100

def train_pix2pix():
    # Model Initialization
    disc = Discriminator().to(DEVICE)
    gen = Generator().to(DEVICE)
    
    # Optimizers
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    # Loss Functions
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    
    # Data Loading
    train_dataset = Pix2PixDataset(root_dir="data/facades/train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = Pix2PixDataset(root_dir="data/facades/val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print("Starting Pix2Pix Training...")
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(train_loader, leave=True)
        for idx, (x, y) in enumerate(loop):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # --- Train Discriminator ---
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())
            D_real_loss = BCE(D_real, torch.ones_like(D_real))
            D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
            
            disc.zero_grad()
            D_loss.backward()
            opt_disc.step()
            
            # --- Train Generator ---
            D_fake = disc(x, y_fake)
            G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
            L1 = L1_LOSS(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1
            
            gen.zero_grad()
            G_loss.backward()
            opt_gen.step()
            
            loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
            loop.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item())

        if epoch % 5 == 0:
            save_some_examples(gen, val_loader, epoch, folder="results_pix2pix")

def train_baseline():
    model = CNNBaseline().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()
    
    train_dataset = Pix2PixDataset(root_dir="data/facades/train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Pix2PixDataset(root_dir="data/facades/val"), batch_size=1, shuffle=False)

    print("Starting CNN Baseline Training...")
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(train_loader, leave=True)
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)
            prediction = model(x)
            loss = criterion(prediction, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f"Baseline Epoch [{epoch}/{NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item())
            
        if epoch % 5 == 0:
            save_some_examples(model, val_loader, epoch, folder="results_baseline")

if __name__ == "__main__":
    if not os.path.exists("data/facades"):
        print("Please run download_data.py first!")
    else:
        # train_baseline()
        # train_pix2pix()
        print("Training scripts ready. Use separate calls to train models.")
