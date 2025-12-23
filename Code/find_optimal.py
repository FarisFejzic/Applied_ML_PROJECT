import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_msssim import ssim
from dataset import MVTecDataset
from model import Autoencoder
import numpy as np

def calculate_threshold(category, k=4):
    # 1. Setup & Load Trained Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(f"autoencoder_{category}.pth"))
    model.eval()

    # 2. Prepare Training Data (Good images only)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # We use the training set to see what 'normal' scores look like
    train_dataset = MVTecDataset(root_dir="Datat/MVTecAD", category=category, is_train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    train_scores = []

    # 3. Collect Hybrid Scores
    print(f"Calculating optimal threshold for {category}...")
    with torch.no_grad():
        for data in train_loader:
            img = data['image'].to(device)
            output = model(img)
            
            # Use the exact same metric as your training/eval: MSE + SSIM
            loss_mse = torch.mean((img - output) ** 2).item()
            loss_ssim = (1 - ssim(output, img, data_range=1.0)).item()
            
            train_scores.append(loss_mse + loss_ssim)

    # 4. Statistical Threshold Calculation
    # We use Mean + (k * StdDev). k=3 covers ~99.7% of normal data distribution.
    mean_score = np.mean(train_scores)
    std_score = np.std(train_scores)
    optimal_threshold = mean_score + (k * std_score)

    print("-" * 40)
    print(f"Mean Good Score:  {mean_score:.6f}")
    print(f"Std Deviation:    {std_score:.6f}")
    print(f"Optimal Threshold: {optimal_threshold:.6f}")
    print("-" * 40)

    # 5. Save the threshold for evaluate.py
    with open(f"threshold_{category}.txt", "w") as f:
        f.write(str(optimal_threshold))
    
    print(f"Threshold saved to threshold_{category}.txt")

if __name__ == "__main__":
    calculate_threshold("bottle")