import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_msssim import ssim
from dataset import MVTecDataset
from model import Autoencoder
import numpy as np

def calculate_threshold(category):
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(f"autoencoder_{category}.pth"))
    model.eval()

    # 2. Prepare Data 
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_dataset = MVTecDataset(root_dir="Datat/MVTecAD", category=category, is_train=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    good_scores = []

    print(f"Analyzing images to establish threshold for {category}...")

    # 3. Collect scores for all known healthy images
    with torch.no_grad():
        for sample in test_loader:
            img = sample['image'].to(device)
            label = sample['label'].item()
            
            # We only use the 'Good' images (label 0) to define what is "Normal"
            if label == 0:
                reconstruction = model(img)
                loss_mse = torch.mean((img - reconstruction) ** 2).item()
                loss_ssim = (1 - ssim(reconstruction, img, data_range=1.0)).item()
                
                # Hybrid score matches your training logic
                score = loss_mse + loss_ssim
                good_scores.append(score)

    #  Statistical Threshold Calculation
    mean_score = np.mean(good_scores)
    std_score = np.std(good_scores)
    
    # Mean + 2.7 * StdDev 
    optimal_threshold = mean_score + (2.7 * std_score)

    print(f"RECOMMENDED THRESHOLD: {optimal_threshold:.6f}")
    
    return optimal_threshold

if __name__ == "__main__":
    calculate_threshold("bottle")