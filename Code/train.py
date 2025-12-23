import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_msssim import ssim
from dataset import MVTecDataset
from model import Autoencoder      

def train_model(category):
    # 1. Hyperparameters (The settings for the "Engine")
    batch_size = 16
    epochs = 200        # How many times the model sees the whole dataset
    learning_rate = 1e-3 
    image_size = 224      
    # 2. Data Preparation
    # We only train on 'good' images, so is_train is True
    transform = transforms.Compose([transforms.Resize((image_size, image_size)),transforms.ToTensor(),])
    
    train_dataset = MVTecDataset(root_dir="Datat/MVTecAD", category=category, is_train=True, transform=transform)
    
    # DataLoader handles shuffling the images so the model doesn't memorize the order
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    
    criterion = nn.MSELoss() 
    
    # Adam is a popular optimizer that automatically adjusts the learning rate
    optimizer = optim.Adam(model.parameters(), learning_rate)

    # 4. The Training Loop
    print(f"Starting training for {category}...")
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for data in train_loader:
            
            imgs = data['image'].to(device)

            
            # Step A: Clear previous gradients
            optimizer.zero_grad()
            
            # Step B: Forward Pass 
            outputs = model(imgs)
            
            # Step C: Calculate Loss
            loss = criterion(outputs, imgs)
            loss_ssim = 1 - ssim(outputs, imgs, data_range=1, size_average=True)
            
            # Combine them: 50% MSE, 50% SSIM
            loss = loss + loss_ssim
            
            # Step D: Backward Pass
            loss.backward()
            
            # Step E: Update Weights
            optimizer.step()

            running_loss += loss.item()
        
        # Print progress every epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")

    # 5. Save the weights so we can use them for Testing later
    torch.save(model.state_dict(), f"autoencoder_{category}.pth")
    print(f"Training complete! Model saved as autoencoder_{category}.pth")

if __name__ == "__main__":
    # You can loop through all your categories here
    train_model("bottle")