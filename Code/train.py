import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MVTecDataset
from model import Autoencoder
# NEW: Import SSIM for structural loss
from pytorch_msssim import ssim 

def train_model(category):
    # 1. Hyperparameters
    batch_size = 16
    epochs = 200        
    learning_rate = 1e-3 
    image_size = 224      

    # 2. Data Preparation
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    train_dataset = MVTecDataset(root_dir="Datat/MVTecAD", category=category, is_train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 3. Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    
    # We keep MSE but add SSIM for a "Hybrid Loss"
    mse_criterion = nn.MSELoss() 
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # NEW: Scheduler to fine-tune the model as it trains
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # 4. The Training Loop
    print(f"Starting optimized training for {category}...")
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for data in train_loader:
            imgs = data['image'].to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            
            # Step C: Loss Calculation
            loss_mse = mse_criterion(outputs, imgs)
            loss_ssim = 1 - ssim(outputs, imgs, data_range=1, size_average=True)
            
            # Combine them: 50% MSE, 50% SSIM
            loss = loss_mse + loss_ssim
            
            loss.backward()
            
            optimizer.step()

            running_loss += loss.item()
        
        # NEW: Update the learning rate
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, LR: {current_lr}")

    # 5. Save the weights
    torch.save(model.state_dict(), f"autoencoder_{category}.pth")
    print(f"Training complete! Model saved as autoencoder_{category}.pth")

if __name__ == "__main__":
    train_model("bottle")