import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_msssim import ssim
from dataset import MVTecDataset
from model import Autoencoder      

def train_model(category):
    # 1. Hyperparameters
    batch_size = 16
    epochs = 300 # Limit increased to 300
    learning_rate = 1e-3 
    image_size = 224      
    
    # Early Stopping Parameters
    patience = 5 # Patience kept at 5
    patience_counter = 0
    best_loss = float('inf')

    # 2. Data Preparation
    transform = transforms.Compose([transforms.Resize((image_size, image_size)),transforms.ToTensor(),])
    train_dataset = MVTecDataset(root_dir="Datat/MVTecAD", category=category, is_train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 3. Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=1e-4)

    # NEW: ReduceLROnPlateau Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

    # 4. The Training Loop
    print(f"Starting training for {category}...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            imgs = data['image'].to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            
            loss_mse = criterion(outputs, imgs)
            loss_ssim = 1 - ssim(outputs, imgs, data_range=1, size_average=True)
            loss = loss_mse + loss_ssim
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        
        # Update Scheduler based on epoch loss
        scheduler.step(epoch_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, LR: {current_lr}")

        # Early Stopping Logic
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"autoencoder_{category}.pth")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    print(f"Training complete! Best model saved as autoencoder_{category}.pth")

if __name__ == "__main__":
    train_model("cable")