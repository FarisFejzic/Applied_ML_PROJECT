import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_msssim import ssim
from dataset import MVTecDataset
from model import Autoencoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt
from threshold import calculate_threshold

def evaluate_and_plot(category):
    
    # Setup & Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(f"autoencoder_{category}.pth"))
    model.eval()


    # Prepare Data
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_dataset = MVTecDataset(root_dir="Datat/MVTecAD", category=category, is_train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Plot improvement in loss
    try:
        train_loss = np.load(f"loss_history_{category}.npy")
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label='Training Loss', color='orange')
        plt.title(f"Loss Minimization during Training for {category.upper()}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE + SSIM)")
        plt.legend()
        plt.grid(True)
        plt.show()
    except FileNotFoundError:
        print("Warning: Loss history file not found. Run train.py first to generate it.")
        
        
    # Store Results
    y_true = []
    y_pred = []
    
    # Calculate threshold
    threshold = calculate_threshold(category)

    print(f"Testing {len(test_dataset)} images...")


    with torch.no_grad():
        for sample in test_loader:
            img = sample['image'].to(device)
            label = sample['label'].item()
            
            # Reconstruction and scoring
            reconstruction = model(img)
            loss_mse = torch.mean((img - reconstruction) ** 2).item()
            loss_ssim = (1 - ssim(reconstruction, img, data_range=1.0)).item()
            
            score = loss_mse + loss_ssim
        
            # Prediction
            prediction = 1 if score > threshold else 0
            
            y_true.append(label)
            y_pred.append(prediction)

    # Create the Confusion Matrix Plot
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Good', 'Anomaly'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f"Classification Results for {category.upper()}")
    plt.show()

    # Print Summary
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("-" * 30)
    print(f"Final Accuracy:  {accuracy*100:.2f}%")
    print(f"Final Precision: {precision*100:.2f}%")
    print(f"Final Recall: {recall*100:.2f}%")
    print(f"Final F1 Score:  {f1*100:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    import numpy as np 
    evaluate_and_plot("bottle")