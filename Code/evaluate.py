import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_msssim import ssim
from dataset import MVTecDataset
from model import Autoencoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_and_plot(category):
    # 1. Setup & Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(f"autoencoder_{category}.pth"))
    model.eval()

    # 2. Prepare Data
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_dataset = MVTecDataset(root_dir="Datat/MVTecAD", category=category, is_train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 3. Store Results
    y_true = []
    y_pred = []
    
    
    with open(f"threshold_{category}.txt", "r") as f:
        threshold = float(f.read())

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
            # label is 0 for Good and 1 for Anomaly/Bad in MVTec
            if label == 0:
                print(f"Good score     {score} \n")
            else:
                print(f"Bad score     {score} \n")
            
            # Prediction
            prediction = 1 if score > threshold else 0
            
            y_true.append(label)
            y_pred.append(prediction)

    # 4. Create the Confusion Matrix Plot
    # This shows the True Positives, True Negatives, False Positives, False Negatives
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Good', 'Anomaly'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f"Classification Results for {category.upper()}")
    plt.show()

    # 5. Print Summary
    accuracy = (np.array(y_true) == np.array(y_pred)).sum() / len(y_true)
    print(f"Final Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    import numpy as np # Adding numpy for accuracy calculation
    evaluate_and_plot("bottle")