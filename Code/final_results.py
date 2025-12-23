import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from dataset import MVTecDataset
from model import Autoencoder

def calculate_auroc(category):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(f"autoencoder_{category}.pth"))
    model.eval()

    # 2. Data
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_dataset = MVTecDataset(root_dir="Datat/MVTecAD", category=category, is_train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    labels = []
    scores = []

    # 3. Evaluation Loop
    print(f"Evaluating {category}...")
    with torch.no_grad():
        for data in test_loader:
            img = data['image'].to(device)
            label = data['label'] # 0 for good, 1 for anomaly
            
            # Reconstruct
            reconstruction = model(img)
            
            # Calculate Anomaly Score
            # We use the Mean Squared Error between input and output as the "score"
            # High error = Model thinks it is an anomaly
            mse_score = torch.mean((img - reconstruction) ** 2).item()
            
            labels.append(label.item())
            scores.append(mse_score)

    # 4. Calculate AUROC using sklearn
    # This compares our predicted scores against the real labels
    auroc = roc_auc_score(labels, scores)
    return auroc

if __name__ == "__main__":
    # List of all categories in MVTec
    categories = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
    
    results = {}
    
    for cat in categories:
        try:
            score = calculate_auroc(cat)
            results[cat] = score
            print(f"Category: {cat} | AUROC: {score:.4f}")
        except FileNotFoundError:
            print(f"Weights for {cat} not found. Skip.")

    # Final Report
    print("\n--- FINAL MVTec REPORT ---")
    for cat, score in results.items():
        print(f"{cat}: {score:.4f}")
    
    mean_auroc = np.mean(list(results.values()))
    print(f"\nMean AUROC: {mean_auroc:.4f}")