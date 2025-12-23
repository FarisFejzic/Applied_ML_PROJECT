import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # --- ENCODER ---
        # We increase filters and add a layer to shrink the spatial size further
        self.encoder = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            
            # 112x112 -> 56x56
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # 56x56 -> 28x28
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # NEW LAYER: 28x28 -> 14x14 (Tighter Bottleneck)
            # This forces the model to ignore small defects/noise
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # --- DECODER ---
        # We reverse the process to get back to the original image size
        self.decoder = nn.Sequential(
            # 14x14 -> 28x28
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # 28x28 -> 56x56
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # 56x56 -> 112x112
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # 112x112 -> 224x224
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Keep pixels 0-1 to match normalized input
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x