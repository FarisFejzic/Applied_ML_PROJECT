import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, is_train=True, transform=None):
        self.root_dir = root_dir
        self.category = category
        self.is_train = is_train
        self.transform = transform
        
        # Define paths based on MVTec structure
        self.phase = 'train' if is_train else 'test'
        self.img_path = os.path.join(root_dir, category, self.phase)
        self.mask_path = os.path.join(root_dir, category, 'ground_truth')
        
        # Collect image files
        self.image_files = glob.glob(os.path.join(self.img_path, "**/*.png"), recursive=True)
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Determine the label (0 for good, 1 for anomaly)
        label = 0 if "good" in img_path else 1
        
        if self.transform:
            image = self.transform(image)

        # For testing anomalous images, we also need the ground truth mask
        if not self.is_train and label == 1:
            # Logic to find matching mask in ground_truth folder
            rel_path = os.path.relpath(img_path, self.img_path)
            mask_file = os.path.join(self.mask_path, rel_path.replace(".png", "_mask.png"))
            mask = Image.open(mask_file).convert('L') # Load as grayscale
            mask = transforms.Resize(image.shape[1:], interpolation=Image.NEAREST)(mask)
            mask = transforms.ToTensor()(mask)
        else:
            # Normal images or training images have no masks (all black)
            mask = torch.zeros((1, *image.shape[1:]))

        return {
            'image': image, 
            'label': label, 
            'mask': mask, 
            'path': img_path
        }