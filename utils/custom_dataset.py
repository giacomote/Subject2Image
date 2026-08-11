import os

import torch
from torch.utils.data import Dataset

from torchvision import transforms
from PIL import Image


class CustomDataset(Dataset):
    """
    Load and prepare the subject images from a specified folder.
    """

    def __init__(self, image_folder: str, size: int = 1024):
        self.image_paths = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]

        if len(self.image_paths) == 0:
            raise ValueError(f'({image_folder}) No images found')

        print(f'[INFO] Found {len(self.image_paths)} training images')
        
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1, 1] normalization (VAE accepted format)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(image)