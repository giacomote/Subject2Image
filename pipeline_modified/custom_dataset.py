import os

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image


class CustomDataset(Dataset):
    """
    Load and prepare the subject images from a specified folder.
    """

    def __init__(self, image_folder: str, size: int = 1024, augmentation_target: int = 10):
        self.target = augmentation_target

        self.image_paths = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]

        if len(self.image_paths) == 0:
            raise ValueError(f'({image_folder}) No images found')

        print(f'[INFO] Found {len(self.image_paths)} real images. Target: {self.target}')
        
        self.base_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # Augmentation pipeline
        self.aug_transform = transforms.Compose([
            transforms.Resize(int(size * 1.15), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomRotation(degrees=20, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return self.target

    def __getitem__(self, idx):
        # Bringing the index in the range [0, self.target]
        idx = idx % self.target

        # Locating which real file must be opened
        real_idx = idx % len(self.image_paths)
        image = Image.open(self.image_paths[real_idx]).convert('RGB')

        # Original image if (requested image index < number of real images)
        # Replicated and modified image otherwise
        if idx < len(self.image_paths):
            return self.base_transform(image)
        else:
            return self.aug_transform(image)