import os
import numpy as np
from scipy.linalg import sqrtm

import torch
import torch.nn as nn
from PIL import Image

import torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights


class DatasetMetrics:
    """
    Compute evaluation metrics which are related to the whole dataset.
    Those metrics are FID and KID.
    """

    def __init__(self, device='cuda'):
        self.device = device
        
        # Loading Inception-v3 network (to compute FID and KID metrics)
        weights = Inception_V3_Weights.DEFAULT
        self.inception = inception_v3(weights=weights).to(self.device)
        self.inception.fc = nn.Identity()  # Removing classification layer
        self.inception.eval()

        self.transforms = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_images_from_subdirs(self, root_dir: str) -> list[Image.Image]:
        images = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    img_path = os.path.join(root, file)
                    images.append(Image.open(img_path).convert('RGB'))
        return images

    def _extract_features(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_features = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            tensors = torch.stack([self.transforms(img) for img in batch_imgs]).to(self.device)
            with torch.no_grad():
                feats = self.inception(tensors)
            all_features.append(feats.cpu().numpy())
        return np.concatenate(all_features, axis=0)

    def extract_dataset_features(self, real_dataset_dir: str, gen_dataset_dir: str) -> tuple[np.ndarray, np.ndarray]:
        real_images = self._load_images_from_subdirs(real_dataset_dir)
        gen_images = self._load_images_from_subdirs(gen_dataset_dir)

        if len(real_images) < 2 or len(gen_images) < 2:
            print('[WARN] Not enough images to extract dataset features')
            return np.array([]), np.array([])

        print(f'\n\tTotal real dataset images loaded : {len(real_images)}')
        print(f'\tTotal generated images loaded    : {len(gen_images)}')

        real_feats = self._extract_features(real_images)
        gen_feats = self._extract_features(gen_images)

        return real_feats, gen_feats

    def compute_fid(self, real_feats: np.ndarray, gen_feats: np.ndarray) -> float:
        if len(real_feats) == 0 or len(gen_feats) == 0:
            return float('nan')

        mu_real, sigma_real = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
        mu_gen, sigma_gen = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)

        # Regularization for stability
        eps = 1e-6
        sigma_real += np.eye(sigma_real.shape[0]) * eps
        sigma_gen += np.eye(sigma_gen.shape[0]) * eps

        ssdiff = np.sum((mu_real - mu_gen) ** 2.0)
        covmean = sqrtm(sigma_real.dot(sigma_gen))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma_real + sigma_gen - 2.0 * covmean)
        return float(fid)

    def compute_kid(self, real_feats: np.ndarray, gen_feats: np.ndarray) -> float:
        if len(real_feats) == 0 or len(gen_feats) == 0:
            return float('nan')

        d = real_feats.shape[1]
        m = real_feats.shape[0]
        n = gen_feats.shape[0]

        # Polynomial kernel: k(x, y) = (x^T y / d + 1)^3
        K_XX = (np.dot(real_feats, real_feats.T) / d + 1.0) ** 3
        K_YY = (np.dot(gen_feats, gen_feats.T) / d + 1.0) ** 3
        K_XY = (np.dot(real_feats, gen_feats.T) / d + 1.0) ** 3

        # Unbiased MMD estimator terms
        term_XX = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1))
        term_YY = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1))
        term_XY = np.mean(K_XY)

        kid = term_XX + term_YY - 2 * term_XY
        return float(kid)