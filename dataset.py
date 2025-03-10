import os
import torch
from torch.utils.data import Dataset
import random

class MRIDataset(Dataset):
    def __init__(self, data_dir, noise_level=0.1):
        """
        Args:
            data_dir (str): Directory with all the MRI images
            noise_level (float): Standard deviation of Gaussian noise
        """
        self.data_dir = data_dir
        self.noise_level = noise_level
        self.file_list = []
        
        # Get all .pt files in the directory
        for file in os.listdir(data_dir):
            if file.endswith('.pt'):
                self.file_list.append(os.path.join(data_dir, file))
                
    def __len__(self):
        return len(self.file_list)
    
    def add_noise(self, image):
        """Add Gaussian noise to image"""
        noise = torch.randn_like(image) * self.noise_level
        noisy_image = image + noise
        return noisy_image.clamp(0, 1), image  # Return (noisy, clean) pair
    
    def __getitem__(self, idx):
        # Load the tensor
        image = torch.load(self.file_list[idx])
        
        # Add noise and return the pair
        noisy_image, clean_image = self.add_noise(image)
        return noisy_image, clean_image 