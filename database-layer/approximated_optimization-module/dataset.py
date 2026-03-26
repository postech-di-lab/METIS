import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from utils import set_seed

# set_seed(42)
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

def preprocess_and_cache(raw_dataset, transform=None):
        preprocessed_data = []
        for image, label in raw_dataset:
            if transform is not None:
                image = transform(image)
            preprocessed_data.append((image, label))
        return preprocessed_data

def compress_idx():
    raw_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    images = np.array([raw_dataset[i][0].numpy() for i in range(len(raw_dataset))])

    std = np.std(images, axis=0).squeeze()
    horizontal_std = np.mean(std, axis=1)
    vertical_std = np.mean(std, axis=0)

    threshold = 0.2
    horizontal_indices = np.where(horizontal_std >= threshold)[0]
    vertical_indices = np.where(vertical_std >= threshold)[0]
    top = np.min(horizontal_indices)
    bottom = np.max(horizontal_indices)
    left = np.min(vertical_indices)
    right = np.max(vertical_indices)

    return top, bottom, left, right

class CustomCrop:
    def __init__(self, comp_idx):
        self.top = comp_idx[0]
        self.bottom = comp_idx[1]
        self.left = comp_idx[2]
        self.right = comp_idx[3]

    def __call__(self, tensor):
        cropped_tensor = tensor[:, self.top:self.bottom, self.left:self.right]
        return cropped_tensor

class MnistDataset(Dataset):
    def __init__(self, train=True, compress = False):
        raw_dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
        
        if compress == True:
            comp_idx = compress_idx()
            compress_transform = transforms.Compose([CustomCrop(comp_idx)])
        else:
            compress_transform = None

        self.dataset = preprocess_and_cache(raw_dataset, compress_transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label

    