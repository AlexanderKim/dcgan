import os

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def load_dataset(path, batch_size):
    path = os.path.join(os.getcwd(), "data")
    dataset_transforms = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(root=path, transform=dataset_transforms)
    data_loader = DataLoader(dataset=dataset, batch_size=128)

    return data_loader