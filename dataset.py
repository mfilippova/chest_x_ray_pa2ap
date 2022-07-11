import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset


class UnpairedGenerationDataset(Dataset):
    '''
    Custom dataset for unpaired AP to PA translation
    for RSNA Pneumonia Detection Challenge dataset
    '''
    def __init__(self, paths_ap, paths_pa, transform=None, mode='train'):
        self.transform = transform
        self.paths_ap = paths_ap
        self.paths_pa = paths_pa
        self.new_perm()

    def __len__(self):
        return len(self.paths_ap)
    
    def new_perm(self):
        self.randperm = torch.randperm(len(self.paths_pa))[:len(self.paths_ap)]

    def __getitem__(self, index):
        path_ap = f'train_png/{self.paths_ap[index]}.png'
        path_pa = f'train_png/{self.paths_pa[self.randperm[index]]}.png'
        
        item_ap = torchvision.io.read_image(path_ap)[0, None, :, :] / 255
        item_pa = torchvision.io.read_image(path_pa)[0, None, :, :] / 255
        
        if self.transform:
            item_ap = self.transform(item_ap)
            item_pa = self.transform(item_pa)

        if index == len(self) - 1:
            self.new_perm()

        return item_ap, item_pa


class ClassificationDataset(Dataset):
    '''
    Custom dataset for classification
    for RSNA Pneumonia Detection Challenge dataset
    '''
    def __init__(self, paths, targets, transform=None, mode='train'):
        self.transform = transform
        self.paths = paths
        self.targets = targets

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = f'train_png/{self.paths[index]}.png'
        
        image = torchvision.io.read_image(path)[0, None, :, :] / 255
        
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.targets[index])
        
        return {'image': image, 'label': label}
