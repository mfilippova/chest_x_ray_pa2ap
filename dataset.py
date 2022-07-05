import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    '''
    Custom dataset for unpaired AP to PA translation
    for RSNA Pneumonia Detection Challenge dataset
    '''
    def __init__(self, paths_ap, paths_pa, transform=transforms.ToTensor(), mode='train'):
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

        item_ap = np.array(Image.open(path_ap))[:, :, None, 0]
        item_pa = np.array(Image.open(path_pa))[:, :, None, 0]

        if self.transform:
            item_ap = self.transform(item_ap)
            item_pa = self.transform(item_pa)

        if index == len(self) - 1:
            self.new_perm()

        return item_ap, item_pa
