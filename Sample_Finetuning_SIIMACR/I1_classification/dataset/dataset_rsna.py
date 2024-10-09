from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from dataset.randaugment import RandomAugment
import torchxrayvision as xrv
import torch
from torch.utils.data import Subset
import numpy as np
from collections import Counter


class RSNA_Dataset(Dataset):
    def __init__(self, is_train=True):
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        def to_3_channels(img):
            return img.repeat(3, 1, 1)

        if is_train:
            self.transform = transforms.Compose([        
                xrv.datasets.ToPILImage(),   
                transforms.Resize([224, 224]),               
                transforms.ToTensor(),
                transforms.Lambda(lambda img: img if img.shape[0] == 3 else to_3_channels(img)),
                normalize,
            ])   
        else:
            self.transform = transforms.Compose([   
                xrv.datasets.ToPILImage(),                     
                transforms.Resize([224, 224]),
                transforms.ToTensor(), 
                transforms.Lambda(lambda img: img if img.shape[0] == 3 else to_3_channels(img)),
                normalize,
            ]) 

        if is_train:
            path = "/home/zuzanna/rsna/train"
        else:
            path = "/home/zuzanna/rsna/test"
            
        self.rsna = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath="/home/zuzanna/rsna/train",transform=self.transform,views=["PA","AP"])  

        num_samples = len(self.rsna)
        indices = np.arange(num_samples)

        self.dataset = Subset(self.rsna, indices[:20])

    def __getitem__(self, index):
        sample = self.dataset[index]
        return {
            "image": sample["img"],
            "label": torch.tensor(0 if np.array_equal(sample['lab'], [0.0, 0.0]) else 1, dtype=torch.long)
        }


    def __len__(self):
        return len(self.dataset)
