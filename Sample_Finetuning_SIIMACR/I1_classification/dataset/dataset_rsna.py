from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from dataset.randaugment import RandomAugment
import torchxrayvision as xrv
import torch
import random
import numpy as np


class RSNA_Dataset(Dataset):
    def __init__(self, rsna, indices, is_train=True, undersample=False):
        self.indices = indices
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

        if is_train and undersample:
            # Split into majority (label 0) and minority (label 1) classes
            self.majority_class = []
            self.minority_class = []
            for i in indices:
                sample = rsna[i]
                label = 0 if np.array_equal(sample['lab'], [0.0, 0.0]) else 1
                if label == 0:
                    self.majority_class.append(sample)
                else:
                    self.minority_class.append(sample)

            # Undersample the majority class
            majority_count = len(self.minority_class)
            self.majority_class = random.sample(self.majority_class, majority_count)

            # Combine the classes
            self.dataset = self.minority_class + self.majority_class
        else:
            self.dataset = rsna

    def __getitem__(self, index):
        sample = self.dataset[self.indices[index]]

        image = sample["img"]
        image = self.transform(image)  # Apply the defined transformations

        label = torch.tensor(0 if np.array_equal(sample['lab'], [0.0, 0.0]) else 1, dtype=torch.long)

        return {
            "image": image,
            "label": label
        }


    def __len__(self):
        return len(self.indices)
