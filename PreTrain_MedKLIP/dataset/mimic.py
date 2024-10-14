import json
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import random
from dataset.randaugment import RandomAugment
import torch
import h5py


class MIMIC_Dataset(Dataset):
    def __init__(self, path, np_path, mode, transform=None):
        self.hdf5_file_path = f'{path}/{mode}_224.h5'
        self.transform = transform
        
        # Open the HDF5 file
        self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')
        
        # Get the datasets
        self.images_dataset = self.hdf5_file['images']
        # self.reports_dataset = self.hdf5_file['reports']
        self.label_ids = self.hdf5_file['medklip_ids']

        
        # Calculate the length of the dataset
        self.length = len(self.images_dataset)

        self.anaomy_list = [
            'trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
            'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
            'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',
            'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
            'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',
            'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',
            'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach', 'right_atrium', 'right_ventricle', 'aorta', 'svc',
            'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other'
        ]
        self.obs_list = [
            'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
            'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
            'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
            'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
            'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
            'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
            'tail_abnorm_obs', 'excluded_obs'
        ]
        self.rad_graph_results = np.load(np_path)
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if mode == 'train':
            self.transform = transforms.Compose([                        
                # transforms.RandomResizedCrop(224,scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])   
        if mode == 'test':
            self.transform = transforms.Compose([                        
            # transforms.Resize([224, 224]),
            # transforms.RandomHorizontalFlip(),
            # RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
            #                                   'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
            ])   

    def __getitem__(self, idx):
        image = self.images_dataset[idx]
        # report = self.reports_dataset[idx]
        label_id = self.label_ids[idx]
        class_label = self.rad_graph_results[class_label] # (51, 75)
        labels = np.zeros(class_label.shape[-1]) -1
        labels, index_list = self.triplet_extraction(class_label)
        index_list = np.array(index_list)
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        # Convert label to tensor
        label = torch.FloatTensor(label)
        
        return {
            "image": image,
            "label": labels,
            'index': index_list
            }
    
    def triplet_extraction(self, class_label):
        exist_labels = np.zeros(class_label.shape[-1]) -1
        position_list = []
        for i in range(class_label.shape[1]):
            temp_list = []
            ### extract the exist label for each entity and maintain -1 if not mentioned. ###
            if 0 in class_label[:,i]:
                exist_labels[i] = 0
                
            if 1 in class_label[:,i]:
                exist_labels[i] = 1
                ### if the entity exists try to get its position.### 
                ### Note that, the contrastive loss will only be caculated on exist entity as it is meaningless to predict their position for the non-exist entities###
                temp_list.append(random.choice(np.where(class_label[:,i] == 1)[0]))
                try:
                    temp_list = temp_list + random.sample(np.where(class_label[:,i] != 1)[0].tolist(),7)
                except:
                    print('fatal error')
            if temp_list == []:
                temp_list = temp_list +random.sample(np.where(class_label[:,i] != 1)[0].tolist(),8)
            position_list.append(temp_list)
        return exist_labels, position_list

    def __len__(self):
        return self.length
    

