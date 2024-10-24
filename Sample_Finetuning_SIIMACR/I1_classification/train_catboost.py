import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from catboost import CatBoostClassifier
from tqdm import tqdm
import utils
from models.model_MedKLIP import MedKLIP
from models.tokenization_bert import BertTokenizer
from dataset.dataset_rsna import RSNA_Dataset
from scheduler import create_scheduler
from optim import create_optimizer
import wandb
import torchxrayvision as xrv
from sklearn.model_selection import train_test_split
from utils import EarlyStopping
from torchinfo import summary

def compute_AUCs(gt, pred, n_class):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

def get_tokenizer(tokenizer,target_text):
    
    target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length= 64, return_tensors="pt")
    
    return target_tokenizer

def extract_features(model, dataset, device, config, limit_percent=0.5):
    """Extract features from a limited portion of the dataset."""
    model.eval()
    features_list = []
    labels_list = []

    limit = int(len(dataset) * limit_percent)  # Set the limit to 25% of the dataset
    dataset_subset = torch.utils.data.Subset(dataset, range(limit))

    with torch.no_grad():
        for idx in tqdm(range(len(dataset_subset))):
            item = dataset_subset[idx]
            images, labels = item['image'].unsqueeze(0).to(device), item['label'].unsqueeze(0).to(device)
            # print(labels.shape)
            
            # Forward pass to extract features
            features_output = model(images, return_intermediate=True)  # or access the feature layer directly
            features_output = torch.mean(features_output, dim=0)
            
            features_list.append(features_output.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

            # Optionally print shapes to debug
            # print(f'Feature shape at index {idx}: {features_output.shape}')
            

    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    return features_array, labels_array

def train_catboost(features, labels):
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Initialize CatBoost
    catboost_model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        loss_function='Logloss',
        eval_metric='Accuracy',
        verbose=100,
        task_type="GPU"  # Set to GPU if you want to use GPU for CatBoost
    )
    
    # Train CatBoost
    catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    # Evaluate on validation set
    val_predictions = catboost_model.predict(X_val)
    accuracy = accuracy_score(y_val, val_predictions)
    print(f"Validation Accuracy (CatBoost): {accuracy}")

    return catboost_model

def evaluate_catboost(catboost_model, features, labels):
    """Evaluate CatBoost model and calculate AUC."""
    preds = catboost_model.predict_proba(features)[:, 1]  # Get probabilities for the positive class
    auc_score = roc_auc_score(labels, preds)
    accuracy = accuracy_score(labels, (preds > 0.5).astype(int))
    return auc_score, accuracy


def main(args, config):
    # wandb.init(project="DeformableMedKLIP", config=config)
    torch.cuda.empty_cache()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')

    #### Dataset #### 
    print("Creating dataset")
    if config['dataset'] == 'rsna':
        dataset = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath=config['data_path'],views=["PA","AP"]) 
    elif config['dataset'] == 'siim':
        dataset = xrv.datasets.SIIM_Pneumothorax_Dataset(imgpath=config['siim_path'] + 'stage_2_images/', csvpath=config['siim_path'] + 'stage_2_train.csv') 
    if config['undersample']:
        train_indices = np.load('/home/zuzanna/MedKLIP/Sample_Finetuning_SIIMACR/I1_classification/data_file/train_indices.npy').tolist()
        val_indices = np.load('/home/zuzanna/MedKLIP/Sample_Finetuning_SIIMACR/I1_classification/data_file/val_indices.npy').tolist()
    else:
        indices = list(range(len(dataset)))
        train_indices, val_indices = train_test_split(indices, test_size=0.3, random_state=42)

    train_dataset = RSNA_Dataset(dataset, train_indices, is_train = True, undersample=config['undersample']) 
    # train_dataloader = DataLoader(
    #         train_dataset,
    #         batch_size=config['batch_size'],
    #         num_workers=4,
    #         pin_memory=True,
    #         sampler=None,
    #         shuffle=True,
    #         collate_fn=None,
    #         drop_last=True,
    #     )            
    
    val_dataset = RSNA_Dataset(dataset, val_indices, is_train = False) 
    # val_dataloader = DataLoader(
    #         val_dataset,
    #         batch_size=config['batch_size'],
    #         num_workers=4,
    #         pin_memory=True,
    #         sampler=None,
    #         shuffle=False,
    #         collate_fn=None,
    #         drop_last=False,
    #     )

    json_book = json.load(open(config['disease_book'],'r'))
    disease_book = [json_book[i] for i in json_book]
    ana_book = [ 'It is located at ' + i for i in ['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
            'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
            'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',
            'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
            'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',
            'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',
            'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach', 'right_atrium', 'right_ventricle', 'aorta', 'svc',
            'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other']]
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    ana_book_tokenizer = get_tokenizer(tokenizer,ana_book).to(device)
    disease_book_tokenizer = get_tokenizer(tokenizer,disease_book).to(device)
    
    print("Creating model")
    model = MedKLIP(config,ana_book_tokenizer, disease_book_tokenizer)
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model = model.to(device)  

    # wandb.watch(model, log="all", log_freq=20)

    # arg_opt = utils.AttrDict(config['optimizer'])
    # optimizer = create_optimizer(arg_opt, model)
    # arg_sche = utils.AttrDict(config['schedular'])
    # lr_scheduler, _ = create_scheduler(arg_sche, optimizer) 

    # criterion = nn.BCEWithLogitsLoss()

    if args.pretrain_path:
        checkpoint = torch.load(args.pretrain_path, map_location='cpu')
        state_dict = checkpoint['model']
        model_dict = model.state_dict()
        model_checkpoint = {k:v for k,v in state_dict.items() if k in model_dict}
        model_dict.update(model_checkpoint)
        model.load_state_dict(model_dict)
        print('load pretrain_path from %s'%args.pretrain_path)

    print("Start training")
    start_time = time.time()

    train_features, train_labels = extract_features(model, train_dataset, device, config)
    # print(train_features)

    #### Train CatBoost on extracted features ####
    print("Training CatBoost model")
    catboost_model = train_catboost(train_features, train_labels)

    val_features, val_labels = extract_features(model, val_dataset, device, config)
    #### Evaluate CatBoost on validation set ####
    val_auc, val_acc = evaluate_catboost(catboost_model, val_features, val_labels)

    print(f"Validation AUC (CatBoost): {val_auc:.4f}, Validation Accuracy (CatBoost): {val_acc:.4f}")
    # wandb.log({"CatBoost Validation AUC": val_auc, "CatBoost Validation Accuracy": val_acc})
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='Path/To/Res_train.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--model_path', default='') 
    parser.add_argument('--pretrain_path', default='')
    parser.add_argument('--output_dir', default='Path/To/Outputdir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str,default='0', help='gpu')
    args = parser.parse_args()

    yaml = yaml.YAML(typ='rt')
    config = yaml.load(open(args.config, 'r'))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))     

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu !='-1':
        torch.cuda.current_device()
        torch.cuda._initialized = True

    main(args, config)
