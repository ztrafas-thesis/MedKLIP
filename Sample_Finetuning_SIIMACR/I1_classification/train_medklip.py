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
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
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


def compute_AUCs(gt, pred, n_class):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    
    # Handle single class scenario
    if n_class == 1:
        AUROCs.append(roc_auc_score(gt_np, pred_np))
    else:
        for i in range(n_class):
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    
    return AUROCs

def get_tokenizer(tokenizer,target_text):
    
    target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length= 64, return_tensors="pt")
    
    return target_tokenizer

def train(model, data_loader, optimizer, criterion, epoch, warmup_steps, device, scheduler, args,config,writer):
    model.train()  
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.update(loss=1.0)
    metric_logger.update(lr = scheduler._get_lr(epoch)[0])

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size 
    scalar_step = epoch*len(data_loader)


    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = sample['image']
        label = sample['label'].float().to(device) #batch_size,num_class
        input_image = image.to(device,non_blocking=True)  

        optimizer.zero_grad()
        pred_class = model(input_image) #batch_size,num_class

        if config['num_classes'] == 1:
            label = label.unsqueeze(1)

        loss = criterion(pred_class,label)
        loss.backward()
        optimizer.step()  
        writer.add_scalar('loss/loss', loss, scalar_step)
        wandb.log({"train/loss": loss.item(), "step": scalar_step})
        scalar_step += 1

        metric_logger.update(loss=loss.item())
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)
     
        metric_logger.update(lr = scheduler._get_lr(epoch)[0])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} #,loss_epoch.mean()

def valid(model, data_loader, criterion,epoch,device,config,writer):
    model.eval()
    val_scalar_step = epoch*len(data_loader)
    val_losses = []

    gt = torch.FloatTensor().cuda()  # assuming multi-class or binary classification
    pred = torch.FloatTensor().cuda()
    for i, sample in enumerate(data_loader):
        image = sample['image']
        label = sample['label'].float().to(device)
        gt = torch.cat((gt, label), 0)
        input_image = image.to(device,non_blocking=True)  
        with torch.no_grad():
            pred_class = model(input_image)
            if config['num_classes'] == 1:
                label = label.unsqueeze(1)
            pred = torch.cat((pred, pred_class), 0)
            val_loss = criterion(pred_class,label)
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            wandb.log({"val/loss": val_loss.item(), "val_step": val_scalar_step})
            val_scalar_step += 1

    avg_val_loss = np.array(val_losses).mean()
    AUROCs = compute_AUCs(gt, pred,config['num_classes'])
    AUROC_avg = np.array(AUROCs).mean()
    wandb.log({"val/AUROC_avg": AUROC_avg, "epoch": epoch})  # Log epoch train loss to wandb
    # Check the shape of ground truth and predictions
    if gt.ndim == 2 and gt.shape[1] > 1:  # If gt is 2D (multi-class or multi-label)
        gt_np = gt[:, 0].cpu().numpy()  # Select the first class
    else:  # 1D ground truth
        gt_np = gt.cpu().numpy()

    if pred.ndim == 2 and pred.shape[1] > 1:  # If pred is 2D (multi-class or multi-label)
        pred_np = pred[:, 0].cpu().numpy()  # Select the first class
    else:  # 1D predictions
        pred_np = pred.cpu().numpy()              
    precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)

    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    max_f1 = np.max(f1_scores)
    wandb.log({"val/max_f1": max_f1, "epoch": epoch})  # Log epoch train loss to wandb
    max_f1_thresh = thresholds[np.argmax(f1_scores)]
    accuracy = accuracy_score(gt_np, pred_np>max_f1_thresh)
    wandb.log({"val/accuracy": accuracy, "epoch": epoch})  # Log epoch train loss to wandb
    return avg_val_loss

# def valid(model, data_loader, criterion,epoch,device,config,writer):
#     model.eval()
#     val_scalar_step = epoch*len(data_loader)
#     val_losses = []
#     for i, sample in enumerate(data_loader):
#         image = sample['image']
#         label = sample['label'].float().to(device)
#         input_image = image.to(device,non_blocking=True)  
#         with torch.no_grad():
#             pred_class = model(input_image)
#             val_loss = criterion(pred_class,label)
#             val_losses.append(val_loss.item())
#             writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
#             val_scalar_step += 1
#     avg_val_loss = np.array(val_losses).mean()
#     return avg_val_loss


def main(args, config):
    wandb.init(project="DeformableMedKLIP", config=config)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset #### 
    print("Creating dataset")
    rsna = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath=config['data_path'],views=["PA","AP"])  
    indices = list(range(len(rsna)))
    train_indices, val_indices = train_test_split(indices, test_size=0.3, random_state=42)

    train_dataset = RSNA_Dataset(rsna, train_indices, is_train = True, undersample=config['undersample']) 
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            collate_fn=None,
            drop_last=True,
        )            
    
    val_dataset = RSNA_Dataset(rsna, val_indices, is_train = False) 
    val_dataloader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )

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

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer) 

    criterion = nn.BCEWithLogitsLoss()

    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                      
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']+1    
        model.load_state_dict(state_dict)    
        print('load checkpoint from %s'%args.checkpoint)
    elif args.pretrain_path:
        checkpoint = torch.load(args.pretrain_path, map_location='cpu')
        state_dict = checkpoint['model']
        model_dict = model.state_dict()
        model_checkpoint = {k:v for k,v in state_dict.items() if k in model_dict}
        model_dict.update(model_checkpoint)
        model.load_state_dict(model_dict)
        print('load pretrain_path from %s'%args.pretrain_path)

    print("Start training")
    start_time = time.time()

    best_val_loss = 10.0
    writer = SummaryWriter(os.path.join(args.output_dir,  'log'))
    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)
        train_stats = train(model, train_dataloader, optimizer, criterion,epoch, warmup_steps, device, lr_scheduler, args,config,writer) 

        for k, v in train_stats.items():
            train_loss_epoch = v
        
        writer.add_scalar('loss/train_loss_epoch', float(train_loss_epoch), epoch)
        writer.add_scalar('loss/leaning_rate',  lr_scheduler._get_lr(epoch)[0] , epoch)
        wandb.log({"train/loss_epoch": float(train_loss_epoch), "epoch": epoch})
        wandb.log({"train/learning_rate": lr_scheduler._get_lr(epoch)[0], "epoch": epoch})

        val_loss = valid(model, val_dataloader, criterion,epoch,device,config,writer)
        writer.add_scalar('loss/val_loss_epoch', val_loss, epoch)
        wandb.log({"val/loss_epoch": val_loss, "epoch": epoch}) 

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'val_loss': val_loss.item()
                        }                     
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_state.pth'))  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if val_loss < best_val_loss:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'best_valid.pth'))  
            best_val_loss = val_loss
            args.model_path = os.path.join(args.output_dir, 'best_valid.pth')
        
        if epoch % 20 == 1 and epoch>1:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_'+str(epoch)+'.pth'))         
                
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
    parser.add_argument('--gpu', type=str,default='1', help='gpu')
    args = parser.parse_args()

    yaml = yaml.YAML(typ='rt')
    config = yaml.load(open(args.config, 'r'))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))     

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.current_device()
    torch.cuda._initialized = True

    main(args, config)
