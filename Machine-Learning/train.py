import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from dataset import COCODataset
from model import YOLOModel
from utils import save_checkpoint, load_checkpoint
import yaml
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(config_path='configs/config.yaml'):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = YOLOModel(
        model_cfg=config['model'],
        pretrained=config['pretrained']
    )
    model.train()
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Dataset and DataLoader
    train_dataset = COCODataset(
        data_yaml=config['data'],
        img_size=config['imgsz'],
        augment=True,
        mode='train'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr0'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    lf = lambda x: (1 - x / config['epochs']) * (1.0 - config['lrf']) + config['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        
        for i, (imgs, labels, paths) in enumerate(progress_bar):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            loss_dict = model(imgs, labels)
            loss = sum(loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        scheduler.step()
        
        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, filename=f'checkpoint_{epoch+1}.pth')
        
        print(f'Epoch {epoch+1} completed. Avg Loss: {epoch_loss/len(train_loader):.4f}')

def collate_fn(batch):
    imgs, labels, paths = zip(*batch)
    return torch.stack(imgs, 0), labels, paths

if __name__ == '__main__':
    train()# -*- coding: utf-8 -*-

