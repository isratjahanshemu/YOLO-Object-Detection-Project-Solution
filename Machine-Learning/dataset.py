import yaml
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torch
import random

class COCODataset(Dataset):
    def __init__(self, data_yaml, img_size=640, augment=False, mode='train'):
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        self.img_dir = os.path.join(data_config['path'], data_config[mode])
        self.img_size = img_size
        self.augment = augment
        self.mode = mode
        
        # Get image paths
        self.image_files = []
        for file_name in os.listdir(self.img_dir):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                self.image_files.append(os.path.join(self.img_dir, file_name))
        
        # Load class names
        self.classes = data_config['names']
        self.nc = data_config['nc']
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Get label path (assuming same directory structure for labels)
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        
        # Load labels
        labels = []
        if os.path.exists(label_path) and self.mode != 'test':
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, xc, yc, w, h = map(float, line.split())
                    labels.append([class_id, xc, yc, w, h])
        
        # Convert to tensor
        img, labels = self.preprocess(img, labels)
        return img, labels, img_path
    
    def preprocess(self, img, labels):
        # Resize
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img) / 255.0  # Normalize
        
        # Augmentations
        if self.augment and self.mode == 'train':
            img, labels = self.augment_image(img, labels)
        
        # Convert to tensor
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = torch.as_tensor(img, dtype=torch.float32)
        
        if labels:
            labels = torch.as_tensor(labels, dtype=torch.float32)
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)
            
        return img, labels
    
    def augment_image(self, img, labels):
        # HSV color-space augmentation
        if random.random() < 0.5:
            r = np.random.uniform(-1, 1, 3) * [0.015, 0.7, 0.4] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
            dtype = img.dtype
            
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
            
            img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), 
                                 cv2.LUT(sat, lut_sat), 
                                 cv2.LUT(val, lut_val)))
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        
        # Flip left-right
        if random.random() < 0.5:
            img = np.fliplr(img)
            if labels.shape[0]:
                labels[:, 1] = 1 - labels[:, 1]  # x-center
        
        return img, labels# -*- coding: utf-8 -*-

