import torch
import os
import shutil

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, scheduler, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
        return checkpoint['epoch']
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0

def prepare_dataset():
    # This would contain dataset download and preparation logic
    # For COCO, you'd use the download script mentioned earlier
    print("Dataset preparation not implemented. Please download COCO manually.")# -*- coding: utf-8 -*-

