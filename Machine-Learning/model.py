import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOModel(nn.Module):
    def __init__(self, model_cfg='yolov8n.pt', pretrained=True):
        super().__init__()
        self.model = YOLO(model_cfg) if pretrained else YOLO(model_cfg).model
        
    def forward(self, x, targets=None):
        if self.training:
            return self.model(x, targets)
        else:
            return self.model(x)
    
    def load_weights(self, weights_path):
        self.model = YOLO(weights_path)
    
    def save_weights(self, save_path):
        torch.save(self.model.state_dict(), save_path)
