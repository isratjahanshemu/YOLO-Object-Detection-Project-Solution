import torch
from dataset import COCODataset
from model import YOLOModel
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate(config_path='configs/config.yaml', weights_path='best.pt'):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = YOLOModel()
    model.load_weights(weights_path)
    model.eval()
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Dataset and DataLoader
    test_dataset = COCODataset(
        data_yaml=config['data'],
        img_size=config['imgsz'],
        augment=False,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize COCO API
    ann_path = os.path.join(config['data']['path'], 'annotations/instances_val2017.json')
    coco_gt = COCO(ann_path)
    coco_dt = []
    
    # Run inference
    for imgs, _, paths in tqdm(test_loader, desc='Evaluating'):
        imgs = imgs.to(device)
        
        with torch.no_grad():
            results = model(imgs)
        
        # Process results
        for i, (path, pred) in enumerate(zip(paths, results)):
            img_id = int(os.path.basename(path).split('.')[0])
            
            for det in pred:
                x1, y1, x2, y2, conf, cls = det[:6]
                w = x2 - x1
                h = y2 - y1
                
                coco_dt.append({
                    'image_id': img_id,
                    'category_id': int(cls),
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(conf)
                })
    
    # COCO Evaluation
    coco_dt = coco_gt.loadRes(coco_dt)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Print metrics
    stats = coco_eval.stats
    metrics = {
        'mAP@0.5': stats[0],
        'mAP@0.5:0.95': stats[1],
        'Recall@0.5': stats[8]
    }
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    return metrics

if __name__ == '__main__':
    evaluate()# -*- coding: utf-8 -*-

