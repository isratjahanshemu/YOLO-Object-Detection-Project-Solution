# YOLO Object Detection

This repository contains a modular implementation of YOLOv8 for object detection on the COCO dataset.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yolo-object-detection.git
cd yolo-object-detection

2.Install dependencies:
pip install -r requirements.txt

3.Prepare COCO dataset:

Download COCO 2017 dataset

Extract to data/coco with structure:
data/coco/
├── annotations/
├── images/
│   ├── train2017/
│   └── val2017/
└── labels/
    ├── train2017/
    └── val2017/