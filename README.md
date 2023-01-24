# face-mask-detection
Training and evaluating Faster-RCNN for face mask detection.

## Brainstorming ideas
- Train on colab (clone repo and run script each time)
- Use OpenCV for viz and inference
- PyTorch
- Use a subset of the dataset to test training locally
- Use W&B for logging and weights uploading
- Use .env for Kaggle and W&B API Key

## Repository structure
```
.
├── [root]
│   ├── [raw]
│   └── [processed]
├── notebooks
├── model
│   ├── utils
│   └── faster_rcnn.py
├── dataset
│   ├── utils
│   └── facemask.py
├── scripts
│   ├── train.py
│   ├── evaluate.py
├── └── run.py
└── .env
```