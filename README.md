# Face Mask Detection
This repository serves mainly as a boilerplate for future Machine Learning projects. Kaggle's Face Mask Monitoring dataset is used to train a Faster-RCNN. 

## Repository structure
```
.
├── [.kaggle]
│   └── [kaggle.json]           <-- Authentification information for kaggle
├── [data]                      <-- Raw Face Mask Wearing dataset
│   └── [annotations]
│   └── [images]
├── configs                     <-- Training parameters for easier access purposes
│   └── [train.cfg]
├── model                       <-- Model architecture, training and evaluation setup
│   └── fasterrcnn.py
├── dataset                     <-- Pytorch Dataset class and batch loaders
│   ├── utils
│   └── facemask.py
│   └── loaders.py
├── loggers                     <-- Loggers MLOps
│   └── wandb.py
├── setup.py                    <-- Script to download dataset
├── train.py                    <-- Script to train model
└── [.env]                      <-- Environment variables
```

## How it works
### Setup
Create file `.kaggle/kaggle.json` with the following content.
```json
{
    "username": "username",
    "key": "kaggle-api-key"
}
```
Create file `.env` and set the following environment variables.
```
WANDB_API_KEY = wandb_api_key
WANDB_MODE = online
KAGGLE_CONFIG_DIR = .kaggle/
```
Download data using the setup script.
```
> python setup.py
```
### Training
Edit training configuration file `configs/train.cfg` to fit your needs and run the training script.
```
> python train.py --logger --config configs/train.cfg
```
