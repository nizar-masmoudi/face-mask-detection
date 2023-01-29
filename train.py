import argparse
import configparser
from dataset.facemask import FaceMask
from dataset.utils.transforms import ParseXML
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from dataset.loaders import DeviceDataLoader
from model.fasterrcnn import FasterRCNN
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from dataset.utils import train_val_split
import warnings
from loggers.wandb import WandbLogger

def main():
  parser = argparse.ArgumentParser(description = 'Train the model')
  parser.add_argument('-c', '--config', help = 'Path to .cfg file containing training configuration.', default = './configs/train.cfg')
  parser.add_argument('-l', '--logger', help = 'If True a wandb logger will be synced', action='store_false')
  args = parser.parse_args()
  
  config = configparser.ConfigParser()
  try:
    config.read(args.config)
  except:
    raise FileNotFoundError('Config file not found.')
  
  batch_size = config['TRAINING'].getint('BATCH_SIZE')
  lr = config['TRAINING'].getfloat('LEARNING_RATE')
  momentum = config['TRAINING'].getfloat('MOMENTUM')
  weight_decay = config['TRAINING'].getfloat('WEIGHT_DECAY')
  n_epochs = config['TRAINING'].getint('MAX_EPOCHS')
  wandb_project = config['WANDB'].get('PROJECT_NAME')
  wandb_run = config['WANDB'].get('RUN_NAME')
  
  logger = WandbLogger(wandb_project, wandb_run, reinit = True)
  
  logger.log_hyperparameters({
    'batch_size': batch_size,
    'learning_rate': lr,
    'momentum': momentum,
    'weight_decay': weight_decay,
  })
  
  dataset = FaceMask(transform = ToTensor(), target_transform = ParseXML())
  dataset = torch.utils.data.Subset(dataset, list(range(10)))
  
  if not torch.cuda.is_available():
    warnings.warn('Training on CPU. This might take a long time!', UserWarning)

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  train_ds, valid_ds = train_val_split(dataset, val_split = .05)
  train_dl = DeviceDataLoader(
    DataLoader(train_ds, batch_size = batch_size, shuffle = False, collate_fn = lambda batch: tuple(zip(*batch))), # Add custom collate_fn because DataLoader expects all inputs to be same size
    device
  )
  valid_dl = DeviceDataLoader(
    DataLoader(valid_ds, batch_size = batch_size, shuffle = False, collate_fn = lambda batch: tuple(zip(*batch))), # Add custom collate_fn because DataLoader expects all inputs to be same size
    device
  )
  
  model = FasterRCNN(num_classes = 3, weights = 'COCO_V1').to(device)
  opt = torch.optim.SGD(model.parameters(), lr = lr, weight_decay = weight_decay, momentum = momentum)
  
  model.fit(train_dl, valid_dl, metric = MeanAveragePrecision(), opt = opt, n_epochs = n_epochs, logger = logger)

if __name__ == '__main__':
  main()