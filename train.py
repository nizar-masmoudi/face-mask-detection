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

def main():
  parser = argparse.ArgumentParser(description = 'Train the model')
  parser.add_argument('-c', '--config', help = 'Path to .cfg file containing training configuration.', default = './configs/train.cfg')
  args = parser.parse_args()
  
  config = configparser.ConfigParser()
  try:
    config.read(args.config)
  except:
    raise FileNotFoundError('Config file not found.')
  
  batch_size = config['TRAINING'].getint('BatchSize')
  lr = config['TRAINING'].getfloat('LearningRate')
  weight_decay = config['TRAINING'].getint('WeightDecay')
  n_epochs = config['TRAINING'].getint('MaxEpochs')
  
  dataset = FaceMask(transform = ToTensor(), target_transform = ParseXML())
  if not torch.cuda.is_available():
    raise UserWarning('Training on CPU. This might take a long time!')
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
  
  model = FasterRCNN(num_classes = 3).to(device)
  opt = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
  
  model.fit(train_dl, valid_dl, metric = MeanAveragePrecision(), opt = opt, n_epochs = n_epochs)

if __name__ == '__main__':
  main()