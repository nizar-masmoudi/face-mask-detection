from dataset.facemask import FaceMask
from torch.utils.data import random_split

def train_val_split(dataset: FaceMask, val_split: float = .05):
  train_split = int(.95*len(dataset))
  valid_split = len(dataset) - train_split
  return random_split(dataset, [train_split, valid_split])