from typing import Union, List
from torch.utils.data import DataLoader
import torch

def to_device(data: Union[torch.Tensor, List[torch.Tensor]], device: str): # Move data to a device ('CPU' or 'GPU')
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data] # If data is a list of tensors
    elif isinstance(data, dict):
      return {k: to_device(v, device) for k, v in data.items()} # If data is a dict and its values are tensors
    return data.to(device, non_blocking = True) # If data is a single tensor

class DeviceDataLoader(): # Wraps a dataloader to move data to a device
    def __init__(self, dl: DataLoader, device: str):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl: # For each batch
            yield to_device(b, self.device) # Return but doesn't stop the for loop (NB: Garabage collection happens automatically)
    def __len__(self):
        return (len(self.dl))