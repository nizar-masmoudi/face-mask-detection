# Load .env to OS
from dotenv import load_dotenv
load_dotenv()
# Libraries
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import torch.nn as nn
from typing import Callable
from torchvision.utils import draw_bounding_boxes
from PIL import Image # Would use torchvision but it only supports 8-bit png files
from torchvision.transforms import ToPILImage
import torch

class FaceMask(nn.Module):
  def __init__(self, root: str = './data', transform: Callable = None, target_transform: Callable =  None):
    super().__init__()
    self.root = root
    self.imgs = list(sorted(os.listdir(os.path.join(root, 'images'))))
    self.targets = list(sorted(os.listdir(os.path.join(root, 'annotations'))))
    self.transform = transform
    self.target_transform = target_transform
    
  def __getitem__(self, idx):
    img_path = os.path.join(self.root, 'images', f'maksssksksss{idx}.png')
    target_path = os.path.join(self.root, 'annotations', f'maksssksksss{idx}.xml')
    
    img = Image.open(img_path).convert('RGB') # Forcing tensor to be float tensor
    with open(target_path) as tfile:
      target = tfile.read()
    
    # Transforms
    if self.target_transform is not None:
      target = self.target_transform(target)
    if self.transform is not None:
      img = self.transform(img)
      
    return img, target
    
  def __len__(self):
    return len(self.imgs)
    
  @staticmethod
  def download(output_dir: str = './data'):
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('andrewmvd/face-mask-detection', path = output_dir, quiet = False, unzip = True)
    
  def display(self, idx: int):
    display_dict = {
      0: ['No mask', 'red'],
      1: ['Mask worn incorrectly', 'orange'],
      2: ['Mask', 'green']
    }
    img = draw_bounding_boxes(
      image = (self[idx][0]*255).to(torch.uint8),
      boxes = self[idx][1]['boxes'],
      colors = [display_dict[label.item()][1] for label in self[idx][1]['labels']]
    )
    img = ToPILImage()(img)
    img.show()