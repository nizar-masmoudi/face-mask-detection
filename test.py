from dataset.facemask import FaceMask
from dataset.utils.transforms import ParseXML
from torchvision.transforms import ToTensor
import torch
dataset = FaceMask(transform = ToTensor(), target_transform = ParseXML())
dataset.display(201)