from dataset.facemask import FaceMask
from dataset.utils.transforms import ParseXML
from torchvision.transforms import ToTensor

dataset = FaceMask(transform = ToTensor(), target_transform = ParseXML())