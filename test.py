from dataset.facemask import FaceMask
from dataset.utils.transforms import ParseXML
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from dataset.loaders import DeviceDataLoader
from model.fasterrcnn import FasterRCNN
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from dataset.utils import train_val_split

subset_size = 10
dataset = FaceMask(transform = ToTensor(), target_transform = ParseXML())
dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Running on', device)

train_ds, valid_ds = train_val_split(dataset, val_split = .2)
train_dl = DeviceDataLoader(
  DataLoader(train_ds, batch_size = 1, shuffle = False, collate_fn = lambda batch: tuple(zip(*batch))), # Add custom collate_fn because DataLoader expects all inputs to be same size
  device
  
)
valid_dl = DeviceDataLoader(
  DataLoader(valid_ds, batch_size = 1, shuffle = False, collate_fn = lambda batch: tuple(zip(*batch))), # Add custom collate_fn because DataLoader expects all inputs to be same size
  device
)

model = FasterRCNN(num_classes = 3)
opt = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

model.fit(train_dl, valid_dl, metric = MeanAveragePrecision(), opt = opt, n_epochs = 1)