from dataset.facemask import FaceMask
from dataset.utils.transforms import ParseXML
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from model.fasterrcnn import FasterRCNN

dataset = FaceMask(transform = ToTensor(), target_transform = ParseXML())

# dataset.display(0)

data_loader = DataLoader(dataset, batch_size = 1, shuffle = False, collate_fn = lambda batch: tuple(zip(*batch))) # Add custom collate_fn because DataLoader expects all inputs to be same size

batch = next(iter(data_loader))
img, target = batch
print('Successfully loaded a batch of {} images and {} boxes'.format(len(img), len(target)))

model = FasterRCNN(num_classes = 3)
out = model(*batch)
print('Output is', out)