import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Any, Callable
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm

class FasterRCNN(nn.Module):
  def __init__(self, num_classes: int, weights: str = None) -> None:
    super(FasterRCNN, self).__init__()
    self.fasterrcnn = fasterrcnn_resnet50_fpn(weights = weights)
    in_features = self.fasterrcnn.roi_heads.box_predictor.cls_score.in_features
    self.fasterrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
  def forward(self, images: torch.Tensor, targets: torch.Tensor):
    return self.fasterrcnn(images, targets)
  
  def loss_batch(self, batch: tuple, opt = None, metric: Metric = None):
    opt.zero_grad()
    loss_dict = self(*batch) # Calculate losses (this returns a dict of multiple losses)
    losses = sum(loss for loss in loss_dict.values())
    losses.backward()
    opt.step()
    return losses.item()/len(batch) # Average loss
  
  def evaluate(self, dataloader: DataLoader, metric: Metric = None):
    with torch.no_grad():
      with tqdm(dataloader, unit = 'batch', desc = 'Validation') as vepoch:
        for batch in vepoch:
          preds = self(*batch)
          metric.update(preds, batch[1]) # preds, targets
          metric_dict =  metric.compute()
          vepoch.set_postfix_str('mAP = {:.4f} - mAP@0.5 = {:.4f} - mAP@0.75 = {:.4f}'.format(metric_dict['map'], metric_dict['map_50'], metric_dict['map_75']))
    metric.reset()
    return metric_dict
  
  def fit(self, train_dl: DataLoader, valid_dl: DataLoader, n_epochs: int, metric: Metric = None, opt = None):
    for epoch in range(n_epochs):
      # Training (1 epoch)
      running_loss = 0.
      self.train()
      with tqdm(train_dl, unit = 'batch', desc = 'Epoch [{:>2}/{:>2}]'.format(epoch+1, n_epochs)) as tepoch:
        for i, batch in enumerate(tepoch):
          loss = self.loss_batch(batch, opt, metric)
          running_loss += loss
          if i == len(train_dl) - 1:
            avg_loss = running_loss/len(train_dl)
            tepoch.set_postfix_str('Average loss = {:.4f}'.format(avg_loss))
      # Validation
      self.eval()
      metric_dict = self.evaluate(valid_dl, metric)
    return metric_dict