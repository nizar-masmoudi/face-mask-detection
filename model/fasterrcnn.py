import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Any

class FasterRCNN(nn.Module):
  def __init__(self, num_classes: int, weights: Any = None) -> None:
    super(FasterRCNN, self).__init__()
    self.fasterrcnn = fasterrcnn_resnet50_fpn(weights = weights)
    in_features = self.fasterrcnn.roi_heads.box_predictor.cls_score.in_features
    self.fasterrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
  def forward(self, images: torch.Tensor, targets: torch.Tensor):
    return self.fasterrcnn(images, targets)