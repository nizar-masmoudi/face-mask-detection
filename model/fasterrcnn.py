import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FasterRCNN(nn.Module):
  def __init__(self, num_classes: int, pretrained: bool = False) -> None:
    super(FasterRCNN).__init__()
    self.fasterrcnn = fasterrcnn_resnet50_fpn(pretrained = pretrained)
    in_features = self.fasterrcnn.roi_heads.box_predictor.cls_score.in_features
    self.fasterrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
  def forward(self, x: torch.Tensor):
    return self.fasterrcnn(x)