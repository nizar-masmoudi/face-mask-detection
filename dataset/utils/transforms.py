from bs4 import BeautifulSoup
from typing import Any
import torch

class ParseXML(object):
  def __call__(self, xml: str) -> Any:
    target = {'boxes': [], 'labels': []}
    soup = BeautifulSoup(xml, 'html.parser')
    objects = soup.find_all('object')
    for obj in objects:
      # Parse bboxes
      xml_bbox = obj.find('bndbox')
      bbox = [
        int(xml_bbox.find('xmin').text), 
        int(xml_bbox.find('ymin').text), 
        int(xml_bbox.find('xmax').text), 
        int(xml_bbox.find('ymax').text)
      ]
      target['boxes'].append(bbox)
      # Parse labels
      if obj.find('name').text == 'with_mask':
        target['labels'].append(2)
      elif obj.find('name').text == 'mask_weared_incorrect':
        target['labels'].append(1)
      else:
        target['labels'].append(0)
    # Return target dict of Tensors
    target['boxes'] = torch.as_tensor(target['boxes'], dtype = torch.float32)
    target['labels'] = torch.as_tensor(target['labels'], dtype = torch.int64)
    return target