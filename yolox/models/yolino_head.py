import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, cxcywh2xyxy, meshgrid, visualize_assign

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv

class YOLinOHead(nn.Module):
    def __init__(self, 
        in_channels: int,
        num_predictors_per_cell: int,
        num_classes: int):
    
        super().__init__()

        num_predicted_channels = num_predictors_per_cell * (2 + num_classes) # 5 = 1 * (2 (1D Border points) + num_classes(3))

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_predicted_channels,
            kernel_size=3,
            padding="same",
            stride=1,
        )

    def get_losses(self):
      L_loc = None
      L_resp = None
      L_noresp = None
      L_class = None
      total_loss = None

      return (L_loc, L_resp, L_noresp, L_class, total_loss)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if self.training:
            return self.get_losses() # TODO
        
        else:
            return x # torch.Tensor([batch_size, num_predicted_channels, width, height]) => torch.Tensor([1, 5, 20, 20]) for dark5
        


