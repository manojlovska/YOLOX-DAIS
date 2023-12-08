import math
from loguru import logger
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, cxcywh2xyxy, meshgrid, visualize_assign

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class YOLinOHead(nn.Module):
    def __init__(self, 
        in_channels: int,
        num_predictors_per_cell: int,
        conf: int,
        act="silu"):
    
        super().__init__()

        self.num_predictors_per_cell = num_predictors_per_cell
        self.conf = conf

        num_predicted_channels = self.num_predictors_per_cell * (4 + self.conf) # 5 = 1 * (4 (2D Cartesian points) + 1 (confidence/ either 0 or 1))

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_predicted_channels,
            kernel_size=3,
            padding="same",
            stride=1,
        )

        self.bn = nn.BatchNorm2d(num_predicted_channels)
        self.act = get_activation(act, inplace=True)

    def reshape_tensor(self, tensor):
        """
        Args:
            tensor (torch.tensor):
                with shape [batch_size, num_predictors*coordinates, rows, cols]

        Returns:
            torch.tensor:
                with shape [batch_size, num_cells, num_predictors, coordinates]
        """
        batch_size = tensor.shape[0]
        tensor = tensor.permute(0, 2, 3, 1)
        tensor = tensor.reshape(batch_size, -1, self.num_predictors_per_cell, 5)
        return tensor

    def get_losses(self, outputs, target_tensors, p=0.5):
        target_tensors = self.reshape_tensor(target_tensors)

        # shapes: [batch_size, num_cells, num_predictors, coordinates]
        coords_gt = target_tensors[:, :, :, :4].float()
        confs_gt = target_tensors[:, :, :, -1].float()

        coords_pred = outputs[:, :, :, :4].float().sigmoid()
        confs_pred = outputs[:, :, :, -1].float().sigmoid()

        # shape: [1] => 1: average of all batches
        # sum for all cells, mean for all batches, only include cells where in the ground truth there are line segments i.e. conf_gt > 0
        L_loc = torch.where(confs_gt > 0, (torch.linalg.norm(coords_gt[:, :, :, :2] - coords_pred[:, :, :, :2], dim=3)**2 + \
        torch.linalg.norm(coords_gt[:, :, :, 2:] - coords_pred[:, :, :, 2:], dim=3)**2), torch.zeros_like(confs_gt)).sum(dim=1).mean(dim=0) 

        # sum for all cells, mean for all batches, only include cells where in the ground truth there are line segments i.e. conf_gt > 0
        L_resp = torch.where(confs_gt > 0, (confs_pred-torch.ones_like(confs_pred))**2, torch.zeros_like(confs_pred)).sum(dim=1).mean(dim=0) 

        # sum for all cells, mean for all batches, only include cells where in the ground truth there are no line segments i.e. conf_gt < 1 (penalize the error where the network is confident there is a gt_line when there is not)
        L_noresp = torch.where(confs_gt < 1, (confs_pred-torch.zeros_like(confs_pred))**2, torch.zeros_like(confs_pred)).sum(dim=1).mean(dim=0)

        # For normalization of the losses
        # num_cells = target_tensors.shape[1]
        # num_gt = (confs_gt > 0).squeeze(0).sum()

        # L_loc /= num_cells
        # L_resp /= num_gt
        # L_noresp /= (num_cells - num_gt)
        p = wandb.config.loss_param

        total_loss = p * L_loc + (1-p)/2 * (L_resp + L_noresp)

        return (L_loc, L_resp, L_noresp, total_loss)
    
    def forward(self, x, target_tensors=None):
        x = self.conv(x)
        x = self.reshape_tensor(x) # torch.Tensor([batch_size, num_predicted_channels, width, height]) => torch.Tensor([batch_size, num_cells, num_predictors, variables (coordinates + confidence)])

        if self.training:
            return self.get_losses(x, target_tensors) # TODO
        
        else:
            return x.sigmoid() # torch.Tensor([batch_size, num_cells, num_predictors, variables (coordinates + confidence)]) => torch.Tensor([batch_size, 400, 1, 5]) for dark5
        


