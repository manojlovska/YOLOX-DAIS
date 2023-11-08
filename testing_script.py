import os

import torch
import torch.nn as nn
import torch.distributed as dist
from loguru import logger

from yolox.exp import Exp as MyExp

import os
import random

# from yolox.data.datasets import DAISDataset2
from yolox.data.datasets import DAISDataset
from yolox.models import Darknet

data_directory = '/home/manojlovska/Documents/YOLOX/datasets/DAIS-COCO'

train_data = DAISDataset(data_dir=data_directory, img_size=(640, 640))
print(train_data.__len__())
print(train_data[0])
print(train_data[0][0].shape)

model = Darknet(depth=53)
in_tensor = torch.rand((1, 3, 640, 640))
target_tensor = torch.rand((1, 3, 640, 640), dtype=float)
output = model(in_tensor)
print(output["dark3"].size())
print(output["dark4"].size())
print(output["dark5"].size())

print("test")

