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
from yolox.models import YOLinOHead
from yolox.models import YOLOXHead, YOLOPAFPN, YOLOX

data_directory = '/home/manojlovska/Documents/YOLOX/datasets/DAIS-COCO'

train_data = DAISDataset(data_dir=data_directory, img_size=(640, 640))
print(train_data.__len__())
print(train_data[0])
print(train_data[0][0].shape)

model = Darknet(depth=53)
in_tensor = torch.rand((1, 3, 640, 640))
output = model(in_tensor)

print("Dark3: ", output["dark3"].size())
print("Dark4: ",output["dark4"].size())
print("Dark5: ",output["dark5"].size())

yolo_pafpn = YOLOPAFPN()
output_fpn = yolo_pafpn(in_tensor)
print("Output yolo_fpn: ", output_fpn)

yolino_head = YOLinOHead(in_channels=512, num_predictors_per_cell=1, num_classes=3)
yolox_head = YOLOXHead(num_classes=3)

output_head_yolino = yolino_head(output["dark5"])
print("Output head yolino: ", output_head_yolino)

yolino_head_fpn = YOLinOHead(in_channels=1024, num_predictors_per_cell=1, num_classes=3)
output_head_yolino_fpn = yolino_head_fpn(output_fpn[2])
print("Output head yolino fpn: ", output_head_yolino_fpn)

output_head_yolox = yolox_head(output_fpn)
print("Output head yolox: ", output_head_yolox.size())

darknet_yolino = YOLOX(backbone=yolo_pafpn, head=None, head_yolino=yolino_head)
output_all = darknet_yolino(in_tensor)