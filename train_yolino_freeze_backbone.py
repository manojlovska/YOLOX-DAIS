import os

import torch
import torch.nn as nn
import torch.distributed as dist
from loguru import logger
import wandb
import os
import random

from dvclive import Live
import wandb 

from train_yolino import Exp as BaseExp

os.environ['WANDB_PROJECT'] = 'YOLOX-YOLinO'
run = wandb.init(project='YOLOX-YOLinO')

class Exp(BaseExp):
    def __init__(self):
        super(Exp, self).__init__()

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 80
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0 
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9

        # When training YOLinO set mag_tape to True
        self.mag_tape = True

        # --------------- transform config ----------------- #
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True
        self.sweeps = False


        # --------------- basic config ----------------- #
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 3
        self.data_num_workers = 1
        self.input_size = (640, 640)
        self.random_size = (10, 20)
        self.test_size = (640, 640)
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # modify 'silu' to 'relu' for deployment on DPU
        self.output_dir = "./YOLOX_outputs"
        self.wandb_name = run.name
        self.act = 'relu'
        self.thresh_lr_scale = 10
        # self.device = torch.device('cuda:1')
        torch.backends.cudnn.enabled = False

        logger.info("GPU MEMORY AVAILABLE: " + str(torch.cuda.mem_get_info()))

        # --------------- dataset path config ----------------- #
        self.data_dir = 'datasets/DAIS-COCO'
        self.train_ann = 'instances_train.json'
        self.val_ann = 'instances_valid.json'

    def get_model(self):
        from yolox.utils import freeze_module
        model = super().get_model()
        freeze_module(model.backbone.backbone)
        return model