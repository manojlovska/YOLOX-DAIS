#!/bin/bash
CUDA_VISIBLE_DEVICES=0 wandb agent $1 --project $2
CUDA_VISIBLE_DEVICES=1 wandb agent $1 --project $2
CUDA_VISIBLE_DEVICES=2 wandb agent $1 --project $2
CUDA_VISIBLE_DEVICES=3 wandb agent $1 --project $2