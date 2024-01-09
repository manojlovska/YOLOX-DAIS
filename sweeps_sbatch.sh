#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1  
#SBATCH --mem=42G
#SBATCH --time=7-0:0:0
#SBATCH --partition=e7 
#SBATCH --job-name=yolino_head_agent
#SBATCH --output=yolino_head_agent%j.log
#SBATCH --reservation=e7
#SBATCH --export=WANDB_API_KEY
#SBATCH --export=HTTPS_PROXY
#SBATCH --export=https_proxy

export HTTPS_PROXY="http://www-proxy.ijs.si:8080"
export https_proxy="http://www-proxy.ijs.si:8080"
echo $HTTPS_PROXY
echo $https_proxy

echo "singularity exec --bind /ceph/grid/home/am6417/Projects/YOLOX-DAIS:/workspace/yolox --nv yolino_head_pip_sweeps.sif "$@""
singularity exec --bind /ceph/grid/home/am6417/Projects/YOLOX-DAIS:/workspace/yolox --nv yolino_head_pip_sweeps.sif "$@"
