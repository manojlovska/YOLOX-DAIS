# YOLOX-DAIS
## Introduction
This project focuses on training the YOLOX model for the task of object detection of people and forklifts, and additionally for the task of magnetic tape detection.
The original README file can be found [here](https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/README.md).

## Steps to reproduce
### Step 1: Clone the repository
```shell
git clone git@github.com:manojlovska/YOLOX-DAIS.git
```

### Step 2: Create virtual environment with conda and activate it
```shell
conda create -n env_name python=3.8.5
conda activate env_name
```

### Step 3: Install the yolox module and the requirements
```shell
cd YOLOX-DAIS
pip install -v -e .
pip install -r requirements.txt
```

### Step 4: Download the DAIS dataset
```shell
cd datasets/
# TODO
```

### Step 5: Download the pretrained Darknet53 model
The link for downloading the pretrained YOLOX-Darknet53 model is provided [here](https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/README.md#standard-models).

### Step 6: Train the model for object detection
```shell
python tools/train.py -f train_custom_data.py -d 1 -b 16 --fp16 -c /path/to/your/yolox_darknet.pth
```
* -f: configuration file of the experiment
* -d: number of GPU devices
* -b: batch size
* --fp16: mixed precision training
* -c: .pth file of the pretrained model
  
**(Optional)** If you want to visualize the training in real-time, use the following command:
```shell
python tools/train.py -f train_custom_data.py -d 1 -b 16 --fp16 -c /path/to/your/yolox_darknet.pth --logger wandb wandb-project <project-name>
```
**NOTE:** You have to change the name of the project in the *train_custom_data.py* as well:

```shell
os.environ['WANDB_PROJECT'] = 'project-name'
run = wandb.init(project='project-name')
```

### Step 7: Train the model for magnetic tape detection
For training the magnetic tape detection head, we use the best trained model for object detection and freeze its weights. Use the following command:
```shell
python tools/train.py -f train_yolino_freeze_backbone.py -d 1 -b 16 -c /path/to/your/best_ckpt.pth --logger wandb wandb-project <project-name>
```
Again, logging to Weights and Biases is optional, and you have to change the project name in the *train_yolino_freeze_backbone.py* script as well as before.

### Training the model for magnetic tape detection only
For training the model for magnetic tape detection only, without freezing the backbone, please use the following command:
```shell
python tools/train.py -f train_yolino.py -d 1 -b 8 --fp16 -c /path/to/your/yolox_darknet.pth --logger wandb wandb-project <project_name>
```

## Hyperparameter optimization of magnetic tape detection head with W&B Sweeps
### Training on a PC
### Step 1: Change the parameters
* In *train_yolino.py* change the **basic_lr_per_img** parameter
```shell
self.basic_lr_per_img = 0.01 / 64.0     =>     self.basic_lr_per_img = wandb.config.lr / 64.0
```
* In */yolox/models/yolino_head.py* change the **p** parameter
```shell
p = 0.5     =>     p = wandb.config.loss_param
```
* In *train_yolino.py*, function *get_dataset()* change the **sweeps** parameter
```shell
return DAISDataset(
    data_dir=self.data_dir,
    json_file=self.train_ann,
    img_size=self.input_size,
    mag_tape=self.mag_tape,
    preproc=TrainTransformYOLinO(sweeps=True),
    cache=cache,
    cache_type=cache_type,
)
```
### Step 2: Initialize a wandb sweep
```shell
wandb sweep --project <propject-name> wandb_sweeps.yaml
```
### Step 3: Start the sweep agent
```shell
wandb agent <sweep-ID>
```
### Training on SLING
### Step 1: Change the parameters
Change the parameters as mentioned in the previous chapter.

### Step 2: Build the Singularity container, if not already built
```shell
sudo singularity build container_name.sif Singularity.def
```
### Step 3: Initialize a wandb sweep
On the login node initialize the wandb sweep.
```shell
wandb sweep wandb_sweeps.yaml
```
### Step 4: Run the sbatch script
```shell
sbatch sweeps_sbatch.sh wandb agent <sweep-ID>
```
**NOTE:** In the *sweeps_sbatch.sh* script adapt the directives according to your specific requirements. Also, be careful to bind the correct path to the yolox module.
```shell
singularity exec --bind /path/to/your/YOLOX-DAIS:/workspace/yolox --nv container_name.sif "$@"
```


