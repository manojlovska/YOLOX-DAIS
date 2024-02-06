import wandb
import subprocess
import yaml
import os

# Set API key
if os.path.exists("/ceph/grid/home/am6417/.wandb_key.txt"):
    file = open("/ceph/grid/home/am6417/.wandb_key.txt", "r")
    api_key = file.read()
    os.environ["WANDB_API_KEY"] = api_key
    file.close()

# Gather nodes allocated to current slurm job
result = subprocess.run(['scontrol', 'show', 'hostnames'], stdout=subprocess.PIPE)
node_list = result.stdout.decode('utf-8').split('\n')[:-1]

config_yaml = "/ceph/grid/home/am6417/Projects/YOLOX-DAIS/wandb_sweeps.yaml"
project_name = "YOLinO-DAIS-SLING"


def run(config_yaml, project_name):

    wandb.init(project=project_name)

    with open(config_yaml) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(config_dict, project=project_name)

    sp = []
    for node in node_list:
        sp.append(subprocess.Popen(['srun',
                                    '--nodes=1',
                                    '--ntasks=4',
                                    '-w',
                                    node,
                                    'start-agent.sh',
                                    sweep_id,
                                    project_name], shell=True))
    exit_codes = [p.wait() for p in sp]  # wait for processes to finish
    return exit_codes


if __name__ == '__main__':
    run()
