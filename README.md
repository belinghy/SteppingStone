# (WIP) ALLSTEPS: Curriculum-driven Learning of Stepping Stone Skills

## Installation

There is no need for compilation. You can install all requirements using Pip, however, you might prefer to install some manully, including:
 - [PyTorch](https://pytorch.org/get-started/locally/)
 - [PyBullet](https://pybullet.org)

### Installation using Pip
```bash
# TODO: create and activate your virtual env of choice

# download the repo as well as the submodules (including )
git clone https://github.com/belinghy/SteppingStone --recurse-submodules

cd SteppingStone
pip install -r requirements  # you might prefer to install some packages (including PyTorch) yourself
```

## Running Locally

To run an experiment named `test_experiment` with the MikeStepper environment you can run:

```bash
./scripts/local_run_playground_train.sh test_experiment  env_name='mocca_envs:MikeStepperEnv-v0'

# run with curriculum (see plaground/train.py for more options)
./scripts/local_run_playground_train.sh  curriculum_experiment  env_name='mocca_envs:MikeStepperEnv-v0' use_curriculum=True
```

This command will create a new experiment directory inside the `runs` directory that contains the following files:

- `pid`: the process ID of the task running the training algorithm
- `progress.csv`: a CSV file containing the data about the the training progress
- `slurm.out`: the output of the process. You can use `tail -f` to view the contents
- `configs.json`: a JSON file containing all the hyper-parameter values used in this run
- `run.json`: extra useful stuff about the run including the host information and the git commit ID (only works if GitPython is installed)
- `models`: a directory containing the saved models

In case you use [Compute Canada](http://computecanada.ca) you also use the other scripts like `cedar_run_playground_train.sh` to create a batch job. These scripts use the same argument sctructure but also allow you to run the same task with multiple replicates using the `num_replicates` variable.

## Plotting Results

The `plot_from_csv.py` script can be helpful for plotting the learning curves:

```bash
python -m playground.plot_from_csv --load_paths runs/*/*/  --columns mean_rew max_rew  --smooth 2

# group results based on the name
python -m playground.plot_from_csv --load_paths runs/*/*/  --columns mean_rew max_rew  --name_regex ".*__([^_\/])*" --group 1
```

- The `load_paths` argument specifies which directories the script should look into
- It opens the `progress.csv` file and plots the `columns` as the y-axis and uses the `row` for the x-axis (defaults to `total_num_steps`)
- You can also provide a `name_regex` to make the figure legends simpler and more readable, e.g. `--name_regex 'walker-(.*)mirror\/'` would turn `runs/2019_07_08__23_53_20__walker-lossmirror/1` to simply `loss`.
- `group` can be used to aggregate the results of multiple runs of the same experiment into one. `name_regex` is used to specify the groups.

## Running Learned Policy

The `enjoy.py` script can be used to run a learned policy and render the results:

```bash
# run pretrained Mike controller
python -m playground.enjoy --env mocca_envs:MikeStepperEnv-v0 --net playground/models/mocca_envs:MikeStepperEnv-v0_latest.pt

# run pretrained Walker3D controller
python -m playground.enjoy --env mocca_envs:Walker3DStepperEnv-v0 --net playground/models/mocca_envs:Walker3DStepperEnv-v0_latest.pt
```


## Citation
Please cite the following paper if you found our work useful. The preprint is available on [ArXiv](https://arxiv.org/abs/2005.04323). 