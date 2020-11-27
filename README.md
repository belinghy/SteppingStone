# ALLSTEPS: Curriculum-driven Learning of Stepping Stone Skills

Updated instructions for 2D environments.

## Quick Start

This library should run on Linux, Mac, or Windows.

### Install Requirements

```bash
# TODO: Create and activate virtual env

# Download the repo as well as the submodules
git clone https://github.com/belinghy/SteppingStone --recurse-submodules

# switch to walker2d branch, master branch is not updated yet
# 2d env in master branch is broken
cd SteppingStone
git checkout walker2d

# make sure mocca_envs is also updated
# should be on walker2d branch as well 
cd .environments; git checkout walker2d; cd ..

# install required libraries
pip install -r requirements
```

## Train from Scratch

To start a new training experiment named `test_experiment`:

```bash
# Walker2D, see plaground/train.py for arguments
./scripts/local_run_playground_train.sh walker_experiment \
    env='mocca_envs:Walker2DCustomEnv-v0'

# Crab2D
./scripts/local_run_playground_train.sh crab_experiment \
    env='mocca_envs:Crab2DCustomEnv-v0'
```

This command will create a new experiment directory inside the `runs` directory that contains the following files:

- `pid`: the process ID of the task running the training algorithm
- `progress.csv`: a CSV file containing the data about the the training progress
- `slurm.out`: the output of the process. You can use `tail -f` to view the contents
- `configs.json`: a JSON file containing all the hyper-parameter values used in this run
- `run.json`: extra useful stuff about the run including the host information and the git commit ID (only works if GitPython is installed)
- `models`: a directory containing the saved models

If you use [Compute Canada](http://computecanada.ca), we also have scripts like `cedar_run_playground_train.sh` to create a batch job. These scripts use the same argument sctructure but also allow you to run the same task with multiple replicates using the `num_replicates` variable.

### Run Trained Policies

The `enjoy.py` script can be used to run pretrained policies and render the results. Hit `r` in the PyBullet window to reset.

```bash
# Run Walker2D controller
python playground/enjoy.py --env mocca_envs:Walker2DCustomEnv-v0 \
    --net <experiment_path>/models/mocca_envs:Walker2DCustomEnv-v0_latest.pt

# See help for saving replay as video
# Needs either ffmpeg or moviepy (pip)
python playground/enjoy.py -h
```

## Plotting Results

The `plot_from_csv.py` script can be helpful for plotting the learning curves:

```bash
python -m playground.plot_from_csv --load_paths runs/*/ \
    --columns mean_rew max_rew  --smooth 2

# group results based on the name
python -m playground.plot_from_csv --load_paths runs/*/  \
    --columns mean_rew max_rew  --name_regex ".*__([^_\/])*" --group 1
```

- The `load_paths` argument specifies which directories the script should look.
- It opens the `progress.csv` file and plots the `columns` as the y-axis and uses the `row` for the x-axis (defaults to `total_num_steps`).
- You can also provide a `name_regex` to make the figure legends simpler and more readable, e.g. `--name_regex 'walker-(.*)\/'`.
- `group` can be used to aggregate the results of multiple runs of the same experiment into one. `name_regex` is used to specify the groups.

