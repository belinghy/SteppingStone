# ALLSTEPS: Curriculum-driven Learning of Stepping Stone Skills

This repo is the codebase for the SCA 2020 paper with the title above. The full talk is available [here](https://www.youtube.com/watch?v=lMNH4xk9c1I).

## Quick Start

This library should run on Linux, Mac, or Windows.

### Install Requirements

```bash
# TODO: Create and activate virtual env

# Download the repo as well as the submodules
git clone https://github.com/belinghy/SteppingStone --recurse-submodules

cd SteppingStone
pip install -r requirements
```

### Run Pretrained Policies

The `enjoy.py` script can be used to run pretrained policies and render the results. Hit `r` in the PyBullet window to reset.

```bash
# Run Mike controller
python -m playground.enjoy --env mocca_envs:MikeStepperEnv-v0 \
    --net playground/models/mocca_envs:MikeStepperEnv-v0_latest.pt

# Run Walker3D controller
python -m playground.enjoy --env mocca_envs:Walker3DStepperEnv-v0 \
    --net playground/models/mocca_envs:Walker3DStepperEnv-v0_latest.pt
```

## Train from Scratch

To start a new training experiment named `test_experiment` for the MikeStepper environment you can run:

```bash
./scripts/local_run_playground_train.sh test_experiment \
    env_name='mocca_envs:MikeStepperEnv-v0'

# Train with curriculum (see plaground/train.py for arguments)
./scripts/local_run_playground_train.sh  curriculum_experiment \
    env_name='mocca_envs:MikeStepperEnv-v0' use_curriculum=True
```

This command will create a new experiment directory inside the `runs` directory that contains the following files:

- `pid`: the process ID of the task running the training algorithm
- `progress.csv`: a CSV file containing the data about the the training progress
- `slurm.out`: the output of the process. You can use `tail -f` to view the contents
- `configs.json`: a JSON file containing all the hyper-parameter values used in this run
- `run.json`: extra useful stuff about the run including the host information and the git commit ID (only works if GitPython is installed)
- `models`: a directory containing the saved models

If you use [Compute Canada](http://computecanada.ca), we also have scripts like `cedar_run_playground_train.sh` to create a batch job. These scripts use the same argument sctructure but also allow you to run the same task with multiple replicates using the `num_replicates` variable.

## Plotting Results

The `plot_from_csv.py` script can be helpful for plotting the learning curves:

```bash
python -m playground.plot_from_csv --load_paths runs/*/*/ \
    --columns mean_rew max_rew  --smooth 2

# group results based on the name
python -m playground.plot_from_csv --load_paths runs/*/*/  \
    --columns mean_rew max_rew  --name_regex ".*__([^_\/])*" --group 1
```

- The `load_paths` argument specifies which directories the script should look.
- It opens the `progress.csv` file and plots the `columns` as the y-axis and uses the `row` for the x-axis (defaults to `total_num_steps`).
- You can also provide a `name_regex` to make the figure legends simpler and more readable, e.g. `--name_regex 'mike-(.*)\/'`.
- `group` can be used to aggregate the results of multiple runs of the same experiment into one. `name_regex` is used to specify the groups.

## Citation

Please cite the following paper if you find our work useful.

```bibtex
@inproceedings{2020-SCA-ALLSTEPS,
  title={ALLSTEPS: Curriculum-driven Learning of Stepping Stone Skills}
  author={Xie, Zhaoming and Ling, Hung Yu and Kim, Nam Hee and van de Panne, Michiel},
  booktitle = {Proc. ACM SIGGRAPH / Eurographics Symposium on Computer Animation},
  year={2020}
}
```

The preprint is also available on [ArXiv](https://arxiv.org/abs/2005.04323).
