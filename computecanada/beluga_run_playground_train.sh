#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=400M
cd /home/belinghy/projects/def-vandepan/belinghy/rl-experiments/playground
. /home/belinghy/projects/def-vandepan/belinghy/research-env/bin/activate
python train.py
