#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=400M
cd /home/belinghy/projects/def-vandepan/belinghy/SteppingStone/playground
. /home/belinghy/projects/def-vandepan/belinghy/venv/bin/activate
python train.py
