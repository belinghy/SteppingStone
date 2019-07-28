#!/bin/bash
set -e

git pull --recurse-submodules

project_path=$(pwd)
today=`date '+%Y_%m_%d__%H_%M_%S'`

name=$1
if [ $# -eq 0 ]
then
    echo "No arguments supplied: experiment name required"
    exit 1
fi
shift;

cd $project_path
log_path=runs/${today}__${name}
mkdir -p runs
mkdir $log_path

. ../venv/bin/activate
nohup python -m playground.train with experiment_dir="$log_path/" $@ &> $log_path/slurm.out &

PID=$!
echo "Process spawned with ID: $PID"
echo "Experiment directory: $log_path"
echo $PID > $log_path/pid

