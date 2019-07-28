#!/bin/bash
set -e

git pull --recurse-submodules

NUM_REPLICATES=1

# One folder above the folder containing this file
PROJECT_PATH=$(dirname $(dirname $(realpath -s $0)))
EXPERIMENT_PATH=$PROJECT_PATH
TODAY=`date '+%Y_%m_%d__%H_%M_%S'`

NAME=$1
if [ $# -eq 0 ]
then
    echo "No arguments supplied: experiment name required"
    exit 1
fi
shift;

LOG_PATH=$EXPERIMENT_PATH/runs/${TODAY}__${NAME}
mkdir -p $LOG_PATH
cat > $LOG_PATH/run_script.sh <<EOF
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=32000M
#SBATCH --job-name=$NAME
#SBATCH --array=1-$NUM_REPLICATES
. $PROJECT_PATH/../venv/bin/activate
cd $PROJECT_PATH
python -m playground.train with experiment_dir="$LOG_PATH/\$SLURM_ARRAY_TASK_ID" replicate_num=\$SLURM_ARRAY_TASK_ID $@
EOF

cd $LOG_PATH

for ((i=1;i<=$NUM_REPLICATES;i++)) do
    mkdir $i
done

sbatch run_script.sh
echo "Logging at: $LOG_PATH"
