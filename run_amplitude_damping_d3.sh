#!/bin/bash -l
#
#SBATCH --job-name="amp-d3"
#SBATCH --account=research-eemcs-diam
#SBATCH --partition=compute-p1
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=0
#SBATCH --mem-per-cpu=3968MB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=job_outputs/amp-d3.%j.out
#SBATCH --error=job_outputs/amp-d3.%j.err

module load 2024r1
module load python

export ENV_NAME="quantumsim-frankenstein"
source ${HOME}/virtual_environments_gpu/${ENV_NAME}/bin/activate
export PYTHONPATH="/home/$USER/virtual_environments_gpu/$ENV_NAME/lib/python3.10/site-packages/:$PYTHONPATH"

srun python -m example_circuits.action_simulate_d3_amplitude_damping

deactivate

