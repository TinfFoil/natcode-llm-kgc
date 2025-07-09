#!/bin/bash
#SBATCH -J train_kgc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log

module load gcc arrow
source venv-llmkgc/bin/activate

python $1