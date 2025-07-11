#!/bin/bash
#SBATCH --time=1-00:00
#SBATCH --job-name=train_kgc
 
#SBATCH --nodes=1
 
# Request 1 process per GPU
#SBATCH --gpus-per-node=h100:2
#SBATCH --tasks-per-node=2
 
# Request more CPUs to enable multiple data-loader workers to load data in parallel.
#SBATCH --cpus-per-task=4
 
# Request the whole memory of the node
#SBATCH --mem=512G
 
#SBATCH --mail-user=paolo.gajo2@unibo.it
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log
 
#### LOAD MODULES ####
# module load scipy-stack
# module load python/3.9
module load StdEnv/2023
module load gcc/12.3
module load cudacore/.12.2.2
module load cuda/12.2
module load arrow
module load protobuf

source .env/bin/activate

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())"
nvcc --version

# # #### SET ENVIRONMENT VARS ####
# export PROJECT_DIR="$(pwd)"
# export DATA_DIR=${PROJECT_DIR}/data
# export CONFIG_DIR=${PROJECT_DIR}/configs
# export PYTHONPATH=$PYTHONPATH:${PROJECT_DIR}
export MASTER_ADDR=$(hostname)
export MASTER_PORT=34567
export TORCH_NCCL_BLOCKING_WAIT=1
export TRANSFORMERS_OFFLINE=1
# export HF_HOME=~/scratch/cache/huggingface/transformers
# export HF_DATASETS_CACHE=~/scratch/cache/huggingface/datasets
export WANDB_MODE=offline
export LOG_DIR=~/scratch/logs/${PROJECT_DIR}
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# #### CREATE VIRTUAL ENV ####
# if [[ -z "${SLURM_TMPDIR}" ]]; then
#   ENVDIR=/tmp/$RANDOM
# else
#   ENVDIR=$SLURM_TMPDIR/env
# fi
# virtualenv --no-download $ENVDIR
# source $ENVDIR/bin/activate
# pip3 install --no-index --upgrade pip
 
# #### INSTALL PACKAGES ####
# pip3 install --no-index transformers
# pip3 install --no-index datasets
# pip3 install --no-index sentencepiece
# pip3 install --no-index protobuf
# pip3 install --no-index torch
# pip3 install --no-index --find-links ~/scratch/packages/ -r requirements.txt
# pip3 install --no-index -U pandas numpy
 
# CHECK GPU AVAILABILITY
nvidia-smi

# Define model_list and whether they are chat models
# declare -A model_list=(
# ["unsloth/Meta-Llama-3.1-8B"]=false
# ["unsloth/Meta-Llama-3.1-8B-Instruct"]=true

# ["mistralai/Mistral-7B-v0.3"]=false
# ["mistralai/Mistral-7B-Instruct-v0.3"]=true

# ["deepseek-ai/deepseek-coder-7b-base-v1.5"]=false
# ["deepseek-ai/deepseek-coder-7b-instruct-v1.5"]=true

# ["Qwen/CodeQwen1.5-7B"]=false
# ["Qwen/CodeQwen1.5-7B-Chat"]=true
# )

# Define model_list and whether they are chat models
declare -A model_list=(
# ["unsloth/Meta-Llama-3.1-70B"]=false
# ["unsloth/Meta-Llama-3.1-70B-Instruct"]=true

# ["unsloth/Meta-Llama-3.1-8B"]=false
# ["unsloth/Meta-Llama-3.1-8B-Instruct"]=true

["meta-llama/Meta-Llama-3.1-8B-Instruct"]=true

# ["mistralai/Mistral-7B-v0.3"]=false
# ["mistralai/Mistral-7B-Instruct-v0.3"]=true

# ["deepseek-ai/deepseek-coder-7b-base-v1.5"]=false
# ["deepseek-ai/deepseek-coder-7b-instruct-v1.5"]=true

# ["Qwen/CodeQwen1.5-7B"]=false
# ["Qwen/CodeQwen1.5-7B-Chat"]=true

# ["Qwen/Qwen2.5-32B"]=false
# ["Qwen/Qwen2.5-32B-Instruct"]=true
)

dataset_list=(
    "ade"
    "conll04"
    "scierc"
    )

# Create a timestamp string
timestamp=$(date +"%Y%m%d_%H%M%S")

# Log file with unique name
log_file="./logging/training_log_$timestamp.log"

# Create or clear the log file
echo "" > $log_file

# Function to log information
log_info() {
    echo "$1" >> $log_file
}

rationale_toggle=(
    true
    false
)

natlang_toggle=(
    true
    false
)

train_steps=200
n_icl_samples=3
load_in_4bit=false

# lora_modules_list=("q-k-v-o-gate-up-down")
lora_modules_list=(
    # "q"
    # "k"
    # "v"
    # "q-k"
    # "q-v"
    # "k-v"
    # "q-k-v"
    "q-k-v-o-gate-up-down"
    # "full_ft"
    )

# Loop through model_list
for natlang in ${natlang_toggle[@]}; do
    for rationale in ${rationale_toggle[@]}; do
        for model in "${!model_list[@]}"; do
            is_chat=${model_list[$model]}
            # Loop through dataset_list
            for dataset in "${dataset_list[@]}"; do
                for lora_modules in "${lora_modules_list[@]}"; do
                    # Base command
                    cmd="torchrun --nproc_per_node=2 \
                        ./src/train_hf_parallel.py \
                        -m $model \
                        -d $dataset \
                        --n_icl_samples $n_icl_samples \
                        --lora_modules $lora_modules"

                    # Add chat flag if it's a chat model
                    if $is_chat; then
                        cmd+=" --chat"
                    fi

                    if $load_in_4bit; then
                        cmd+=" --load_in_4bit"
                    fi

                    if $rationale; then
                        cmd+=" --rationale"
                        rationale_suffix="rationale"
                        rationale_flag="--rationale"
                    else
                        rationale_suffix="base"
                        rationale_flag=""
                    fi

                    if $natlang; then
                        cmd+=" --natlang"
                        natlang_suffix="natlang"
                        natlang_flag="--natlang"
                    else
                        natlang_suffix="code"
                        natlang_flag=""
                    fi
                    
                    # Construct the model name
                    model_name="${model}_ft_${dataset}_${natlang_suffix}_${rationale_suffix}_steps=${train_steps}_icl=${n_icl_samples}_mod=${lora_modules}"
                    echo "Training this model: $model_name"
                    
                    if python ./src/check_models.py -m "$model_name" -d './models'; then
                        echo "$model_name has already been trained: skipping"
                        echo '**************'
                        continue
                    fi

                    # Log information
                    log_info "[$(date +"%Y-%m-%d %H:%M:%S")] Training model: $model, dataset: $dataset, chat: $is_chat, natlang: $natlang, rationale: $rationale", 
                    $cmd
                done
            done
        done
    done
done