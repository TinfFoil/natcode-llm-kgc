#!/bin/bash
#SBATCH -J graphing
#SBATCH -p local
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log

# Define model_list and whether they are chat models
declare -A model_list=(
["unsloth/Meta-Llama-3.1-8B"]=false
["unsloth/Meta-Llama-3.1-8B-Instruct"]=true

# ["mistralai/Mistral-7B-v0.3"]=false
# ["mistralai/Mistral-7B-Instruct-v0.3"]=true

# ["deepseek-ai/deepseek-coder-7b-base-v1.5"]=false
# ["deepseek-ai/deepseek-coder-7b-instruct-v1.5"]=true

# ["Qwen/CodeQwen1.5-7B"]=false
# ["Qwen/CodeQwen1.5-7B-Chat"]=true
)

dataset_list=("ade" "conll04" "scierc")

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

# Loop through model_list
for natlang in ${natlang_toggle[@]}; do
    for rationale in ${rationale_toggle[@]}; do
        for model in "${!model_list[@]}"; do
            is_chat=${model_list[$model]}

            # Loop through dataset_list
            for dataset in "${dataset_list[@]}"; do
                # Base command
                cmd="python ./src/train.py \
                    --model $model \
                    --dataset $dataset"

                # Add chat flag if it's a chat model
                if $is_chat; then
                    cmd+=" --chat"
                fi

                if $rationale_toggle; then
                    cmd+=" --rationale"
                fi

                if $natlang_toggle; then
                    cmd+=" --natlang"
                fi

                # Log information
                log_info "[$(date +"%Y-%m-%d %H:%M:%S")] Training model: $model, dataset: $dataset, chat: $is_chat, natlang: $natlang, rationale: $rationale", 
                $cmd $1
            done
        done
    done
done