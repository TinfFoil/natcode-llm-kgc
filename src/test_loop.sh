#!/bin/bash
#SBATCH -J graphing
#SBATCH -p local
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log

declare -A model_list=(
# non fine-tuned models
# ["unsloth/Meta-Llama-3.1-8B"]=false
# ["mistralai/Mistral-7B-v0.3"]=false
# ["deepseek-ai/deepseek-coder-7b-base-v1.5"]=false
# ["Qwen/CodeQwen1.5-7B"]=false

# ["unsloth/Meta-Llama-3.1-8B-Instruct"]=true
# ["mistralai/Mistral-7B-Instruct-v0.3"]=true
# ["deepseek-ai/deepseek-coder-7b-instruct-v1.5"]=true
# ["Qwen/CodeQwen1.5-7B-Chat"]=true

# fine-tuned models
# ["./models/Meta-Llama-3.1-8B"]=false
# ["./models/Meta-Llama-3.1-8B-Instruct"]=true

["./models/Mistral-7B-v0.3"]=false
["./models/Mistral-7B-Instruct-v0.3"]=true

["./models/deepseek-coder-7b-base-v1.5"]=false
["./models/deepseek-coder-7b-instruct-v1.5"]=true

["./models/CodeQwen1.5-7B"]=false
["./models/CodeQwen1.5-7B-Chat"]=true

)

dataset_list=(
    'ade'
    'conll04'
    'scierc'
)

# Create a timestamp string
timestamp=$(date +"%Y%m%d_%H%M%S")

# Log file with unique name
log_file="./logging/testing_log_$timestamp.log"

# Create or clear the log file
echo "" > $log_file

# Function to log information
log_info() {
    echo "$1" >> $log_file
}

langs=(
    # true
    false
)

for natlang in "${langs[@]}"; do
    for model in "${!model_list[@]}"; do
        for dataset in "${dataset_list[@]}"; do
            if $natlang; then
                natlang_suffix="natlang"
                natlang_flag="--natlang"
            else
                natlang_suffix="code"
                natlang_flag=""
            fi
            
            # Construct the model name with parameter expansion
            model_name="${model}_ft_${dataset}_${natlang_suffix}"

            is_chat_model=${model_list[$model]}
            # Check if the model is a chat model
            if $is_chat_model; then
                chat_flag="--chat"
            else
                chat_flag=""
            fi

            cmd="python ./src/test.py -m $model_name \
                    -d $dataset \
                    $chat_flag"

            # echo Model name: $model_name
            # echo Dataset name: $dataset
            # echo Chat model: $is_chat_model
            # echo Language: natlang
            # echo \*\*\*\*\*\*\*\*\*\*\*\*

            # Log information
            log_info "[$(date +"%Y-%m-%d %H:%M:%S")] Testing model: $model, dataset: $dataset, chat_model: $is_chat_model, language: natlang"
            # Run with natlang
            $cmd $natlang_flag

            # echo Model name: $model_name
            # echo Dataset name: $dataset
            # echo Chat model: $is_chat_model
            # echo Language: code
            # echo \*\*\*\*\*\*\*\*\*\*\*\*
            
            # # Log information
            # log_info "[$(date +"%Y-%m-%d %H:%M:%S")] Testing model: $model, dataset: $dataset, chat_model: $is_chat_model, language: code"
            # # Run without natlang
            # $cmd
        done
    done
done