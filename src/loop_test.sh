#!/bin/bash
#SBATCH -J test_kgc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:1
#SBATCH --time=72:00:00
#SBATCH --mem=256G
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log
nvidia-smi

declare -A model_list=(
    # non fine-tuned models
    # ["unsloth/Meta-Llama-3.1-8B"]=false
    # ["mistralai/Mistral-7B-v0.3"]=false
    # ["deepseek-ai/deepseek-coder-7b-base-v1.5"]=false
    # ["Qwen/CodeQwen1.5-7B"]=false

    ["unsloth/Meta-Llama-3.1-8B-Instruct"]=true
    ["mistralai/Mistral-7B-Instruct-v0.3"]=true
    ["deepseek-ai/deepseek-coder-7b-instruct-v1.5"]=true
    ["Qwen/CodeQwen1.5-7B-Chat"]=true

    # fine-tuned models
    ["./models/Meta-Llama-3.1-8B"]=false
    ["./models/Meta-Llama-3.1-8B-Instruct"]=true

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

natlang_toggle=(
    true
    false
)

rationale_toggle=(
    true
    false
)

train_steps=200
n_icl_samples=3
num_tests=3

command=$1
if [[ $command =~ --test_split[[:space:]]+([[:alnum:]_]+) ]]; then
    test_split=${BASH_REMATCH[1]}
else
    test_split='test'
fi

test_split_flag="--test_split ${test_split}"

for rationale in "${rationale_toggle[@]}"; do
    for natlang in "${natlang_toggle[@]}"; do
        for model in "${!model_list[@]}"; do
            is_chat_model=${model_list[$model]}
            
            if [[ $model == ./models/* ]]; then
                is_fine_tuned=true
                fine_tuned_flag='--fine_tuned'
                model_type_dir='fine-tuned'
            else
                is_fine_tuned=false
                fine_tuned_flag=''
                model_type_dir='base'
            fi

            echo "Fine-tuned model: $is_fine_tuned"

            for dataset in "${dataset_list[@]}"; do
                if $natlang; then
                    natlang_suffix="natlang"
                    natlang_flag="--natlang"
                else
                    natlang_suffix="code"
                    natlang_flag=""
                fi

                if $rationale; then
                    rationale_suffix="rationale"
                    rationale_flag="--rationale"
                else
                    rationale_suffix="base"
                    rationale_flag=""
                fi
                
                if $is_fine_tuned; then
                    model_name="${model}_ft_${dataset}_${natlang_suffix}_${rationale_suffix}_steps=${train_steps}_icl=${n_icl_samples}"
                    results_name=$model_name
                else
                    model_name="${model}"
                    results_name="${model}_${dataset}_${natlang_suffix}_${rationale_suffix}"
                fi

                echo "Current model: $model_name"

                if $is_chat_model; then
                    chat_flag="--chat"
                else
                    chat_flag=""
                fi

                # Check if the directory exists for fine-tuned models
                if [[ ! -d "$model_name" && "$is_fine_tuned" == "true" ]]; then
                    log_info "[$(date +"%Y-%m-%d %H:%M:%S")] Skipping model: $model_name - directory does not exist."
                    echo "Skipping model: $model_name - directory does not exist."
                    echo '*****************************'
                    continue
                fi

                cmd="python ./src/test.py -m $model_name \
                        -d $dataset \
                        $chat_flag"

                log_info "[$(date +"%Y-%m-%d %H:%M:%S")] Testing model: $model_name, dataset: $dataset, chat_model: $is_chat_model, language: $natlang_suffix, rationale: $rationale_suffix, fine-tuned: $is_fine_tuned"

                # Repeat the test until check_results.py reports num_tests == 3
                while ! python ./src/check_results.py \
                  -r "$results_name" \
                  -d "./results/${test_split}/${model_type_dir}" \
                  --split ${test_split} \
                  -n $num_tests; do

                    echo 'Running test command...'
                    $cmd $natlang_flag $rationale_flag $fine_tuned_flag $1
                    echo '*****************************'
                done

            done
        done
    done
done
