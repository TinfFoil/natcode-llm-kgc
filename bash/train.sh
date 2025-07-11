#!/bin/bash
#SBATCH -J llm-parsing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=./.slurm/%A/%a_output.log
#SBATCH --error=./.slurm/%A/%a_error.log
#SBATCH --mem=64g
#SBATCH --array=0-N

slurm_dir="./.slurm/$SLURM_ARRAY_JOB_ID"
mkdir -p $slurm_dir
echo "Creating directory: $slurm_dir"
nvidia-smi
module load rust gcc arrow
. .env/bin/activate

cartesian_product() {
    local result=("")
    local -n arrays=$1
    
    for array_name in "${arrays[@]}"; do
        local -n current_array=$array_name
        local new_result=()
        
        for existing in "${result[@]}"; do
            for item in "${current_array[@]}"; do
                new_result+=("${existing:+$existing,}$item")
            done
        done
        result=("${new_result[@]}")
    done
    
    printf '%s\n' "${result[@]}"
}

declare -a model=(
# meta-llama/Llama-3.1-70B
# meta-llama/Llama-3.1-70B-Instruct
# meta-llama/Llama-3.2-1B
# meta-llama/Llama-3.2-1B-Instruct
# meta-llama/Llama-3.2-3B
# meta-llama/Llama-3.2-3B-Instruct
# meta-llama/Llama-3.1-8B
# meta-llama/Llama-3.1-8B-Instruct
# mistralai/Mistral-7B-v0.3
mistralai/Mistral-7B-Instruct-v0.3
)

seed=(
    0
    # 1
    # 2
    # 3
    # 4
)

dataset=(
    ade
    # conll04
    # scierc
    # erfgc
    # scidtb
    # enewt
    )

natlang=(
    1
    # 0
)

rationale=(
    # 1
    0
)

lora_modules=(
    # "q"
    # "k"
    # "v"
    # "q-k"
    # "q-v"
    # "k-v"
    "q-k-v"
    # "q-k-v-o-gate-up-down"
    # "full_ft"
    )

do_train=(
    1
    # 0
)

# Generate all combinations
array_names=(
            model
            seed
            dataset
            natlang
            rationale
            lora_modules
            do_train
            )
combinations=$(cartesian_product array_names)

train_steps=5
n_icl_samples=3
load_in_4bit=0
save_prompt=1
lr=2e-4
save_prompt=1
verbose_preds=1
verbose_metrics=1
# load_in_4bit=false
# load_in_8bit=false
# load_in_8bit=true

# lr=1e-5

# Convert combinations to commands
declare -a commands=()
while IFS= read -r combo; do
    IFS=',' read -ra params <<< "$combo"

    if [[ ${params[3]} == *"70B"* ]]; then
        load_in_4bit=1
    fi
    cmd="python ./src/train.py
                --model ${params[0]}
                --seed ${params[1]}
                --dataset ${params[2]}
                --natlang ${params[3]}
                --rationale ${params[4]}
                --lora_modules ${params[5]}
                --do_train ${params[6]}
                --train_steps $train_steps
                --n_icl_samples $n_icl_samples
                --load_in_4bit $load_in_4bit
                --save_prompt $save_prompt
                --verbose_preds $verbose_preds
                --verbose_metrics $verbose_metrics
                "
    # echo "$cmd"
    commands+=("$cmd")
done <<< "$combinations"

# for command in ${commands[@]}; do
#     echo $command
# done

total_combinations=${#commands[@]}

if [[ -n "$SLURM_ARRAY_TASK_ID" || $1 == 'override' ]]; then
    command_to_run="${commands[$SLURM_ARRAY_TASK_ID]}"
    echo "$command_to_run"
    $command_to_run
    {
        for array_name in "${array_names[@]}"; do
            # Access array by name using indirect expansion
            values="${array_name}[@]"
            echo "$array_name: ${!values}"
        done
    } > "${slurm_dir}/hyperparameters.txt"
else
    echo "This script should be run as a SLURM array job."
    echo "Use: sbatch --array=0-$((total_combinations-1)) $0"
    echo "This will distribute $total_combinations jobs across N GPUs."
fi