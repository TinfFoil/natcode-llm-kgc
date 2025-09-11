#!/bin/bash
#SBATCH -J infextr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --output=./.slurm/%A/%a_output.log
#SBATCH --error=./.slurm/%A/%a_error.log
#SBATCH --mem=64g
#SBATCH --array=0-N

slurm_dir="./.slurm/$SLURM_ARRAY_JOB_ID"
mkdir -p $slurm_dirs
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
# meta-llama/Llama-3.3-70B-Instruct
# mistralai/Mistral-7B-v0.3
mistralai/Mistral-7B-Instruct-v0.3
# Qwen/Qwen3-30B-A3B-Thinking-2507
)

declare -a seed=(
    0
    # 1
    # 2
    # 3
    # 4
    # 5
    # 6
    # 7
)

declare -a dataset=(
    ade
    # conll04
    # scierc
    # erfgc
    # scidtb
    # enewt
    )

declare -a natlang=(
    1
    # 0
)

declare -a rationale=(
    # 1
    0
)

declare -a lora_modules=(
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
    # 0
    1
)

declare -a n_icl_samples=(
    # 0
    1
    # 2
    # 3
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
            n_icl_samples
            )
combinations=$(cartesian_product array_names)

train_steps=100
eval_steps=5
load_in_4bit=0
save_prompt=1
lr=2e-4
save_prompt=1
verbose_preds=1
verbose_metrics=1
max_length=10000
max_new_tokens=5000
# load_in_4bit=false
# load_in_8bit=false
# load_in_8bit=true

date=$(date '+%Y%m%d%H%M%S')

batch_size_train=4
batch_size_eval=4
evaluate=0

# lr=1e-5

# Convert combinations to commands
declare -a commands=()
declare -i count=0
while IFS= read -r combo; do
    IFS=',' read -ra params <<< "$combo"

    # if [[ ${params[3]} == *"70B"* ]]; then
    #     load_in_4bit=1
    # fi
    # if [[ ${params[2]} == "ade" || ${params[2]} == "conll04" || ${params[2]} == "scierc" ]]; then
    #     n_icl_samples=3
    # else
    #     n_icl_samples=1
    # fi
    if [[ ${SLURM_ARRAY_JOB_ID} != '' ]]; then
        run_id="icl_${params[7]}_${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
    else
        run_id=${date}-${count}
    fi
    cmd="python ./src/train.py
                --model ${params[0]}
                --seed ${params[1]}
                --dataset ${params[2]}
                --natlang ${params[3]}
                --rationale ${params[4]}
                --lora_modules ${params[5]}
                --do_train ${params[6]}
                --n_icl_samples ${params[7]}
                --train_steps $train_steps
                --load_in_4bit $load_in_4bit
                --save_prompt $save_prompt
                --verbose_preds $verbose_preds
                --verbose_metrics $verbose_metrics
                --eval_steps $eval_steps
                --max_length $max_length
                --max_new_tokens $max_new_tokens
                --run_id ${run_id}
                --batch_size_train $batch_size_train
                --batch_size_eval $batch_size_eval
                --evaluate $evaluate
                "
                # --run_id ${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
    # echo "$cmd"
    commands+=("$cmd")
    count+=1
done <<< "$combinations"

# for command in ${commands[@]}; do
#     echo $command
# done

total_combinations=${#commands[@]}

if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
    command_to_run="${commands[$SLURM_ARRAY_TASK_ID]}"
    # echo "$command_to_run"
    $command_to_run
elif [[ $1 ]]; then
    for (( i=start; i<${#commands[@]}; i++ ))
    do
        echo "$((i+1)) of ${#commands[@]}"
        cmd="${commands[$i]}"
        echo "${cmd}"
        $cmd
    done
else
    echo "This script should be run as a SLURM array job."
    echo "Use: sbatch --array=0-$((total_combinations-1)) $0"
    echo "This will distribute $total_combinations jobs across N GPUs."
fi