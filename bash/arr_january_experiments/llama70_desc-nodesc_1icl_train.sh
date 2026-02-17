#!/bin/bash
#SBATCH -J llama70_desc-nodesc_1icl_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
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
# Qwen/Qwen3-30B-A3B-Thinking-2507
# unsloth/Qwen3-32B
# unsloth/Qwen3-0.6B
# Qwen/Qwen3-14B-Base
# meta-llama/Llama-3.1-70B
meta-llama/Llama-3.1-70B-Instruct
# meta-llama/Llama-3.2-1B
# meta-llama/Llama-3.2-1B-Instruct
# meta-llama/Llama-3.2-3B
# meta-llama/Llama-3.2-3B-Instruct
# meta-llama/Llama-3.1-8B
# meta-llama/Llama-3.1-8B-Instruct
# meta-llama/Llama-3.3-70B-Instruct
# mistralai/Mistral-7B-v0.3  # this
# mistralai/Mistral-7B-Instruct-v0.3
)

declare -a seed=(
    0
    1
    2
    3
    4
    # 5
    # 6
    # 7
)

declare -a dataset=(
    ade
    conll04
    scierc
    erfgc
    scidtb
    enewt
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
    # "ft"
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

declare -a desc_schema=(
    0
    1
    )

declare -a prompt_filename=(
    default.yaml
    # uuid.yaml
    )

# Generate all combinations
array_names=(
            model
            seed
            dataset
            rationale
            lora_modules
            do_train
            n_icl_samples
            desc_schema
            prompt_filename
            )
combinations=$(cartesian_product array_names)

epochs=1
train_steps=0
eval_samples=100
load_in_4bit=1
load_in_8bit=0
save_prompt=1
lr=2e-4
save_prompt=1
verbose_preds=1
verbose_metrics=1
max_length=20000
max_new_tokens=5000
dtype_str=bfloat16

enable_thinking=0

date=$(date '+%Y%m%d%H%M%S')

batch_size_train=1
batch_size_eval=1
evaluate=0

# Convert combinations to commands
declare -a commands=()
declare -i count=0
while IFS= read -r combo; do
    IFS=',' read -ra params <<< "$combo"

    if [[ ${SLURM_ARRAY_JOB_ID} != '' ]]; then
        run_id="${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
    else
        run_id=${date}-${count}
    fi

    if [[ ${params[2]} == 'scidtb' || ${params[2]} == 'enewt' ]]; then
        eval_samples=100
    fi

    cmd="python ./src/train.py
                --model ${params[0]}
                --seed ${params[1]}
                --dataset ${params[2]}
                --rationale ${params[3]}
                --lora_modules ${params[4]}
                --do_train ${params[5]}
                --n_icl_samples ${params[6]}
                --desc_schema ${params[7]}
                --prompt_filename ${params[8]}
                --train_steps $train_steps
                --load_in_4bit $load_in_4bit
                --load_in_8bit $load_in_8bit
                --save_prompt $save_prompt
                --verbose_preds $verbose_preds
                --verbose_metrics $verbose_metrics
                --eval_samples $eval_samples
                --max_length $max_length
                --max_new_tokens $max_new_tokens
                --run_id ${run_id}
                --batch_size_train $batch_size_train
                --batch_size_eval $batch_size_eval
                --evaluate $evaluate
                --dtype_str $dtype_str
                --enable_thinking $enable_thinking
                "
    echo "$cmd" | sed -E 's/[[:space:]]+/ /g' | tr '\n' ' '
    echo
    echo
    commands+=("$cmd")
    count+=1
done <<< "$combinations"

total_combinations=${#commands[@]}

if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
    command_to_run="${commands[$SLURM_ARRAY_TASK_ID]}"
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
    echo "Use: sbatch --array=0-$((total_combinations-1)) ${BASH_SOURCE[0]}"
    echo "This will distribute $total_combinations jobs across N GPUs."
fi