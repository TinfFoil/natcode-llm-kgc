from unsloth import FastLanguageModel
import torch
import os
import pandas as pd
import json
from datasets import Dataset
from tqdm.auto import tqdm

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model_name = "unsloth/llama-3-8b-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name, # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # 4x longer contexts auto supported!
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from utils import Prompter

dataset_name = 'ade'
# dataset_name = 'conll04'
# dataset_name = 'scierc'

dataset_path = f'./data/codekgc-data/{dataset_name}'
train_json = os.path.join(dataset_path, 'train_triples.json')

# schema_path = os.path.join(dataset_path, 'code_expl_prompt')
schema_path = os.path.join(dataset_path, 'code_prompt')
schema_prompt = open(schema_path, 'r', encoding='utf8').read()

n_icl_samples = 15

entity2type_json = os.path.join(dataset_path, 'entity2type.json')
with open(entity2type_json, 'r', encoding='utf8') as f:
    entity2type_dict = json.load(f)

prompter = Prompter(entity2type_dict,
                    natlang = False,
                    )

df_train = pd.read_json(train_json)

text_list = []
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
for index in tqdm(df_train.index, total=len(df_train)):
    # Select the specific row
    sample = df_train.loc[index]
    
    # Exclude the specific row from the DataFrame
    df_nosample = df_train.drop(index)
    text = sample['text']
    triples = sample['triple_list']
    
    # Randomly select N other different rows from the remaining DataFrame
    icl_rows = df_nosample.sample(n=n_icl_samples)
    icl_prompt = prompter.make_icl_prompt(icl_rows)
    text_input = text + '\n' + prompter.pythonize_triples(triples)
    prompt = prompter.make_prompt(schema_prompt, icl_prompt, text_input) + EOS_TOKEN
    text_list.append(prompt)

df = pd.DataFrame(text_list)
dataset = Dataset.from_pandas(df.rename(columns={0: "text"}), split="train")

# from datasets import load_dataset
# dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
# dataset = dataset.map(formatting_prompts_func, batched = True,)

from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)
from transformers import TrainerCallback

class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            log_message = {key: logs[key] for key in ['loss', 'grad_norm', 'learning_rate', 'epoch'] if key in logs}
            logger.info(log_message)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 2000,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
    callbacks=[PrinterCallback()],

)

trainer_stats = trainer.train()

# #@title Show final memory and time stats
# used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
# used_percentage = round(used_memory         /max_memory*100, 3)
# lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
# print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
# print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
# print(f"Peak reserved memory = {used_memory} GB.")
# print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
# print(f"Peak reserved memory % of max memory = {used_percentage} %.")
# print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
model_name_simple = model_name.split('/')[-1]
model_name_ft = f"{model_name_simple}_ft_{dataset_name}"

model.save_pretrained_merged(os.path.join('./models', model_name_ft), tokenizer, save_method = "merged_16bit",)
# model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

