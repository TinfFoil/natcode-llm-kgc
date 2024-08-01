from transformers.trainer_callback import TrainerControl, TrainerState
from unsloth import FastLanguageModel
import torch
import os
import pandas as pd
import json
from datasets import Dataset
from tqdm.auto import tqdm
from utils import Runner
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
import argparse
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

def main(args):

    max_seq_length = args.max_seq_length
    dtype = args.dtype
    load_in_4bit = args.load_in_4bit
    model_name = args.model_name
    chat_model = args.chat_model
    natlang = args.natlang
    dataset_name = args.dataset_name

    print(f'Training model: {model_name}')
    print(f'Chat model: {chat_model}')
    print(f"Language: {'natlang' if natlang else 'code'}")
    print(f"Training data: {dataset_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name, # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit
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

    dataset_path = f'./data/codekgc-data/{dataset_name}'

    entity2type_json = os.path.join(dataset_path, 'entity2type.json')
    with open(entity2type_json, 'r', encoding='utf8') as f:
        entity2type_dict = json.load(f)

    runner = Runner(entity2type_dict,
                        natlang=natlang,
                        tokenizer=tokenizer,
                        chat_model=chat_model,
                        schema_path = os.path.join(dataset_path, 'code_prompt')
                        )

    n_icl_samples = 15

    train_json = os.path.join(dataset_path, 'train_triples.json')
    df_train = pd.read_json(train_json)
    text_list_train = runner.make_samples(tokenizer, df_train, n_icl_samples=n_icl_samples)
    df_train = pd.DataFrame(text_list_train)
    dataset_train = Dataset.from_pandas(df_train.rename(columns={0: "text"}), split="train")
    
    # val_json = os.path.join(dataset_path, 'val_triples.json')
    # df_val = pd.read_json(val_json)
    # text_list_val = runner.make_samples(tokenizer, df_val, n_icl_samples=n_icl_samples)
    # df_val = pd.DataFrame(text_list_val)
    # dataset_val = Dataset.from_pandas(df_val.rename(columns={0: "text"}), split="train")

    class PrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                log_message = {key: logs[key] for key in ['loss', 'grad_norm', 'learning_rate', 'epoch'] if key in logs}
                logger.info(log_message)
        def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            logger.info(state.global_step)
            return super().on_step_begin(args, state, control, **kwargs)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset_train,
        # eval_dataset = dataset_val,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 8,
            per_device_eval_batch_size = 8,
            gradient_accumulation_steps = 1,
            warmup_steps = 5,
            max_steps = 200,
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
            # eval_strategy = 'steps',
            # eval_steps = 10,
        ),
        callbacks=[PrinterCallback()],

    )

    trainer.train()

    model_name_simple = model_name.split('/')[-1]
    model_name_ft = f"{model_name_simple}_ft_{dataset_name}_{'natlang' if natlang else 'code'}"

    model.save_pretrained_merged(os.path.join('./models', model_name_ft), tokenizer, save_method = "merged_16bit",)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument("--model_name", type=str, help="Name of the model to train", default='unsloth/Meta-Llama-3.1-8B')
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to use", default='ade')
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--dtype", type=str, default=None, help="Data type for training")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--chat_model", action="store_true", help="Whether it's a chat model")
    parser.add_argument("--natlang", action="store_true", help="Use natural language prompts")
    args = parser.parse_args()

    # args.natlang = True
    # args.chat_model = True
    main(args)