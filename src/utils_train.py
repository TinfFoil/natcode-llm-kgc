from transformers.trainer_callback import TrainerControl, TrainerState, TrainerCallback
from transformers import TrainingArguments, BitsAndBytesConfig
import random
import torch
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import pandas as pd

def make_dataset(df: pd.DataFrame, runner, tokenizer, n_samples: int = 0):
    data_train = runner.make_samples(tokenizer, df)
    df = pd.DataFrame(data_train)
    if n_samples:
        df = df.sample(n=n_samples)
    return df

def prep_model(config, model):
    if config['do_train'] and config['lora_modules']:
        if config['load_in_4bit'] or config['load_in_8bit']:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config['lora_config'])
        print('Applied LoRA!')
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model

def get_trainer(config, model, tokenizer, dataset_train, dataset_dev = None):
    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        # compute_metrics=evaluator.compute_metrics,
        peft_config=config['lora_config'],
        args=SFTConfig(
            max_length=config['max_length'],
            dataset_num_proc=1,
            packing=False,
            per_device_train_batch_size=config['batch_size_train'],
            per_device_eval_batch_size=config['batch_size_eval'],
            # eval_accumulation_steps=1,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs = {"use_reentrant": False},
            warmup_steps=5,
            max_steps=config['train_steps'],
            # num_train_epochs=1,
            learning_rate=config['lr'],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear", 
            seed=config['seed'],
            # output_dir="outputs",
            report_to="none",
            eval_strategy='no',
            # eval_steps=config['eval_steps'],
            save_strategy='no',
            # save_steps=config['train_steps'] // 5,
            metric_for_best_model="f1",
            # load_best_model_at_end=True,
            label_names=["labels"],
        ),
    )

def get_quant_config(config):
    if config['load_in_4bit']:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
    elif config['load_in_8bit']:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None
    return quantization_config

class OutputPrinterCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, print_interval=10):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.print_interval = print_interval

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        if state.global_step % self.print_interval == 0:
            # Select a random sample from the dataset
            sample = random.choice(self.dataset)
            inputs = self.tokenizer(sample['text'], return_tensors='pt', truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Perform a forward pass instead of generation
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get the most likely token at each step
            predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
            
            # Decode the predicted tokens
            generated_text = self.tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
            
            print(f"\nStep {state.global_step} - Sample Output:")
            print(f"Input: {sample['text'][:100]}...")  # Print first 100 characters of input
            print(f"Model output: {generated_text}\n")

        return control

class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            log_message = {key: logs[key] for key in ['loss', 'grad_norm', 'learning_rate', 'epoch'] if key in logs}
            # print(log_message)
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # print(state.global_step)
        return super().on_step_begin(args, state, control, **kwargs)
