import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from utils_io import setup_config, save_json, get_time
from utils_train import get_quant_config
from runner import Runner
import argparse
import os
import pandas as pd
import json
import yaml

def main(args):
    config = setup_config(args, json.load(open(config['config_path'], 'r')))
    
    dataset_path = f'./data/parsing/{config['dataset']}/rdf'
    schema_path = os.path.join(dataset_path, config['prompt_filename'])
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'],
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    chat_template_dict = yaml.load(open('../model_info/chat_templates.yaml'))
    if tokenizer.chat_template is None and hasattr(model.config, 'model_type'):
        tokenizer.chat_template = chat_template_dict.get(model.config.model_type)
        print(f"Chat template not found, using the one for model type \"{model.config.model_type}\"")

    train_json = os.path.join(dataset_path, 'train.json')
    df_train = pd.read_json(train_json)
    text_list_train = runner.make_samples(tokenizer, df_train, n_icl_samples=config['n_icl_samples'])
    df_train = pd.DataFrame(text_list_train)
    dataset_train = Dataset.from_pandas(df_train.rename(columns={0: "text"}), split="train")

    if config['val']:
        val_json = os.path.join(dataset_path, 'val.json')
        df_val = pd.read_json(val_json)
        text_list_val = runner.make_samples(tokenizer, df_val, n_icl_samples=config['n_icl_samples'])
        n_samples_val = len(df_val) if not config['val_samples'] else config['val_samples']
        df_val = pd.DataFrame(text_list_val).sample(n=n_samples_val)
        dataset_val = Dataset.from_pandas(df_val.rename(columns={0: "text"}), split="val")
        eval_strategy = 'steps'
        load_best_model_at_end=True
    else:
        n_samples_val = 0
        dataset_val = None
        eval_strategy = 'no'
        load_best_model_at_end=False

    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        quantization_config=get_quant_config(config),
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    print(model)
    model.gradient_checkpointing_enable()

    if config['do_train'] and config['target_modules'] != 'full_ft':
        if not config['train_steps']:
            config['train_steps'] = len(dataset_train) // int(config['batch_size_train'])
        target_modules = [el+'_proj' for el in config['target_modules'].split('-')]
        if config['load_in_4bit'] or config['load_in_8bit']:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=False,
        )
        model = get_peft_model(model, lora_config)
        print('Applied LoRA!')
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset_train,
            # eval_dataset=dataset_dev,
            # compute_metrics=evaluator.compute_metrics,
            peft_config=lora_config,
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
                # save_steps=config['train_steps'] // 5,
                metric_for_best_model="f1",
                # load_best_model_at_end=True,
                label_names=["labels"],
            ),
        )

    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    entity2type_json = os.path.join(dataset_path, config['entitytypes'])
    with open(entity2type_json, 'r', encoding='utf8') as f:
        entity2type_dict = json.load(f)

    runner = Runner(model=model,
                    type_dict=entity2type_dict,
                    natlang=config['natlang'],
                    tokenizer=tokenizer,
                    chat_model=config['chat'],
                    schema_path=schema_path,
                    rationale=config['rationale'],
                    verbose_train=config['verbose_train'],
                    max_seq_len=config['max_length'],
                    model_name=config['model_name'],
                    )
    # icl_prompt = runner.make_icl_prompt(df_train)
    
    print('Model system message:', runner.check_system_msg())

    for epoch in config['epochs']:
        trainer_stats = trainer.train()

        best_metric = trainer.state.best_metric
        print(f"Best F1 score: {best_metric}")

        if config['val']:
            eval_results = runner.evaluate(df_val,
                                           df_train, # TODO: this shit ain't working properly for sure
                                           batch_size=config['batch_size_eval'])
            print(f"Final evaluation results: {eval_results}")
        else:
            eval_results = {
                'eval_loss': -1,
                'eval_precision': -1,
                'eval_recall': -1,
                'eval_f1': -1,
            }

    results_dir_path = './results'

    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)
    
    model_save_path = os.path.join(model_dir, config['model_name'].replace('/', '-'), config['run_id'], get_time())
    
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    config_path = os.path.join(model_dir, 'config.json')
    print(f'Config saved to: {config_path}')
    save_json(config, config_path)

    if config['save_model']:
        # Standard model saving instead of Unsloth's save_pretrained_merged
        if config['target_modules'] != 'full_ft':
            # For LoRA models, save the adapter
            model.save_pretrained(model_save_path)
        else:
            # For full fine-tuned models
            model.save_pretrained(model_save_path, safe_serialization=True)
        tokenizer.save_pretrained(model_save_path)
        print(f'Fine-tuned model saved to: {model_save_path}')
    else:
        print('Model was not saved because of --save_model 0 flag')

    print('VRAM usage:',torch.cuda.memory_allocated())
    print('Model deleted, test time...')
    model.to('cpu')
    del model
    torch.cuda.empty_cache()
    print('VRAM usage:',torch.cuda.memory_allocated())
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument("--model_name", type=str, help="Name of the model to train", default='mistralai/Mistral-7B-Instruct-v0.3')
    parser.add_argument("--dataset", type=str, help="Name of the dataset to use", default='ade')
    parser.add_argument("--train_steps", type=int, help="Number of training steps", default=5)
    parser.add_argument("--epochs", type=int, help="Number of training steps", default=1)
    parser.add_argument("--batch_size_train", type=int, help="Batch size for training", default=8)
    parser.add_argument("--batch_size_eval", type=int, help="Batch size for evaluation", default=4)
    parser.add_argument("--grad_acc_steps", type=int, help="Gradient accumulation steps", default=1)
    parser.add_argument("--lr", type=float, help="Learning ratre", default=2e-4)
    parser.add_argument("--max_length", type=int, help="Maximum sequence length", default=4096)
    parser.add_argument("--n_icl_samples", type=int, help="Number of ICL examples", default=3)
    parser.add_argument("--dtype", type=str, help="Data type for training", default=None)
    parser.add_argument("--rationale", help="Whether to include rationale in the prompt", action='store_true')
    parser.add_argument("--entitytypes", help="Filename of the entity2type json", default='entity2type.json')
    parser.add_argument("--prompt_filename", help="Filename of the prompt to use (code_prompt/code_expl_prompt)", default='code_prompt')
    parser.add_argument("--target_modules", type=str, help="List of LoRA modules to use (as dash-separated string).", default='q-k-v-o-gate-up-down')
    parser.add_argument("--val", type=int, help="Evaluate on validation split", default=0)
    parser.add_argument("--save_model", type=int, help="Don't save the fine-tuned model", default=1)
    parser.add_argument("--save_results", type=int, help="Save the training results", default=0)
    parser.add_argument("--val_samples", type=int, help="Number of validation samples", default=0)
    parser.add_argument("--load_in_4bit", type=int, help="Use 4-bit quantization", default=0)
    parser.add_argument("--load_in_8bit", type=int, help="Use 8-bit quantization", default=0)
    parser.add_argument("--chat", type=int, help="Whether it's a chat model", default=0)
    parser.add_argument("--natlang", type=int, help="Use natural language prompts", default=0)
    parser.add_argument("--verbose_train", type=int, help="Verbose training", default=0)
    parser.add_argument("--seed", type=int, help="Seed to use for random processes", default=0)
    parser.add_argument("--run_id", type=str, help="Seed to use for random processes", default='')
    
    args = parser.parse_args()

    main(args)