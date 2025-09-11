import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from utils_io import setup_config, save_json, save_prompt
from utils_train import get_quant_config, prep_model, get_trainer
from runner import Runner
from calculate_metrics import RelationExtractionEvaluator
import argparse
import os
import yaml
import pandas as pd 

def main(args):
    config = setup_config(args)

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        quantization_config=get_quant_config(config),
        dtype=getattr(torch, config['dtype_str']),
        device_map='auto',
    )
    model.gradient_checkpointing_enable()
    evaluator = RelationExtractionEvaluator(mode = 'EE')
    runner = Runner(model=model,
                    tokenizer=tokenizer,
                    config=config,
                    evaluator=evaluator,
                    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token = tokenizer.eos_token
        print('pad_token reassigned to eos_token')
    
    if tokenizer.chat_template is None and hasattr(model.config, 'model_type'):
        chat_template_dict = yaml.safe_load(open('./model_info/chat_templates.yaml'))
        tokenizer.chat_template = chat_template_dict.get(model.config.model_type)
        tokenizer.chat_template = chat_template_dict.get(model.config.model_type)
        print(f"Chat template not found, using the one for model type \"{model.config.model_type}\"")

    df_train = pd.read_json(os.path.join(config['dataset_path'], 'train.json'))
    df_train_prompts = runner.make_dataset(df_train, tokenizer)
    dataset_train = Dataset.from_pandas(df_train_prompts, split="train")
    
    if not config['train_steps']:
        train_size = len(dataset_train)
        batch_size = int(config['batch_size_train'])
        config['train_steps'] = train_size // batch_size
        print(f"Argument `train_steps` not specified, training on the whole dataset ({train_size} samples, {config['train_steps']} steps @ batch size == {batch_size})")
    
    df_val = pd.read_json(os.path.join(config['dataset_path'], 'val.json'))
    df_test = pd.read_json(os.path.join(config['dataset_path'], 'test.json'))
    if config['eval_steps']:
        df_val = df_val[:config['eval_steps'] * config['batch_size_eval']]
        df_test = df_test[:config['eval_steps'] * config['batch_size_eval']]

    if config['save_prompt']:
        txt_path = os.path.join(config['results_dir'], 'train_prompt.txt')
        text = df_train_prompts.iloc[0]['text']
        save_prompt(text, txt_path)

    model = prep_model(config, model)
    trainer = get_trainer(config, model, tokenizer, dataset_train)
    val_results = []
    if config['do_train']:
        for epoch in range(config['epochs']):
            trainer_stats = trainer.train()

            best_metric = trainer.state.best_metric
            print(f"Best F1 score: {best_metric}")

            if config['evaluate']:
                val_results.append(runner.evaluate(df_val, df_train, split = 'val'))
                print(f"Val @ epoch {epoch + 1}: {val_results}")
            else:
                val_results = {
                    'eval_loss': -1,
                    'eval_precision': -1,
                    'eval_recall': -1,
                    'eval_f1': -1,
                }
        save_json(val_results, os.path.join(config['results_dir'], 'val_results.json'))
 
    test_results = runner.evaluate(df_test, df_train, split = 'test')
    print(f"Test results: {test_results}")

    save_json(test_results, os.path.join(config['results_dir'], 'test_results.json'))

    if config['save_model'] and config['do_train']:
        if config['lora_modules']:
            model.save_pretrained(config['model_dir'])
        else:
            model.save_pretrained(config['model_dir'], safe_serialization=True)
        tokenizer.save_pretrained(config['model_dir'])
        print(f"Fine-tuned model saved to: {config['model_dir']}")
    else:
        print(f"Model was not saved because of `save_model`=={config['save_model']}, `do_train`=={config['do_train']}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument("--model_name", type=str, help="Name of the model to train", default='mistralai/Mistral-7B-Instruct-v0.3')
    parser.add_argument("--dataset", type=str, help="Name of the dataset to use", default='ade')
    parser.add_argument("--train_steps", type=int, help="Number of training steps", default=0)
    parser.add_argument("--eval_steps", type=int, help="Number of validation samples", default=0)
    parser.add_argument("--epochs", type=int, help="Number of training steps", default=1)
    parser.add_argument("--batch_size_train", type=int, help="Batch size for training", default=4)
    parser.add_argument("--batch_size_eval", type=int, help="Batch size for evaluation", default=4)
    parser.add_argument("--grad_acc_steps", type=int, help="Gradient accumulation steps", default=1)
    parser.add_argument("--lr", type=float, help="Learning ratre", default=2e-4)
    parser.add_argument("--max_length", type=int, help="Maximum sequence length", default=4096)
    parser.add_argument("--max_new_tokens", type=int, help="Maximum generated tokens during inference", default=5000)
    parser.add_argument("--n_icl_samples", type=int, help="Number of ICL examples", default=3)
    parser.add_argument("--dtype_str", type=str, help="Data type for training (most common are `float16`, `bfloat16`, `float32`)", default='float16')
    parser.add_argument("--rationale", type=int, help="Whether to include rationale in the prompt", default=0)
    parser.add_argument("--entitytypes", help="Filename of the entity2type json", default='entity2type.json')
    parser.add_argument("--prompt_filename", help="Filename of the prompt to use (code_prompt/code_expl_prompt)", default='code_prompt')
    parser.add_argument("--lora_modules", type=str, help="List of LoRA modules to use (as dash-separated string). Empty for full fine-tuning", default='q-k-v-o-gate-up-down')
    parser.add_argument("--evaluate", type=int, help="Evaluate on validation split", default=1)
    parser.add_argument("--save_model", type=int, help="Don't save the fine-tuned model", default=1)
    parser.add_argument("--save_results", type=int, help="Save the training results", default=0)
    parser.add_argument("--results_dir", type=str, help="Target dir in which to save the results", default='')
    parser.add_argument("--load_in_4bit", type=int, help="Use 4-bit quantization", default=0)
    parser.add_argument("--load_in_8bit", type=int, help="Use 8-bit quantization", default=0)
    # parser.add_argument("--chat", type=int, help="Whether it's a chat model", default=0)
    parser.add_argument("--natlang", type=int, help="Use natural language prompts", default=1)
    parser.add_argument("--save_prompt", type=int, help="Verbose training", default=0)
    parser.add_argument("--verbose_preds", type=int, help="Whether to print predictions during testing", default=0)
    parser.add_argument("--verbose_metrics", type=int, help="Whether to print partial metrics during testing", default=0)
    parser.add_argument("--seed", type=int, help="Seed to use for random processes", default=0)
    parser.add_argument("--do_train", type=int, help="Whether to train the model or use the original weights", default=1)
    parser.add_argument("--run_id", type=str, help="ID of the run", default='')
    
    args = parser.parse_args()

    main(args)