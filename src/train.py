import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from utils_io import setup_config, save_json, save_prompt
from utils_train import get_quant_config, prep_model, get_trainer, make_dataset
from runner import Runner
from calculate_metrics import RelationExtractionEvaluator
import argparse
import os
import yaml
import pandas as pd

def main(args):
    config = setup_config(args)

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.chat_template is None and hasattr(model.config, 'model_type'):
        chat_template_dict = yaml.safe_load(open('./model_info/chat_templates.yaml'))
        tokenizer.chat_template = chat_template_dict.get(model.config.model_type)
        print(f"Chat template not found, using the one for model type \"{model.config.model_type}\"")

    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        quantization_config=get_quant_config(config),
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    model.gradient_checkpointing_enable()
    evaluator = RelationExtractionEvaluator(mode = 'SE')
    runner = Runner(model=model,
                    tokenizer=tokenizer,
                    config=config,
                    evaluator=evaluator,
                    )
    
    df_train = pd.read_json(os.path.join(config['dataset_path'], 'train.json'))
    df_train_prompts = make_dataset(df_train, runner, tokenizer)
    dataset_train = Dataset.from_pandas(df_train_prompts, split="train")

    if config['save_prompt']:
        save_prompt(df_train, config)
    
    if not config['train_steps']:
        config['train_steps'] = len(dataset_train) // int(config['batch_size_train'])
    
    df_val = pd.read_json(os.path.join(config['dataset_path'], 'val.json'))
    if config['val_samples']:
        df_val = df_val[:config['val_samples']]

    model = prep_model(config, model)
    trainer = get_trainer(config, model, tokenizer, dataset_train)
    
    if config['do_train']:
        for epoch in range(config['epochs']):
            trainer_stats = trainer.train()

            best_metric = trainer.state.best_metric
            print(f"Best F1 score: {best_metric}")

            if config['evaluate']:
                eval_results = runner.evaluate(df_val, df_train)
                print(f"Final evaluation results: {eval_results}")
            else:
                eval_results = {
                    'eval_loss': -1,
                    'eval_precision': -1,
                    'eval_recall': -1,
                    'eval_f1': -1,
                }
    
    if config['save_model']:
        if config['lora_modules']:
            model.save_pretrained(config['model_dir'])
        else:
            model.save_pretrained(config['model_dir'], safe_serialization=True)
        tokenizer.save_pretrained(config['model_dir'])
        print(f"Fine-tuned model saved to: {config['model_dir']}")
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
    parser.add_argument("--rationale", type=int, help="Whether to include rationale in the prompt", default=0)
    parser.add_argument("--entitytypes", help="Filename of the entity2type json", default='entity2type.json')
    parser.add_argument("--prompt_filename", help="Filename of the prompt to use (code_prompt/code_expl_prompt)", default='code_prompt')
    parser.add_argument("--lora_modules", type=str, help="List of LoRA modules to use (as dash-separated string). Empty for full fine-tuning", default='q-k-v-o-gate-up-down')
    parser.add_argument("--evaluate", type=int, help="Evaluate on validation split", default=1)
    parser.add_argument("--save_model", type=int, help="Don't save the fine-tuned model", default=1)
    parser.add_argument("--save_results", type=int, help="Save the training results", default=0)
    parser.add_argument("--val_samples", type=int, help="Number of validation samples", default=0)
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