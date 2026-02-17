from unsloth import FastLanguageModel
from datasets import Dataset
from utils_io import setup_config, save_json, save_prompt
from utils_train import get_trainer, print_params
from runner import Runner
from calculate_metrics import RelationExtractionEvaluator
import argparse
import os
import yaml
import pandas as pd

def main(args):
    config = setup_config(args)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config['model_name'],
        max_seq_length = config['max_length'],   # Context length - can be longer, but uses more memory
        load_in_4bit = config['load_in_4bit'],     # 4bit uses much less memory
        load_in_8bit = config['load_in_8bit'],    # A bit more accurate, uses 2x memory
        full_finetuning = True if ''.join(config['lora_config']['target_modules']) == 'ft' else False, # We have full finetuning now!
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules = config['lora_config']['target_modules'],
        lora_alpha = 32,  # Best to choose alpha = rank or rank*2
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,   # We support rank stabilized LoRA
        loftq_config = None,  # And LoftQ
    )

    print_params(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.chat_template is None and hasattr(model.config, 'model_type'):
        if 'Qwen3' in config['model_name']:
            tokenizer.chat_template = open('./model_info/qwen3.jinja').read()
        else:
            chat_template_dict = yaml.safe_load(open('./model_info/chat_templates.yaml'))
            tokenizer.chat_template = chat_template_dict.get(model.config.model_type)
        print(f"Chat template not found, using the one for model type \"{model.config.model_type}\"")

    evaluator = RelationExtractionEvaluator(mode = 'EE')

    runner = Runner(model=model,
                    tokenizer=tokenizer,
                    config=config,
                    evaluator=evaluator,
                    )
    
    df_train = pd.read_json(os.path.join(config['dataset_path'], 'train.json'))
    df_train_prompts = runner.make_train_set(df_train, tokenizer)
    dataset_train = Dataset.from_pandas(df_train_prompts, split="train")
    
    if not config['train_steps']:
        train_size = len(dataset_train)
        batch_size = int(config['batch_size_train'])
        config['train_steps'] = train_size // batch_size
        print(f"Argument `train_steps` not specified, training on the whole dataset ({train_size} samples, {config['train_steps']} steps @ batch size == {batch_size})")
    
    df_val = pd.read_json(os.path.join(config['dataset_path'], 'val.json'))
    df_test = pd.read_json(os.path.join(config['dataset_path'], 'test.json'))
    if config['eval_samples']:
        df_val = df_val[:config['eval_samples']]
        df_test = df_test[:config['eval_samples']]

    if config['save_prompt']:
        txt_path = os.path.join(config['results_dir'], 'train_prompt.txt')
        text = df_train_prompts.iloc[0]['text']
        save_prompt(text, txt_path)

    # model = prep_model(config, model)
    
    trainer = get_trainer(config, model, tokenizer, dataset_train)
    
    # model.gradient_checkpointing_enable()

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
    parser.add_argument("--eval_samples", type=int, help="Number of evaluation samples", default=0)
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=1)
    parser.add_argument("--batch_size_train", type=int, help="Batch size for training", default=4)
    parser.add_argument("--batch_size_eval", type=int, help="Batch size for evaluation", default=4)
    parser.add_argument("--grad_acc_steps", type=int, help="Gradient accumulation steps", default=1)
    parser.add_argument("--dtype_str", type=str, help="torch dtype for the model", default='bfloat16')
    parser.add_argument("--lr", type=float, help="Learning rate", default=2e-4)
    parser.add_argument("--max_length", type=int, help="Maximum sequence length", default=4096)
    parser.add_argument("--max_new_tokens", type=int, help="Maximum number of tokens to generate during inference", default=5000)
    parser.add_argument("--n_icl_samples", type=int, help="Number of ICL examples", default=3)
    parser.add_argument("--dtype", type=str, help="Data type for training", default=None)
    parser.add_argument("--rationale", type=int, help="Whether to include rationale in the prompt", default=0)
    parser.add_argument("--entitytypes", help="Filename of the entity2type json", default='entity2type.json')
    parser.add_argument("--prompt_filename", help="Filename of the prompt yaml", default='default.yaml')
    parser.add_argument("--desc_schema", type=int, help="Include descriptions of entities and relations in the instruction prompt (include a dict rather than a list of ents/rels)", default=1)
    parser.add_argument("--lora_modules", type=str, help="List of LoRA modules to use (as dash-separated string). Empty for full fine-tuning", default='q-k-v-o-gate-up-down')
    parser.add_argument("--evaluate", type=int, help="Evaluate on validation split", default=0)
    parser.add_argument("--save_model", type=int, help="Don't save the fine-tuned model", default=1)
    parser.add_argument("--save_results", type=int, help="Save the training results", default=0)
    parser.add_argument("--results_dir", type=str, help="Target dir in which to save the results", default='')
    parser.add_argument("--load_in_4bit", type=int, help="Use 4-bit quantization", default=0)
    parser.add_argument("--load_in_8bit", type=int, help="Use 8-bit quantization", default=0)
    parser.add_argument("--save_prompt", type=int, help="Save the prompts for manual inspection", default=1)
    parser.add_argument("--verbose_preds", type=int, help="Whether to print predictions during testing", default=1)
    parser.add_argument("--verbose_metrics", type=int, help="Whether to print partial metrics during testing", default=1)
    parser.add_argument("--seed", type=int, help="Seed to use for random processes", default=0)
    parser.add_argument("--enable_thinking", type=int, help="Whether to use thinking mode", default=0)
    parser.add_argument("--do_train", type=int, help="Whether to train the model or use the original weights", default=1)
    parser.add_argument("--run_id", type=str, help="ID of the run", default='')
    
    args = parser.parse_args()

    main(args)