from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
import pandas as pd
import json
from datasets import Dataset
from utils import Runner
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse
import logging
import numpy as np
from datetime import datetime
from utils import *

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

def main(args):
    dataset_path = f'./data/parsing/{args.dataset}'
    schema_path = os.path.join(dataset_path, args.prompt_filename)
    
    print('##################################################################')
    print(f'Training model: {args.model_name}')
    print(f"Training data: {args.dataset}")
    print(f'Chat model: {args.chat}')
    print(f'Rationale: {args.rationale}')
    print(f"Language: {'natlang' if args.natlang else 'code'}")
    print(f"Number of ICL samples: {args.n_icl_samples}")
    print(f"4-bit quantization: {args.load_in_4bit}")
    print(f"8-bit quantization: {args.load_in_8bit}")
    print(f"Target modules: {args.target_modules}")
    print(f"Learning rate: {args.lr}")
    print('##################################################################')

    bit_string = '4-bit' if args.load_in_4bit else 'fp16'
    bit_string = '8-bit' if args.load_in_8bit else bit_string
    model_dir = os.path.join('./models', bit_string)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Configure quantization if needed
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    # Load model and tokenizer    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    print(model)
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    # Ensure tokenizer is properly configured
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set appropriate chat template if needed
    if tokenizer.chat_template is None and hasattr(model.config, 'model_type'):
        tokenizer.chat_template = chat_template_dict.get(model.config.model_type)
        print(f"Chat template not found, using the one for model type \"{model.config.model_type}\"")
    
    # Configure LoRA or full fine-tuning
    if args.target_modules != 'full_ft':
        target_modules = [el+'_proj' for el in args.target_modules.split('-')]
        
        # Prepare the model for training, particularly important for quantized models
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
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
    else:
        target_modules = args.target_modules
        lora_config = None

    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Save model information
    print(tokenizer.chat_template, file=open(f"./model_info/{args.model_name.split('/')[-1]}_chat_template.txt", 'w'))
    print(model, file=open(f"./model_info/{args.model_name.split('/')[-1]}_arch.txt", 'w'))

    entity2type_json = os.path.join(dataset_path, args.entitytypes)
    with open(entity2type_json, 'r', encoding='utf8') as f:
        entity2type_dict = json.load(f)

    runner = Runner(model=model,
                    type_dict=entity2type_dict,
                    natlang=args.natlang,
                    tokenizer=tokenizer,
                    chat_model=args.chat,
                    schema_path=schema_path,
                    rationale=args.rationale,
                    verbose_train=args.verbose_train,
                    max_seq_len=args.max_seq_length,
                    model_name=args.model_name,
                    )
    
    print('Model system message:', runner.check_system_msg())

    train_json = os.path.join(dataset_path, 'train_triples.json')
    df_train = pd.read_json(train_json)
    text_list_train = runner.make_samples(tokenizer, df_train, n_icl_samples=args.n_icl_samples)
    df_train = pd.DataFrame(text_list_train)
    dataset_train = Dataset.from_pandas(df_train.rename(columns={0: "text"}), split="train")

    if args.noval:
        n_samples_val = 0
        dataset_val = None
        eval_strategy = 'no'
        load_best_model_at_end=False
    else:
        val_json = os.path.join(dataset_path, 'val_triples.json')
        df_val = pd.read_json(val_json)
        text_list_val = runner.make_samples(tokenizer, df_val, n_icl_samples=args.n_icl_samples)
        n_samples_val = len(df_val) if not args.val_samples else args.val_samples
        df_val = pd.DataFrame(text_list_val).sample(n=n_samples_val)
        dataset_val = Dataset.from_pandas(df_val.rename(columns={0: "text"}), split="val")
        eval_strategy = 'steps'
        load_best_model_at_end=True
    
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        logits_argmax = np.argmax(logits, axis=-1)
        
        predictions = tokenizer.batch_decode(logits_argmax, skip_special_tokens=True)
        
        # Decode labels, ignoring -100 values
        label_tokens = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels_decoded = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
        
        # Extract triples from predictions and labels
        predicted_triples = [runner.extract_triples(pred) for pred in predictions]
        true_triples = [runner.extract_triples(label) for label in labels_decoded]
        
        precision, recall, f1_score = runner.calculate_strict_micro_f1(true_triples, predicted_triples)
        
        return {"precision": precision, "recall": recall, "f1": f1_score}

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        compute_metrics=compute_metrics,
        peft_config=lora_config,
        args=SFTConfig(
            dataset_text_field = "text",
            max_seq_length = args.max_seq_length,
            dataset_num_proc = 2,
            packing = False,
            per_device_train_batch_size=args.batch_size_train,
            per_device_eval_batch_size=args.batch_size_eval,
            gradient_accumulation_steps=args.grad_acc_steps,
            gradient_checkpointing_kwargs={'use_reentrant':False},
            gradient_checkpointing=True,
            warmup_steps=5,
            max_steps=args.train_steps,
            learning_rate=args.lr,
            fp16=False,
            bf16=True,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            eval_strategy=eval_strategy,
            eval_steps=args.train_steps,
            save_steps=args.train_steps,
            metric_for_best_model="f1",
            load_best_model_at_end=load_best_model_at_end,
            use_liger=True,
        ),
    )

    trainer_stats = trainer.train()

    best_metric = trainer.state.best_metric
    print(f"Best F1 score: {best_metric}")

    if not args.noval:
        eval_results = trainer.evaluate()
        print(f"Final evaluation results: {eval_results}")
    else:
        eval_results = {
            'eval_loss': -1,
            'eval_precision': -1,
            'eval_recall': -1,
            'eval_f1': -1,
        }

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    info = [{"Model": args.model_name,
            "target_modules": args.target_modules,
            "Loss_train": trainer_stats.metrics['train_loss'],
            "Loss_val": eval_results['eval_loss'],
            "Precision_val": eval_results['eval_precision'],
            "Recall_val": eval_results['eval_recall'],
            "F1_val": eval_results['eval_f1'],
            "n_icl_samples": args.n_icl_samples,
            "n_samples_val": n_samples_val,
            "dataset": args.dataset,
            "date": dt_string,
            "schema_path": schema_path,
            "split": 'val',
            "noval": args.noval,
            "lerning_rate": args.lr,
            }]
    
    print(info)

    results_dir_path = os.path.join('./results',
                                    f"{args.target_modules}",
                                    )

    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)
    
    model_name_simple = args.model_name.split('/')[-1]
    model_name_ft = f"{model_name_simple}_ft_{args.dataset}_{'natlang' if args.natlang else 'code'}_{'rationale' if args.rationale else 'base'}_steps={args.train_steps}_icl={args.n_icl_samples}_mod={args.target_modules.replace('_proj', '')}"

    model_save_path = os.path.join(model_dir, model_name_ft)
    
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    json_path = os.path.join(model_save_path, f"{model_name_ft}_val.json")
    print(f'Training results saved to: {json_path}')
    save_json(info, json_path)

    if args.no_model_save:
        print('Model was not saved because of --no_model_save flag')
    else:
        # Standard model saving instead of Unsloth's save_pretrained_merged
        if args.target_modules != 'full_ft':
            # For LoRA models, save the adapter
            model.save_pretrained(model_save_path)
        else:
            # For full fine-tuned models
            model.save_pretrained(model_save_path, safe_serialization=True)
        tokenizer.save_pretrained(model_save_path)
        print(f'Fine-tuned model saved to: {model_save_path}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument("-m", "--model_name", type=str, help="Name of the model to train", default='mistralai/Mistral-7B-Instruct-v0.3')
    parser.add_argument("-d", "--dataset", type=str, help="Name of the dataset to use", default='ade')
    parser.add_argument("--train_steps", type=int, help="Number of training steps", default=200)
    parser.add_argument("--batch_size_train", type=int, help="Batch size for training", default=8)
    parser.add_argument("--batch_size_eval", type=int, help="Batch size for evaluation", default=4)
    parser.add_argument("--grad_acc_steps", type=int, help="Gradient accumulation steps", default=1)
    parser.add_argument("--lr", type=float, help="Learning ratre", default=2e-4)
    parser.add_argument("--no_model_save", action="store_true", help="Don't save the fine-tuned model")
    parser.add_argument("--save_results", action="store_true", help="Save the training results")
    parser.add_argument("--val_samples", type=int, help="Number of validation samples", default=0)
    parser.add_argument("--noval", action="store_false", help="Do NOT evaluate on validation split")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--n_icl_samples", type=int, default=3, help="Number of ICL examples")
    parser.add_argument("--dtype", type=str, default=None, help="Data type for training")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--chat", action="store_true", help="Whether it's a chat model")
    parser.add_argument("--natlang", action="store_true", help="Use natural language prompts")
    parser.add_argument("-r", "--rationale", help="Whether to include rationale in the prompt", action='store_true')
    parser.add_argument("-ent", "--entitytypes", help="Filename of the entity2type json", default='entity2type.json')
    parser.add_argument("-pf", "--prompt_filename", help="Filename of the prompt to use (code_prompt/code_expl_prompt)", default='code_prompt')
    parser.add_argument("--verbose_train", action="store_true", help="Verbose training")
    parser.add_argument("--target_modules", type=str, help="List of LoRA modules to use (as dash-separated string).", default='q-k-v-o-gate-up-down')
    
    args = parser.parse_args()

    # args.model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    # args.model_name = 'meta-llama/Llama-3.2-1B-Instruct'

    main(args)