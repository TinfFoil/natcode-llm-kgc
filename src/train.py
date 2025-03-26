from unsloth import FastLanguageModel
import torch
import os
import pandas as pd
import json
from datasets import Dataset
from utils import Runner
from trl import SFTTrainer
from transformers import TrainingArguments
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

    max_seq_length = args.max_seq_length
    dtype = args.dtype
    load_in_4bit = args.load_in_4bit
    model_name = args.model_name
    chat_model = args.chat
    natlang = args.natlang
    dataset_name = args.dataset
    dataset_path = f'./data/codekgc-data/{dataset_name}'
    schema_path = os.path.join(dataset_path, args.prompt_filename)
    print('##################################################################')
    print(f'Training model: {model_name}')
    print(f"Training data: {dataset_name}")
    print(f'Chat model: {chat_model}')
    print(f'Rationale: {args.rationale}')
    print(f"Language: {'natlang' if natlang else 'code'}")
    print(f"Number of ICL samples: {args.n_icl_samples}")
    print(f"Quantized: {args.load_in_4bit}")
    print('##################################################################')

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit
    )

    if tokenizer.chat_template is None:
        tokenizer.chat_template = chat_template_dict[model.config.model_type]
        print(f"Chat template not found, using the one for model type \"{model.config.model_type}\"")

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # 4x longer contexts auto supported!
        random_state = 3407,
        use_rslora = True,  # TODO: redo with rslora
        loftq_config = None, # And LoftQ
    )

    print(tokenizer.chat_template, file=open(f"./model_info/{args.model_name.split('/')[-1]}_chat_template.txt", 'w'))

    print(model, file=open(f"./model_info/{args.model_name.split('/')[-1]}_arch.txt", 'w'))

    entity2type_json = os.path.join(dataset_path, args.entitytypes)
    with open(entity2type_json, 'r', encoding='utf8') as f:
        entity2type_dict = json.load(f)

    runner = Runner(model=model,
                    type_dict=entity2type_dict,
                    natlang=natlang,
                    tokenizer=tokenizer,
                    chat_model=chat_model,
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
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset_train,
        eval_dataset = dataset_val,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        compute_metrics=compute_metrics,
        args = TrainingArguments(
            per_device_train_batch_size = args.batch_size_train,
            per_device_eval_batch_size = args.batch_size_eval,
            gradient_accumulation_steps = args.grad_acc_steps,
            warmup_steps = 5,
            max_steps = args.train_steps,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            # logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",
            eval_strategy = eval_strategy,
            eval_steps = args.train_steps,
            save_steps = args.train_steps,
            metric_for_best_model="f1",
            load_best_model_at_end=load_best_model_at_end,
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
            }]
    
    print(info)

    results_dir_path = './results'

    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)
    
    model_name_simple = model_name.split('/')[-1]
    model_name_ft = f"{model_name_simple}_ft_{args.dataset}_{'natlang' if natlang else 'code'}_{'rationale' if args.rationale else 'base'}_steps={args.train_steps}_icl={args.n_icl_samples}"
    json_path = os.path.join(results_dir_path, f"{model_name_ft}_val.json")

    if args.save_results:
        print(f'Training results saved to: {json_path}')
        save_json(info, json_path)
    else:
        print('Training results were not saved because of NO --save_results flag')

    model_dir = './models'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_save_path = os.path.join(model_dir, model_name_ft)
    if args.no_model_save:
        print('Model was not saved because of --no_model_save flag')
    else:
        model.save_pretrained_merged(model_save_path, tokenizer, save_method = "merged_16bit",)
        print(f'Fine-tuned model saved to: {model_save_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument("-m", "--model_name", type=str, help="Name of the model to train", default='unsloth/Meta-Llama-3.1-8B')
    parser.add_argument("-d", "--dataset", type=str, help="Name of the dataset to use", default='ade')
    parser.add_argument("--train_steps", type=int, help="Number of training samples", default=200)
    parser.add_argument("--batch_size_train", type=int, help="Number of training samples", default=8)
    parser.add_argument("--batch_size_eval", type=int, help="Number of training samples", default=4)
    parser.add_argument("--grad_acc_steps", type=int, help="Number of training samples", default=1)
    parser.add_argument("--no_model_save", action="store_true", help="Don't save the fine-tuned model")
    parser.add_argument("--save_results", action="store_true", help="Don't save the training results")
    parser.add_argument("--val_samples", type=int, help="Number of validation samples", default=0)
    parser.add_argument("--noval", action="store_false", help="Do NOT evaluate on validation split")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--n_icl_samples", type=int, default=3, help="Number of ICL examples")
    parser.add_argument("--dtype", type=str, default=None, help="Data type for training")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--chat", action="store_true", help="Whether it's a chat model")
    parser.add_argument("--natlang", action="store_true", help="Use natural language prompts")
    parser.add_argument("-r", "--rationale", help="Whether to include rationale in the prompt", action='store_true')
    parser.add_argument("-ent", "--entitytypes", help="Filename of the entity2type json", default='entity2type.json')
    parser.add_argument("-pf", "--prompt_filename", help="Filename of the prompt to use (code_prompt/code_expl_prompt)", default='code_prompt')
    parser.add_argument("--verbose_train", action="store_true", help="Verbose training")
    args = parser.parse_args()

    main(args)