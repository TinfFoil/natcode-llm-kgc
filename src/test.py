import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from datetime import datetime
from utils import *
import argparse

def main(args):
    print(f'Testing model: {args.model_name}')
    print(f'Testing dataset: {args.dataset}')
    print(f'Split: {args.test_split}')
    print(f'Chat model: {args.chat}')
    print(f'Rationale: {args.rationale}')
    print(f"Language: {'natlang' if args.natlang else 'code'}")
    print(f"Number of ICL samples: {args.n_icl_samples}")

    model_name_simple = args.model_name.split('/')[-1]

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype='auto',
        device_map='auto',
        quantization_config=nf4_config,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              padding_side = 'left',
                                              )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_path = f'./data/parsing/{args.dataset}'
    
    train_split = f'{args.train_split}_triples.json'
    train_json = os.path.join(dataset_path, train_split)
    test_split = f'{args.test_split}_triples.json'
    test_json = os.path.join(dataset_path, test_split)
    model_type = 'fine-tuned' if args.fine_tuned else 'base'
    
    results_dir_path = os.path.join(args.results_dir,
                                    test_split.split('_')[0],
                                    model_type,
                                    )

    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)

    results_json_path = os.path.join(results_dir_path, f"{model_name_simple}_{args.dataset}_{'natlang' if args.natlang else 'code'}_{'rationale' if args.rationale else 'base'}.json")

    print(f'Will save results to: {results_json_path}')

    schema_path = os.path.join(dataset_path, args.prompt_filename)

    df_test = pd.read_json(test_json)
    n_samples_test = len(df_test)
    df_test = df_test.sample(n=n_samples_test)

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
                    verbose_test=args.verbose_test,
                    model_name=args.model_name,
                    verbose_output_path=args.verbose_output_path,
                    )
    
    df_train = pd.read_json(train_json).sample(n=args.n_icl_samples)
    icl_prompt = runner.make_icl_prompt(df_train)

    eval_dict = runner.evaluate(
                                df_test,
                                icl_prompt,
                                batch_size=8
                                )

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    info = [{"Model": args.model_name,
            "Precision": eval_dict['precision'],
            "Recall": eval_dict['recall'],
            "F1_Score": eval_dict['f1'],
            "rel_type_metrics": eval_dict['rel_type_metrics'],
            "n_icl_samples": args.n_icl_samples,
            "n_samples_test": n_samples_test,
            "dataset": args.dataset,
            "date": dt_string,
            "schema_path": schema_path,
            "split": args.test_split,
            "fine-tuned": args.fine_tuned,
            "rationale": args.rationale,
            "natlang": args.natlang,
            "chat_model": args.chat,
            }]
    
    print(info)

    save_json(info, results_json_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A sample argparse program")
    parser.add_argument("-m", "--model_name", help="Model to test")
    parser.add_argument("-d", "--dataset", help="Name of the testing dataset, options = ['ade', 'conll04', 'scierc']")
    parser.add_argument("--natlang", help="Type of language", action='store_true')
    parser.add_argument("-r", "--rationale", help="Whether to include rationale in the prompt", action='store_true')
    parser.add_argument("--chat", help="Type of model (default = completion model)", action='store_true')
    parser.add_argument("-ent", "--entitytypes", help="Filename of  the entity2type json", default='entity2type.json')
    parser.add_argument("-pf", "--prompt_filename", help="Filename of the prompt to use (code_prompt/code_expl_prompt)", default='code_prompt')
    parser.add_argument("--train_split", help="Filename of the train file to use", default='train')
    parser.add_argument("--test_split", help="Filename of the test file to use", default='test')
    parser.add_argument("--n_icl_samples", type=int, default=3, help="Number of ICL examples")
    parser.add_argument("--verbose_test", default=0, type=int, help="Verbose testing")
    parser.add_argument("--fine_tuned", action="store_true", help="Whether a fine-tuned LLM is beind tested")
    parser.add_argument("--results_dir", help="Dir in which to save results", default='./results')
    parser.add_argument("--verbose_output_path", help="Dir in which to model outputs", default='./results/monitor')
    args = parser.parse_args()
    

    # args.model_name = "unsloth/Meta-Llama-3.1-8B"
    # args.model_name = "mistralai/Mistral-7B-v0.3" 
    # args.model_name = "deepseek-ai/deepseek-coder-7b-base-v1.5"
    # args.model_name = "Qwen/CodeQwen1.5-7B"

    # args.model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
    # args.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    # args.model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    # args.model_name = "Qwen/CodeQwen1.5-7B-Chat"

    # args.model_name = "./models/Meta-Llama-3.1-8B-Instruct_ft_ade_natlang_base_steps=200_icl=3"
    # args.model_name = "./models/Mistral-7B-Instruct-v0.3_ft_ade_natlang_base_steps=200_icl=3"
    # args.model_name = "./models/deepseek-coder-7b-instruct-v1.5_ft_ade_natlang_base_steps=200_icl=3"
    # args.model_name = "./models/CodeQwen1.5-7B-Chat_ft_ade_natlang_base_steps=200_icl=3"

    # args.model_name = "mistralai/Mistral-7B-v0.3"
    # args.dataset = "ade"
    # args.chat = 1
    # args.rationale = 0
    # args.natlang = 1
    # args.verbose_test = 1
    # args.fine_tuned = 1
    # args.bfloat16 = 1

    main(args)