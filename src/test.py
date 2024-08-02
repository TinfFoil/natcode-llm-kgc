import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from typing import List
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
from utils import *
import argparse

def main(args):
    print(f'Testing model: {args.model}')
    print(f'Testing dataset: {args.dataset}')
    print(f'Chat model: {args.chat}')
    print(f'Rationale: {args.rationale}')
    print(f"Language: {'natlang' if args.natlang else 'code'}")

    model_name_simple = args.model.split('/')[-1]   
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype='auto',
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset_path = f'./data/codekgc-data/{args.dataset}'

    train_json = os.path.join(dataset_path, args.train)
    test_json = os.path.join(dataset_path, args.test)

    schema_path = os.path.join(dataset_path, args.prompt_filename)

    df_test = pd.read_json(test_json)
    n_samples_test = len(df_test)
    df_test = df_test.sample(n=n_samples_test)
    n_icl_samples = 15

    entity2type_json = os.path.join(dataset_path, args.entitytypes)
    with open(entity2type_json, 'r', encoding='utf8') as f:
        entity2type_dict = json.load(f)

    runner = Runner(entity2type_dict,
                        natlang=args.natlang,
                        tokenizer=tokenizer,
                        chat_model=args.chat,
                        schema_path=schema_path,
                        rationale=args.rationale,
                        )
    df_train = pd.read_json(train_json).sample(n=n_icl_samples)
    icl_prompt = runner.make_icl_prompt(df_train)
    
    precision, recall, f1_score = runner.evaluate(model,
                                tokenizer,
                                df_test,
                                icl_prompt,
                                )

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    info = [{"Model": args.model,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1_score,
            "n_icl_samples": n_icl_samples,
            "n_samples_test": n_samples_test,
            "dataset": args.dataset,
            "date": dt_string,
            "schema_path": schema_path,
            }]
    
    print(info)
    json_path = f"./results/{model_name_simple}_{args.dataset}_{'natlang' if args.natlang else 'code'}.json"

    save_json(info, json_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A sample argparse program")
    parser.add_argument("-m", "--model", help="Model to test")
    parser.add_argument("-d", "--dataset", help="Name of the testing dataset, options = ['ade', 'conll04', 'scierc']")
    parser.add_argument("--natlang", help="Type of language", action='store_true')
    parser.add_argument("-r", "--rationale", help="Whether to include rationale in the prompt", action='store_true')
    parser.add_argument("--chat", help="Type of model (default = completion model)", action='store_true')
    parser.add_argument("-ent", "--entitytypes", help="Filename of  the entity2type json", default='entity2type.json')
    parser.add_argument("-pf", "--prompt_filename", help="Filename of the prompt to use (code_prompt/code_expl_prompt)", default='code_prompt')
    parser.add_argument("--train", help="Filename of the train file to use", default='train_triples.json')
    parser.add_argument("--test", help="Filename of the test file to use", default='test_triples.json')
    args = parser.parse_args()

    args.model = "./models/Mistral-7B-Instruct-v0.3_ft_scierc_natlang"
    args.dataset = "scierc"
    args.chat = True
    args.rationale = False
    args.natlang = True

    main(args)