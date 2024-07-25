import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from typing import List
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
from utils import *

def main():

    dataset_name = 'ade'
    # dataset_name = 'conll04'
    # dataset_name = 'scierc'

    # model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'

    model_name = f'./models/llama-3-8b-bnb-4bit_ft_{dataset_name}'

    model_name_simple = model_name.split('/')[-1]   
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype='auto',
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset_path = f'./data/codekgc-data/{dataset_name}'

    train_json = os.path.join(dataset_path, 'train_triples.json')
    test_json = os.path.join(dataset_path, 'test_triples.json')

    # schema_path = os.path.join(dataset_path, 'code_expl_prompt')
    schema_path = os.path.join(dataset_path, 'code_prompt')
    
    schema_prompt = open(schema_path, 'r', encoding='utf8').read()

    df_test = pd.read_json(test_json)
    n_samples_test = len(df_test)
    df_test = df_test.sample(n=n_samples_test)
    n_icl_samples = 15

    entity2type_json = os.path.join(dataset_path, 'entity2type.json')
    with open(entity2type_json, 'r', encoding='utf8') as f:
        entity2type_dict = json.load(f)

    prompter = Prompter(entity2type_dict,
                        natlang = False,
                        )
    df_train = pd.read_json(train_json).sample(n=n_icl_samples)
    icl_prompt = prompter.make_icl_prompt(df_train)
    precision, recall, f1_score = evaluate(model, tokenizer, df_test, schema_prompt, icl_prompt, prompter, chat_model=False)

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    info = [{"Model": model_name,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1_score,
            "n_icl_samples": n_icl_samples,
            "n_samples_test": n_samples_test,
            "dataset_name": dataset_name,
            "date": dt_string,
            "schema_path": schema_path,
            }]
    
    print(info)
    json_path = f'./results/{model_name_simple}_{dataset_name}.json'

    save_json(info, json_path)

if __name__ == "__main__":
    main()