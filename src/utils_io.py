import json
import yaml
from datetime import datetime
import argparse
import os
from typing import Dict, List
from peft import LoraConfig
import pandas as pd

def setup_config(namespace: argparse.Namespace, default_cfg: dict = {}):
    args = vars(namespace)
    config = default_cfg
    for k, v in args.items():
        config[k] = v
    if not config['run_id']:
        config['run_id'] = get_time()
    
    config['dataset_path'] = f"./data/parsing/{config['dataset']}/rdf"
    config['schema_path'] = os.path.join(config['dataset_path'], config['prompt_filename'])

    lora_modules = [el+'_proj' for el in config['lora_modules'].split('-') if el]
    config['lora_config'] = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=lora_modules,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=False,
    ) if lora_modules else None
    
    config['lr'] = 2e-4 if lora_modules else 1e-5
    
    config['model_name_string'] = config['model_name'].replace('/', '-')
    config['model_dir'] = os.path.join('./models',
                                             config['model_name_string'],
                                             config['run_id'],
                                             )
    if not os.path.exists(config['model_dir']): os.makedirs(config['model_dir'], exist_ok=True)
    config['results_dir'] = './results'
    if not os.path.exists(config['results_dir']): os.makedirs(config['results_dir'], exist_ok=True)
    
    config['log_path'] = os.path.join(config['model_dir'], f"preds.log")
    model_chat_dict = yaml.safe_load(open('./model_info/model_chat_dict.yaml', 'r'))
    config['chat'] = model_chat_dict[config['model_name']]

    config_path = os.path.join(config['model_dir'], 'config.json')
    print(f'Config saved to: {config_path}')
    save_json(config, config_path)

    return config

def save_prompt(df: pd.DataFrame, config):
    txt_path = os.path.join(config['model_dir'], f"{config['model_name_string']}.txt")
    text = df.iloc[0]['text']
    open(txt_path, 'w', encoding='utf8').write(text)

def dict_to_records(input: Dict[str, List]):
    # check that all entries are lists
    assert all([isinstance(v, list) for k, v in input.items()])
    # and that all of them have the same length
    lengths = [len(v) for k, v in input.items()]
    assert len(set(lengths)) == 1
    list_of_dicts = []
    for i in range(lengths[0]):
        list_of_dicts.append({k: v[i] for k, v in input.items()})
    return list_of_dicts

def save_json(info, json_path):
    try:
        with open(json_path, 'r', encoding='utf8') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    
    data.extend(info)

    with open(json_path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii = False)

def get_time():
    return str(datetime.now()).split('.')[0].replace(' ', '').replace('-', '').replace(':', '')[2:]

def print_run_info(args):
    print('##################################################################')
    print(f'Training model: {args.model_name}')
    print(f"Training data: {args.dataset}")
    print(f'Chat model: {args.chat}')
    print(f'Rationale: {args.rationale}')
    print(f"Language: {'natlang' if args.natlang else 'code'}")
    print(f"Number of ICL samples: {args.n_icl_samples}")
    print(f"4-bit quantization: {args.load_in_4bit}")
    print(f"8-bit quantization: {args.load_in_8bit}")
    print(f"Target modules: {args.lora_modules}")
    print(f"Learning rate: {args.lr}")
    print('##################################################################')
