import json
import yaml
from datetime import datetime
import argparse
import os
from typing import Dict, List
from pathlib import Path

def setup_config(namespace: argparse.Namespace, default_cfg: dict = {}):
    args = vars(namespace)
    model_dir = os.path.abspath(args.get('model_name', ''))
    if os.path.exists(model_dir):
        config_path = os.path.join(os.path.dirname(model_dir), 'config.json')
        with open(config_path, 'r', encoding='utf8') as f:
            config = json.load(f)
    else:
        config = default_cfg
        for k, v in args.items():
            config[k] = v
    if not config['run_id']:
        config['run_id'] = get_current_time_string()
    config['dataset_path'] = f"./data/{config['dataset']}/rdf"
    if args['do_train']:
        if 'lora_config' not in config:
            lora_modules = [el+'_proj' for el in args['lora_modules'].split('-') if el] if args['lora_modules'] != 'ft' else None
            config['lora_config'] = {
                    'r': 16,
                    'lora_alpha': 16,
                    'target_modules': lora_modules,
                    'lora_dropout': 0,
                    'bias': "none",
                    'task_type': "CAUSAL_LM",
                    'use_rslora': False,
            } if config['lora_modules'] else None
        config['lr'] = 2e-4 if config['lora_config']['target_modules'] else 1e-5
    config['model_name_string'] = config['model_name'].replace('/', '-')
    if '.testing' in config['results_dir']:
        config['results_dir'] = os.path.join(config['results_dir'], config['model_name_string'])
    else:
        run_id_list = config['run_id'].split('-')
        run_id_list[0] = run_id_list[0] + '_' + config['model_name_string']
        config['results_dir'] = set_save_dir(config['results_dir'], run_id_list, './results')
    config['model_dir'] = os.path.join(config['results_dir'], 'model')
    make_dir(config['model_dir'])
    model_chat_dict = yaml.safe_load(open('./model_info/model_chat_dict.yaml', 'r'))
    
    if config['desc_schema']:
        config['relations_path'] = f"./data/{config['dataset']}/rdf/relations.json"
        config['ent_classes_path'] = f"./data/{config['dataset']}/rdf/ent_classes.json"
    else:
        config['relations_path'] = f"./data/{config['dataset']}/rdf/relations_list.json"
        config['ent_classes_path'] = f"./data/{config['dataset']}/rdf/ent_classes_list.json"
    
    config['chat'] = model_chat_dict[config['model_name']]

    config_path = os.path.join(config['results_dir'], 'config.json')
    print(config)
    save_json(config, config_path)
    print(f'Config saved to: {config_path}')

    prompt_config_path = os.path.join('prompts', config['prompt_filename'])
    with open(prompt_config_path) as f:
        config['prompt_config'] = yaml.safe_load(f)

    return config

def save_prompt(text, txt_path):
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

def save_json(data, json_path):
    with open(json_path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii = False, indent = 4)

def save_json_extend(info, json_path):
    try:
        with open(json_path, 'r', encoding='utf8') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    
    data.extend(info)

    with open(json_path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii = False)

def set_save_dir(save_dir, save_suffix_list = [], default_save_dir = './results'):
    if not save_dir:
        save_dir = default_save_dir
        if save_suffix_list:
            save_dir = os.path.join(save_dir, *save_suffix_list, get_current_time_string())
        else:
            save_dir = os.path.join(save_dir, get_current_time_string())
    if not os.path.exists(save_dir):
        make_dir(save_dir)
        print(f"Created dir: {save_dir}")
    else:
        print('results_dir already exists, is this a re-run?')
        print('make sure you are not overwriting inadvertedly!')
    return save_dir

def get_current_time_string():
    return datetime.now().strftime("%Y-%m-%d--%H:%M:%S")

def get_current_time_string():
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

def make_dir(dirname: str):
    """
        make directory
    """
    Path(dirname).mkdir(parents=True, exist_ok=True)