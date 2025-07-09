import json
from datetime import datetime
import argparse

def setup_config(namespace: argparse.Namespace, default_cfg: dict):
    args = vars(namespace)
    config = default_cfg
    for k, v in args.items():
        config[k] = v
    if not config['suffix']:
        config['suffix'] = get_time()
    config['tag_dict'] = json.load(open(config['tag_dict_path'], 'r'))
    tags_path = './misc/prompt_tags_coarse.txt' if config['coarse'] else './misc/prompt_tags_fine.txt'
    config['prompt_tags'] = open(tags_path, 'r').read()
    config['prompt_layout'] = open(config['prompt_layout_path'], 'r').read()
    config['turn_types'] = config['turn_types'].split(',')
    return config

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
    print(f"Target modules: {args.target_modules}")
    print(f"Learning rate: {args.lr}")
    print('##################################################################')
