import os
from typing import List, Dict
from tqdm.auto import tqdm
import pandas as pd
import re
import uuid
import torch
import warnings
from utils_io import dict_to_records, save_json, save_prompt
import json

class Runner:
    def __init__(self,
                 *,
                 model,
                 tokenizer,
                 config,
                 evaluator,
                 ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.evaluator = evaluator
        self.think = config['enable_thinking']
        self.sys_prompt = config['prompt_config']['sys_prompt']
        self.natlang_triple_layout = config['prompt_config']['natlang_triple_layout']
        self.icl_instruction = config['prompt_config']['icl_instruction']
        self.rationale_prompt = config['prompt_config']['rationale_prompt']

    def make_train_set(self, df: pd.DataFrame, tokenizer, n_samples: int = 0):
        data_train = self.make_train_samples(tokenizer, df)
        df = pd.DataFrame(data_train)
        if n_samples:
            df = df.sample(n=n_samples, random_state=42)
        return df

    def check_system_msg(self):
        psw = str(uuid.uuid4())
        response = self.tokenizer.apply_chat_template([{"role": "system", "content": psw},], tokenize=False)
        if psw in response:
            return True
        else:
            return False

    def format_sample(self, item_text: str, triples: List[List[str]], rationale_prompt: str):
        if not triples:
            triple_prompt = ''
        else:
            triple_prompt = self.make_triples(triples)
        out = f"""text: \"{item_text}\"{rationale_prompt}\n{triple_prompt}"""
        return out
    
    def make_prompt(self, ICL_prompt: str, sample_text: str, triples: List[List[str]]):
        prompt = ''
        rationale_prompt = self.make_rationale_prompt(triples) if self.config['rationale'] else ''
        instruction = self.config['prompt_config']['instruction'].format(
            natlang_triple_layout=self.natlang_triple_layout
                                      .replace("{{", "{")
                                      .replace("}}", "}"),
            ent_classes = open(self.config['ent_classes_path'], 'r', encoding='utf8').read(),
            relations = open(self.config['relations_path'], 'r', encoding='utf8').read(),
        )
        if self.config['chat']:
            text_instruct = f"""{instruction}\n{self.format_sample(sample_text, [], rationale_prompt)}"""
            prompt +=  ICL_prompt + '\n' + text_instruct
            if triples: 
                text_triples = self.make_triples(triples)
                if self.check_system_msg():
                    prompt = [
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text_triples},
                    ]
                else:
                    prompt = self.sys_prompt + prompt
                    prompt = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text_triples},
                    ]
                add_generation_prompt = False
            else:
                if self.check_system_msg():
                    prompt = [
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": prompt},
                    ]
                else:
                    prompt = self.sys_prompt + prompt
                    prompt = [
                    {"role": "user", "content": prompt},
                    ]
                add_generation_prompt = True
            apply_chat_template_kwargs = {
                'tokenize': False,
                'add_generation_prompt': add_generation_prompt,
            }
            '''
                `enable_thinking` == True  --> `<|im_start|>assistant\n`
                `enable_thinking` == False --> `<|im_start|>assistant\n<think>\n\n</think>\n\n`
            '''
            if 'qwen3' in self.config['model_name'].lower():
                apply_chat_template_kwargs.update({'enable_thinking': bool(self.think)})
            prompt = self.tokenizer.apply_chat_template(prompt,
                                                        **apply_chat_template_kwargs
                                                        )
        else:
            text_instruct = f"""{instruction}\n{self.format_sample(sample_text, triples, rationale_prompt)}"""
            prompt += ICL_prompt + '\n' + text_instruct
            prompt = self.sys_prompt + '\n\n' + prompt
        return prompt

    def extract_triples(self, model_output: str) -> List[str]:
        print('model_output:', model_output)
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', model_output, re.DOTALL)
        if not json_match:
            print('No JSON-like triple found in response.')
            return []
        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            print('JSON decode error:', e)
            return []
        return data

    def make_triples(self, triple_list: List[List[str]]) -> str:
        natlang_triples = "triple_list: ["
        for i, triple in enumerate(triple_list):
            natlang_triples += self.natlang_triple_layout.format(
                relation_type=triple['rel']['type'],
                entity_head=triple['head']['text'],
                entity_type_head=triple['head']['type'],
                entity_tail=triple['tail']['text'],
                entity_type_tail=triple['tail']['type'],
            )
            
            if i < len(triple_list) - 1:
                natlang_triples += ", "
            else:
                natlang_triples += "]"
        return natlang_triples
    
    def make_rationale_prompt(self, triple_list):
        if triple_list:
            rationale_prompt = f'''\n{self.rationale_prompt}\n'''.format(
                rels = '\n'.join([triple['rel']['type'] for triple in triple_list]),
                ents = '\n'.join(['\n'.join([triple['head']['text'], triple['tail']['text']]) for triple in triple_list])
            )
        else:
            rationale_prompt = ''
        return rationale_prompt

    def make_icl_prompt(self, df: pd.DataFrame, index: int|None = None) -> str:
        df_filtered = df.drop(index) if index else df
        n = self.config['n_icl_samples']
        icl_rows = df_filtered.sample(n=n, random_state=42)
        text_list = icl_rows['text'].to_list()
        triple_list = icl_rows['triple_list'].to_list()
        if n > 0:
            prompt = f'\n{self.icl_instruction}\n\n'
        else:
            prompt = '\n'
        for i, (text, triples) in enumerate(zip(text_list, triple_list)):
            rationale_prompt = '' if not self.config['rationale'] else self.make_rationale_prompt(triples)
            prompt += f"""Example {i+1}:\ntext: \"{text}\"{rationale_prompt}\n{self.make_triples(triple_list=triples)}\n"""
        return prompt
    
    def run_model(self, prompts):
        tokenized = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding='longest', 
            truncation=True,
            return_token_type_ids=False,
            add_special_tokens=False,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **tokenized,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.config['max_new_tokens'],
            )

        print('CUDA GBs allocated:', torch.cuda.memory_allocated() / 1024**3)
        print('CUDA GBs reserved:', torch.cuda.memory_allocated() / 1024**3)
        
        decoded = []
        for i, seq in enumerate(outputs):
            in_len = tokenized["input_ids"].shape[-1]
            gen_tokens = seq[in_len:] # only get the generated tokens
            decoded_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            decoded.append(decoded_text)
        return decoded

    def run_evaluation(self, df_test, df_train, split):
        trues, preds, outputs = [], [], []
        pbar = tqdm(range(0, len(df_test), self.config['batch_size_eval']))
        prompt_saved = 0
        for start_idx in pbar:
            end_idx = min(start_idx + self.config['batch_size_eval'], len(df_test))
            batch_texts = df_test['text'][start_idx:end_idx]
            batch_triples = df_test['triple_list'][start_idx:end_idx]
            # NOTE: we can use the whole train set to sample ICL examples during evalution
            # so there's no need to pass `index` to `make_icl_prompt`
            icl_prompt_list = [self.make_icl_prompt(df_train) for _ in range(len(batch_texts))]
            if isinstance(batch_texts, str):
                batch_texts = [batch_texts]
            prompts = [self.make_prompt(icl_prompt, txt, []) for txt, icl_prompt in zip(batch_texts, icl_prompt_list)]
            if self.config['save_prompt'] and not prompt_saved:
                save_prompt(prompts[0], txt_path = os.path.join(self.config['results_dir'], f'{split}_prompt.txt'))
                prompt_saved = 1
            results = self.run_model(prompts)
            
            for text, trues_sample, full_output in zip(batch_texts, batch_triples, results):
                filtered_output = filter_think(full_output)
                preds_sample = self.extract_triples(filtered_output)
                trues.append(trues_sample)
                preds.append(preds_sample)
                outputs.append(filtered_output)

                if self.config['verbose_preds']:
                    print_dict = json.dumps({'text': text,
                                    'true': trues_sample,
                                    'pred': preds_sample,
                                    'filtered': filtered_output,
                                    'full_output': full_output,
                                    }, indent = 4)
                    print(print_dict, flush=True)
                if self.config['verbose_metrics']:
                    metrics_sample = self.evaluator.calculate_strict_micro_f1([trues_sample], [preds_sample])
                    metrics_current = self.evaluator.calculate_strict_micro_f1(trues, preds)
                    pbar.set_description(f"F1: {round(metrics_current['f1_score'], 2)}")
                    print(f'metrics_sample: {metrics_sample}')
                    print(f'metrics_current: {metrics_current}')
        return {
            'texts': df_test['text'].tolist(),
            'trues': trues,
            'preds': preds,
            'outputs': outputs,
        }
    
    def evaluate(self, df_test, df_train, split = 'test'):
        run = self.run_evaluation(df_test, df_train, split=split)

        self.save_logs(run, split=split)
        results = self.evaluator.calculate_strict_micro_f1(run['trues'], run['preds'])
        rel_type_metrics = self.get_type_metrics(run['trues'], run['preds'])
        
        return {
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1_score'],
            'rel_type_metrics': rel_type_metrics
        }

    def get_type_metrics(self, trues, preds):
        all_true_rel_types = set([el['rel']['type'] for t_list in trues for el in t_list])
        rel_trues = {k: [] for k in all_true_rel_types}
        rel_preds = {k: [] for k in all_true_rel_types}
        for trues_sample, preds_sample in zip(trues, preds):
            for rel_type in all_true_rel_types:
                trues_type = [t for t in trues_sample if t['rel']['type'] == rel_type]
                preds_type = [p for p in preds_sample if p['rel']['type'] == rel_type]
                rel_trues[rel_type].append(trues_type)
                rel_preds[rel_type].append(preds_type)
        return {k: self.evaluator.calculate_strict_micro_f1(rel_trues[k], rel_preds[k])
                            for k in all_true_rel_types}
        
    def save_logs(self, run, split = 'test'):
        records = dict_to_records(run)
        preds_log = [{
            'text': el['texts'],
            'true': el['trues'],
            'pred': el['preds'],
            'output': el['outputs'],
            } for el in records]
        save_json(preds_log, os.path.join(self.config['results_dir'], f"preds_{split}.json"))

    def make_train_samples(self, tokenizer, df: pd.DataFrame) -> List[str]:
        data = []
        max_prompt_len = 0
        EOS_TOKEN = tokenizer.eos_token if not self.config['chat'] else ''
        for index in tqdm(df.index, total=len(df)):
            sample = df.loc[index]
            sample_text = sample['text']
            sample_triples = sample['triple_list']
            icl_prompt = self.make_icl_prompt(df, index)
            prompt = ''

            prompt = self.make_prompt(icl_prompt, sample_text, sample_triples) + EOS_TOKEN
            
            prompt_token_len = len(tokenizer(prompt).input_ids)
            if prompt_token_len > self.config['max_length']:
                warnings.warn(f"The prompt is longer than the set maximum sequence length ({self.config['max_length']})")
            if prompt_token_len > max_prompt_len:
                max_prompt_len = prompt_token_len
            
            data.append({
                'text': prompt,
                'triple_list': sample_triples,
            })
        print('max_prompt_len:', max_prompt_len)
        return data

def filter_think(model_output: str, think_token_string:str = "</think>"):
    if think_token_string in model_output:
        think_token_string_pos = model_output.rfind(think_token_string) + len(think_token_string)
        response = model_output[think_token_string_pos:]
    else:
        response = model_output
    return response