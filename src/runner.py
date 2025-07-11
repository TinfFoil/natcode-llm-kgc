import os
import textwrap
from typing import List
from tqdm.auto import tqdm
import pandas as pd
import re
import uuid
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, Normalize
import numpy as np
import warnings
from utils_io import dict_to_records
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
        self.code_schema_prompt = open(config['schema_path'], 'r', encoding='utf8').read()

        if self.config['natlang']:
            self.comment_symbol = ''
            self.instruction = 'Task: Extract a list of [entity, relation, entity] triples from the text below.'
            self.sys_prompt = 'You are an AI specialized in the task of extracting entity-relation-entity triples from texts.'
        else:
            self.comment_symbol = '# '
            self.instruction = f'{self.comment_symbol}Task: Define an instance of Extract from the text below.'
            self.sys_prompt = f'You are a programming AI specialized in the task of extracting entity-relation-entity triples from texts in the form of Python code.'
        self.instruction += ' Do not produce any more text samples after you finish extracting triples from the text below.'
    
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
            triple_prompt = self.make_natlang_triples(triples) if self.config['natlang'] else self.make_python_triples(triples)
        if self.config['natlang']:
            out = f"""text: \"{item_text}\"{rationale_prompt}\n{triple_prompt}"""
        else:
            out = f"""text = \"\"\" {item_text} \"\"\"{rationale_prompt}\n{triple_prompt}"""
        return out

    def make_code_prompt(self, ICL_prompt: str, sample_text: str, triples: List[List[str]]):
        prompt = self.code_schema_prompt + '\n'
        rationale_prompt = self.make_rationale_prompt(triples) if self.config['rationale'] else ''
        if self.config['chat']:
            text_instruct = f"""{self.instruction}\n{self.format_sample(sample_text, [], rationale_prompt)}"""
            if triples:
                text_triples = self.make_python_triples(triples)
                prompt +=  ICL_prompt + '\n' + text_instruct
                if self.check_system_msg():
                    prompt = [
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text_triples},
                    ]
                else:
                    prompt = self.comment_symbol + self.sys_prompt + '\n\n' + prompt
                    prompt = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text_triples},
                    ]
                add_generation_prompt = False
            else:
                prompt +=  ICL_prompt + '\n' + text_instruct
                if self.check_system_msg():
                    prompt = [
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": prompt},
                    ]
                else:
                    prompt = self.comment_symbol + self.sys_prompt + '\n\n' + prompt
                    prompt = [
                    {"role": "user", "content": prompt},
                    ]
                add_generation_prompt = True
            prompt = self.tokenizer.apply_chat_template(prompt,
                                                        tokenize=False,
                                                        add_generation_prompt=add_generation_prompt
                                                        )
        else:
            text_instruct = f"""{self.instruction}\n{self.format_sample(sample_text, triples, rationale_prompt)}"""
            prompt += ICL_prompt + '\n' + text_instruct
            prompt = self.comment_symbol + self.sys_prompt + '\n\n' + prompt
        return prompt
    
    def make_natlang_prompt(self, ICL_prompt: str, sample_text: str, triples: List[List[str]]):
        prompt = ''
        rationale_prompt = self.make_rationale_prompt(triples) if self.config['rationale'] else ''
        if self.config['chat']:
            text_instruct = f"""{self.instruction}\n{self.format_sample(sample_text, [], rationale_prompt)}"""
            if triples: 
                text_triples = self.make_natlang_triples(triples)
                prompt +=  ICL_prompt + '\n' + text_instruct
                if self.check_system_msg():
                    prompt = [
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text_triples},
                    ]
                else:
                    prompt = self.comment_symbol + self.sys_prompt + '\n\n' + prompt
                    prompt = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text_triples},
                    ]
                add_generation_prompt = False
            else:
                prompt +=  ICL_prompt + '\n' + text_instruct
                if self.check_system_msg():
                    prompt = [
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": prompt},
                    ]
                else:
                    prompt = self.comment_symbol + self.sys_prompt + '\n\n' + prompt
                    prompt = [
                    {"role": "user", "content": prompt},
                    ]
                add_generation_prompt = True
            prompt = self.tokenizer.apply_chat_template(prompt,
                                                        tokenize=False,
                                                        add_generation_prompt=add_generation_prompt
                                                        )
        else:
            text_instruct = f"""{self.instruction}\n{self.format_sample(sample_text, triples, rationale_prompt)}"""
            prompt += ICL_prompt + '\n' + text_instruct
            prompt = self.sys_prompt + '\n\n' + prompt
        return prompt

    def extract_triples(self, response: str) -> List[List[str]]:
        print('response:', response)
        if self.config['natlang']:
            pattern = r'\{"([^"]+)":\s+"([^"]+)"\},\s+"([^"]+)",\s+\{"([^"]+)":\s+"([^"]+)"\}'
            matches = re.findall(pattern, response)
            # import pdb; pdb.set_trace()
            return [{'rel': {'text': match[2]},
                     'head': {'text': match[0], 'type': match[1]},
                     'tail': {'text': match[4], 'type': match[3]},
                     } for match in matches]
        else:
            raw_pattern = re.compile(r'Triple\(\s*(\w+)\("([^"]+)"\),\s*Rel\("([^"]+)"\),\s*(\w+)\("([^"]+)"\)\)?')
            pattern = re.compile(raw_pattern)
            matches = pattern.findall(response)
            return [[match[1], match[2], match[4]] for match in matches]

    def make_python_triples(self, triple_list: List[List[str]]) -> str:
        type_dict = self.config['type_dict']
        if triple_list:
            pythonic_triples = "extract = Extract(["
            for i, triple in enumerate(triple_list):
                head, edge, tail = triple
                pythonic_triples += f'Triple({type_dict[head]}("{head}"), Rel("{edge}"), {type_dict[tail]}("{tail}")'
                if i < len(triple_list) - 1:
                    pythonic_triples += "), "
                else:
                    pythonic_triples += ")])"
            return pythonic_triples
        else:
            return ''

    def make_natlang_triples(self, triple_list: List[List[str]]) -> str:
        natlang_triples = "triple_list: ["
        for i, triple in enumerate(triple_list):
            natlang_triples += '''[{{\"{head_text}\": \"{head_type}\"}}, \"{rel}\", {{\"{tail_text}\": \"{tail_type}\"}}]'''.format(
                head_text=triple['head']['text'],
                head_type=triple['head']['type'],
                tail_text=triple['tail']['text'],
                tail_type=triple['tail']['type'],
                rel=triple['rel']['text'],
            )
            
            if i < len(triple_list) - 1:
                natlang_triples += ", "
            else:
                natlang_triples += "]"
        return natlang_triples
    
    def make_rationale_prompt(self, triple_list):
        if triple_list:
            if self.config['natlang']:
                rels = '\n'.join([triple['rel']['text'] for triple in triple_list])
                ents = '\n'.join(['\n'.join([triple['head']['text'], triple['tail']['text']]) for triple in triple_list])
            else:
                rels = '\n'.join([f"{self.comment_symbol}Rel('{triple['rel']['text']}')" for triple in triple_list])
                ents = '\n'.join([f"{self.comment_symbol}{triple['head']['type']}('{triple['head']['text']}')\n{self.comment_symbol}{triple['head']['type']}('{triple['head']['text']}')" for triple in triple_list])
            rationale_prompt = f'''\n{self.comment_symbol}The candidate relations for this text are:\n{rels}\n{self.comment_symbol}The candidate entities for this text are:\n{ents}\n'''
        else:
            rationale_prompt = ''
        return rationale_prompt

    def make_icl_prompt(self, df: pd.DataFrame, index: int|None = None) -> str:
        df_filtered = df.drop(index) if index else df
        icl_rows = df_filtered.sample(n=self.config['n_icl_samples'])
        text_list = icl_rows['text'].to_list()
        triple_list = icl_rows['triple_list'].to_list()
        prompt = f'{self.comment_symbol}Look at the examples below and then carry out the following indicated task.\n\n'
        if self.config['natlang']:
            for i, (text, triples) in enumerate(zip(text_list, triple_list)):
                rationale_prompt = '' if not self.config['rationale'] else self.make_rationale_prompt(triples)
                prompt += f"""{self.comment_symbol}Example {i+1}:\ntext: \"{text}\"{rationale_prompt}\n{self.make_natlang_triples(triple_list=triples)}\n\n"""
        else:
            for i, (text, triples) in enumerate(zip(text_list, triple_list)):
                rationale_prompt = '' if not self.config['rationale'] else self.make_rationale_prompt(triples)
                prompt += f"""{self.comment_symbol}Example {i+1}:\ntext = \"\"\" {text} \"\"\"{rationale_prompt}\n{self.make_python_triples(triple_list=triples)}\n\n"""
        return prompt
    
    def run_model(self, texts, icl_prompt_list):
        if isinstance(texts, str):
            texts = [texts]

        prompt_func = self.make_natlang_prompt if self.config['natlang'] else self.make_code_prompt
        prompts = [prompt_func(icl_prompt, txt, []) for txt, icl_prompt in zip(texts, icl_prompt_list)]

        tokenized = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            return_token_type_ids=False,
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **tokenized,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=1000
            )

        decoded = []
        for i, seq in enumerate(outputs):
            in_len = tokenized["input_ids"].shape[-1]
            print('in_len:', in_len)
            gen_tokens = seq[in_len:] # only get the generated tokens
            decoded_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            decoded.append(decoded_text)
        return decoded

    def run_evaluation(self, df_test, df_train):
        trues, preds, outputs = [], [], []
        pbar = tqdm(range(0, len(df_test), self.config['batch_size_eval']),
                    desc=f"Model: {self.config['model_name_string']}")
        for start_idx in pbar:
            end_idx = min(start_idx + self.config['batch_size_eval'], len(df_test))
            batch_texts = df_test['text'][start_idx:end_idx]
            batch_triples = df_test['triple_list'][start_idx:end_idx]
            icl_prompt_list = [self.make_icl_prompt(df_train) for _ in range(self.config['batch_size_eval'])]
            results = self.run_model(batch_texts, icl_prompt_list)

            for text, trues_sample, output in zip(batch_texts, batch_triples, results):
                preds_sample = self.extract_triples(output)
                trues.append(trues_sample)
                preds.append(preds_sample)
                outputs.append(output)

                if self.config['verbose_preds']:
                    output_dict = json.dumps({'text': text,
                                    'true': trues_sample,
                                    'pred': preds_sample,
                                    'output': output,
                                    }, indent = 4)
                    print(output_dict)
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
    
    def evaluate(self, df_test, df_train):
        run = self.run_evaluation(df_test, df_train)

        self.save_logs(run)

        results = self.evaluator.calculate_strict_micro_f1(run['trues'], run['preds'])
        rel_type_metrics = self.get_type_metrics(run['trues'], run['preds'])
        
        return {
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1_score'],
            'rel_type_metrics': rel_type_metrics
        }

    def get_type_metrics(self, trues, preds):
        all_true_rel_types = set([el['rel']['text'] for t_list in trues for el in t_list])
        rel_trues = {k: [] for k in all_true_rel_types}
        rel_preds = {k: [] for k in all_true_rel_types}
        for trues_sample, preds_sample in zip(trues, preds):
            for rel_type in all_true_rel_types:
                trues_type = [t for t in trues_sample if t['rel']['text'] == rel_type]
                preds_type = [p for p in preds_sample if p['rel']['text'] == rel_type]
                rel_trues[rel_type].append(trues_type)
                rel_preds[rel_type].append(preds_type)
        return {k: self.evaluator.calculate_strict_micro_f1(rel_trues[k], rel_preds[k])
                            for k in all_true_rel_types}
        
    def save_logs(self, run):
        records = dict_to_records(run)
        preds_log = [{
            'text': el['texts'],
            'true': el['trues'],
            'pred': el['preds'],
            'output': el['outputs'],
            } for el in records]
        preds_log = preds_log.lstrip()
        with open(self.config['log_path'], 'w', encoding='utf8') as f:
            f.write(preds_log)

    def make_samples(self, tokenizer, df: pd.DataFrame) -> List[str]:
        data = []
        max_prompt_len = 0
        EOS_TOKEN = tokenizer.eos_token if not self.config['chat'] else ''
        for index in tqdm(df.index, total=len(df)):
            sample = df.loc[index]
            sample_text = sample['text']
            sample_triples = sample['triple_list']
            icl_prompt = self.make_icl_prompt(df, index)
            prompt = ''

            if self.config['natlang']:
                prompt = self.make_natlang_prompt(icl_prompt, sample_text, sample_triples) + EOS_TOKEN
            else:
                prompt = self.make_code_prompt(icl_prompt, sample_text, sample_triples) + EOS_TOKEN
            
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