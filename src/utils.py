import os
import textwrap
from typing import List
from tqdm.auto import tqdm
import pandas as pd
import json
import re
import logging
import uuid
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, Normalize
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from transformers.trainer_callback import TrainerControl, TrainerState, TrainerCallback
from transformers import TrainingArguments
import random

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

class Runner:
    def __init__(self,
                 *,
                 model,
                 type_dict,
                 natlang,
                 schema_path = '',
                 chat_model = False,
                 tokenizer = None,
                 rationale = False,
                 verbose_train = False,
                 verbose_test = False,
                 max_seq_len = 999999,
                 model_name = '',
                 verbose_output_path = '',
                 ) -> None:
        self.model = model
        self.type_dict = type_dict
        self.natlang = natlang
        self.chat_model = chat_model
        self.tokenizer = tokenizer
        self.schema_prompt = open(schema_path, 'r', encoding='utf8').read()
        self.rationale = rationale
        self.verbose_train = verbose_train
        self.verbose_test = verbose_test
        self.max_seq_len = max_seq_len
        self.model_name = model_name.split('/')[-1]
        if verbose_output_path:
            self.verbose_output_path = os.path.join(verbose_output_path, f'{self.model_name}.txt')
        self.verbose_output = ''

        if self.natlang:
            self.comment_symbol = ''
            self.instruction = 'Task: Extract a list of [entity, relation, entity] triples from the text below.'
            self.sys_prompt = 'You are an AI specialized in the task of extracting entity-relation-entity triples from texts.'
            # self.quitting_prompt = '\n\n' + f'{self.comment_symbol}Task completed!'
        else:
            self.comment_symbol = '# '
            self.instruction = f'{self.comment_symbol}Task: Define an instance of Extract from the text below.'
            self.sys_prompt = f'You are a programming AI specialized in the task of extracting entity-relation-entity triples from texts in the form of Python code.'
            # self.quitting_prompt = '\n\n' + f'{self.comment_symbol}Task completed!\n\nexit()'
        self.instruction += ' Do not produce any more text samples after you finish extracting triples from the text below.'

        self.quitting_prompt = ''
        # TODO check if quitting prompt can actually help, but it doesn't seem like it
    
    def check_system_msg(self):
        psw = str(uuid.uuid4())
        response = self.tokenizer.apply_chat_template([{"role": "system", "content": psw},], tokenize=False)
        if psw in response:
            return True
        else:
            return False

    def format_sample(self, item_text: str, triples: List[List[str]], rationale_prompt: str):
        if not triples and self.rationale:
            triple_prompt = ''
        else:
            triple_prompt = self.natlang_triples(triples) if self.natlang else self.pythonize_triples(triples)
        if self.natlang:
            out = f"""text: \"{item_text}\"{rationale_prompt}\n{triple_prompt}"""
        else:
            out = f"""text = \"\"\" {item_text} \"\"\"{rationale_prompt}\n{triple_prompt}"""
        return out

    def make_code_prompt(self, ICL_prompt: str, sample_text: str, triples: List[List[str]]):
        prompt = self.schema_prompt + '\n'
        rationale_prompt = self.make_rationale_prompt(triples) if self.rationale else ''
        if self.chat_model:
            text_instruct = f"""{self.instruction}\n{self.format_sample(sample_text, [], rationale_prompt)}"""
            if triples: 
                text_triples = self.pythonize_triples(triples)
                prompt +=  ICL_prompt + '\n' + text_instruct
                prompt += self.quitting_prompt
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
            if triples:
                prompt += self.quitting_prompt
        return prompt
    
    def make_natlang_prompt(self, ICL_prompt: str, sample_text: str, triples: List[List[str]]):
        prompt = ''
        rationale_prompt = self.make_rationale_prompt(triples) if self.rationale else ''
        if self.chat_model:
            text_instruct = f"""{self.instruction}\n{self.format_sample(sample_text, [], rationale_prompt)}"""
            if triples: 
                text_triples = self.natlang_triples(triples)
                prompt +=  ICL_prompt + '\n' + text_instruct
                prompt += self.quitting_prompt
                if self.check_system_msg():
                    prompt = [
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text_triples},
                    ]
                else:
                    prompt = self.comment_symbol + self.sys_prompt + '\n\n' + prompt
                    prompt += self.quitting_prompt
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
            if triples:
                prompt += self.quitting_prompt
        return prompt

    def extract_triples(self, response: str) -> List[List[str]]:
        # Find the position of the instruction in the response
        # instruction_pos = response.rfind(self.instruction)
        # if instruction_pos != -1:
        #     # If instruction is found, start extracting from after it
        #     response = response[instruction_pos + len(self.instruction):]
        
        if self.natlang:
            # Extract triples from natural language response
            pattern = r'\["([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\]'
            matches = re.findall(pattern, response)
            return list(matches)
        else:
            # Extract triples from pythonic response
            raw_pattern = re.compile(r'Triple\(\s*(\w+)\("([^"]+)"\),\s*Rel\("([^"]+)"\),\s*(\w+)\("([^"]+)"\)\)?')
            pattern = re.compile(raw_pattern)
            matches = pattern.findall(response)
            return [[match[1], match[2], match[4]] for match in matches]

    def pythonize_triples(self, triple_list: List[List[str]]) -> str:
        if triple_list:
            pythonic_triples = "extract = Extract(["
            for i, triple in enumerate(triple_list):
                head, edge, tail = triple
                pythonic_triples += f'Triple({self.type_dict[head]}("{head}"), Rel("{edge}"), {self.type_dict[tail]}("{tail}")'
                
                if i < len(triple_list) - 1:
                    pythonic_triples += "), "
                else:
                    pythonic_triples += ")])"
            return pythonic_triples
        else:
            return ''

    def natlang_triples(self, triple_list: List[List[str]]) -> str:
        natlang_triples = "triple_list: ["
        for i, triple in enumerate(triple_list):
            head, edge, tail = triple
            natlang_triples += f'["{head}", "{edge}", "{tail}"]'
            
            if i < len(triple_list) - 1:
                natlang_triples += ", "
            else:
                natlang_triples += "]"
        return natlang_triples
    
    def make_rationale_prompt(self, triple_list):
        if triple_list:
            if self.natlang:
                rels = '\n'.join([triple[1] for triple in triple_list])
                ents = '\n'.join(['\n'.join([triple[0], triple[2]]) for triple in triple_list])
            else:
                rels = '\n'.join([f"{self.comment_symbol}Rel('{triple[1]}')" for triple in triple_list])
                ents = '\n'.join([f"{self.comment_symbol}{self.type_dict[triple[0]]}('{triple[0]}')\n{self.comment_symbol}{self.type_dict[triple[2]]}('{triple[2]}')" for triple in triple_list])
            rationale_prompt = f'''\n{self.comment_symbol}The candidate relations for this text are:\n{rels}\n{self.comment_symbol}The candidate entities for this text are:\n{ents}\n'''
        else:
            rationale_prompt = ''
        return rationale_prompt

    def make_icl_prompt(self, data: pd.DataFrame) -> str:
        text_list = data['text'].to_list()
        triple_list = data['triple_list'].to_list()
        prompt = f'{self.comment_symbol}Look at the examples below and then carry out the following indicated task.\n\n'
        
        if self.natlang:
            for i, (text, triples) in enumerate(zip(text_list, triple_list)):
                rationale_prompt = '' if not self.rationale else self.make_rationale_prompt(triples)
                prompt += f"""{self.comment_symbol}Example {i+1}:\ntext: \"{text}\"{rationale_prompt}\n{self.natlang_triples(triple_list=triples)}\n\n"""
        else:
            for i, (text, triples) in enumerate(zip(text_list, triple_list)):
                rationale_prompt = '' if not self.rationale else self.make_rationale_prompt(triples)
                prompt += f"""{self.comment_symbol}Example {i+1}:\ntext = \"\"\" {text} \"\"\"{rationale_prompt}\n{self.pythonize_triples(triple_list=triples)}\n\n"""
        return prompt

    def calculate_strict_micro_f1(self, trues: List[List[List[str]]], preds: List[List[List[str]]]) -> List[float]:
        TP = 0
        FP = 0
        FN = 0
        
        for true_triples, pred_triples in zip(trues, preds):
            trues_merged = ['|||'.join(trues) for trues in true_triples]
            preds_merged = ['|||'.join(preds) for preds in pred_triples]
            true_set = set(trues_merged)
            pred_set = set(preds_merged)
            
            TP += len(true_set & pred_set)
            FP += len(pred_set - true_set)
            FN += len(true_set - pred_set)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return [precision, recall, f1_score]
    
    def run_model(self, text, icl_prompt):
        if self.natlang:
            prompt = self.make_natlang_prompt(ICL_prompt=icl_prompt, sample_text=text, triples=[])
        else:
            prompt = self.make_code_prompt(ICL_prompt=icl_prompt, sample_text=text, triples=[])
        
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        
        t0 = time.time()
        outputs = self.model.generate(inputs,
                                num_return_sequences=1,
                                eos_token_id=self.tokenizer.eos_token_id,
                                pad_token_id=self.tokenizer.eos_token_id,
                                max_new_tokens = 1000,
                                )
        t1 = time.time()
        t_diff = t1 - t0
        input_len = inputs.shape[1]
        result = self.tokenizer.decode(outputs[0][input_len:],
                                            skip_special_tokens=True
                                            )
        timeout = 30
        if t_diff > timeout:
            self.t_counter += 1
            if self.t_counter > 2:
                raise TimeoutError(f'Generation for {self.model_name} is taking longer than {timeout} seconds, exiting...')
        else:
            self.t_counter -= 1
        return result

    def evaluate(self, df_test, icl_prompt):
        trues = []
        preds = []
        
        text_test = df_test['text'].to_list()
        triple_list_test = df_test['triple_list'].to_list()
        all_rel_types = set([el[1] for list in triple_list_test for el in list])
        rel_type_preds_trues = {k: {'trues': [], 'preds': []} for k in all_rel_types}

        pbar = tqdm(zip(text_test, triple_list_test), total=len(df_test), desc=f'Model: {self.model_name}')
        
        self.t_counter = 0 # initialize model timeout counter
        
        for text, triple_list in pbar:
            result = self.run_model(text, icl_prompt)
            
            trues.append(triple_list)
            pred = self.extract_triples(result)
            preds.append(pred)
            for el in triple_list:
                rel_type_preds_trues[el[1]]['preds'].append(pred)
                rel_type_preds_trues[el[1]]['trues'].append(triple_list)
            
            if self.verbose_test:
                metrics_sample = self.calculate_strict_micro_f1([triple_list], [pred])
                metrics_current = self.calculate_strict_micro_f1(trues, preds)
                pbar.set_description(f'F1: {round(metrics_current[-1], 2)}')
                output_texts = '\n' + f'text: {text}' + '\n' + f'result: {result}' + '\n' + f'pred: {pred}' + '\n' + f'trues: {triple_list}'
                output_metrics = '\n' + f'metrics_sample: {metrics_sample}' + '\n' + f'metrics_current: {metrics_current}'
                logger.info(output_texts)
                logger.info(output_metrics)
                if self.verbose_output_path:
                    self.verbose_output += output_texts + output_metrics
        if self.verbose_output:
            self.verbose_output = self.verbose_output.lstrip()
            with open(self.verbose_output_path, 'w', encoding='utf8') as f:
                f.write(self.verbose_output)
        precision, recall, f1_score = self.calculate_strict_micro_f1(trues, preds)
        
        rel_type_metrics = {k: {}  for k in all_rel_types}

        for k, v in rel_type_preds_trues.items():
            rel_type_metrics[k]['prec'], rel_type_metrics[k]['rec'], rel_type_metrics[k]['f1'] = self.calculate_strict_micro_f1(v['trues'], v['preds'])

        return {'prec': precision,
                'rec': recall,
                'f1': f1_score,
                'rel_type_metrics': rel_type_metrics
                  }

    def make_samples(self, tokenizer, df: pd.DataFrame, n_icl_samples: int) -> List[str]:
        text_list = []
        EOS_TOKEN = tokenizer.eos_token # add EOS_TOKEN to prevent infinite generation
        for index in tqdm(df.index, total=len(df)):
            # Select the specific row
            sample = df.loc[index]
            
            # Exclude the specific row from the DataFrame
            df_nosample = df.drop(index)
            sample_text = sample['text']
            sample_triples = sample['triple_list']
            
            # Randomly select N other different rows from the remaining DataFrame
            icl_rows = df_nosample.sample(n=n_icl_samples)
            icl_prompt = self.make_icl_prompt(icl_rows)
            prompt = ''
            if self.natlang:
                prompt = self.make_natlang_prompt(icl_prompt, sample_text, sample_triples) + EOS_TOKEN
            else:
                prompt = self.make_code_prompt(icl_prompt, sample_text, sample_triples) + EOS_TOKEN
            if self.verbose_train:
                print(prompt)
                txt_path = f"./prompts/{self.model_name}.txt"
                with open(txt_path, 'w', encoding='utf8') as f:
                    f.write(prompt)
            prompt_token_len = len(tokenizer(prompt).input_ids)
            if prompt_token_len > self.max_seq_len:
                raise Exception(f'prompt is over {self.max_seq_len} tokens')
            text_list.append(prompt)
        
        return text_list
    
    def get_attn(self, model, tokenizer, sample, icl_prompt):
        text = sample['text']
        if self.natlang:
            prompt = self.make_natlang_prompt(ICL_prompt=icl_prompt, sample_text=text, triples=[])
        else:
            prompt = self.make_code_prompt(ICL_prompt=icl_prompt, sample_text=text, triples=[])
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(inputs.input_ids,
                                num_return_sequences=1,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.eos_token_id,
                                max_new_tokens = 1000,
                                return_dict_in_generate=True,
                                output_attentions = True,
                                )
        attn_list = []
        for i, tokens in enumerate(outputs.attentions[1:]):
            # for layer in tokens:
            attn = torch.mean(tokens[-1].squeeze(), dim = 0)
            attn_list.append(attn)

        input_len = inputs.input_ids.shape[1]
        
        output_no_prompt = outputs.sequences.squeeze()[input_len:]

        result = tokenizer.decode(output_no_prompt,
                                skip_special_tokens=True,
                                )
        true = sample['triple_list']        
        pred = self.extract_triples(result)
        print('pred', result)
        stats = self.calculate_strict_micro_f1([true], [pred])
        print('results:', stats)
        print('gt', true)

        return attn_list, outputs.sequences.squeeze(), stats

    def heatmap2d(self, attention_scores, token_ids_full, idx, width=20, height=None, cmap='viridis', norm='log', high_threshold=0.999, savename=''):
        attention_scores = attention_scores.float().cpu().numpy()
        num_tokens = attention_scores.shape[0]
        next_token = token_ids_full[:num_tokens][-1]
        token_ids = token_ids_full[:num_tokens]
        heatmap_token = self.tokenizer.decode(next_token)
        print('heatmap token:', heatmap_token)

        labels = [self.tokenizer.decode(token) for token in token_ids]
        assert len(attention_scores) == len(labels)
        
        # Calculate the number of rows and columns
        n_cols = width
        n_rows = int(np.ceil(num_tokens / n_cols))
        if height:
            n_rows = min(n_rows, height)
        
        # Pad the attention scores and labels if necessary
        pad_length = n_rows * n_cols - num_tokens
        attention_scores_padded = np.pad(attention_scores, (0, pad_length), mode='constant', constant_values=np.nan)
        labels_padded = labels + [''] * pad_length
        
        # Reshape the data into a 2D grid
        attention_scores_2d = attention_scores_padded.reshape(n_rows, n_cols)
        labels_2d = np.array(labels_padded).reshape(n_rows, n_cols)
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(n_cols * 1.5, n_rows * 1.5))
        
        # Choose the normalization
        if norm == 'log':
            norm = LogNorm(vmin=np.nanmin(attention_scores), vmax=np.nanmax(attention_scores))
        else:
            norm = Normalize(vmin=np.nanmin(attention_scores), vmax=np.nanmax(attention_scores))
        
        # Create the heatmap
        im = ax.imshow(attention_scores_2d, cmap=cmap, aspect='equal', norm=norm)
        
        # Apply black color to high values
        high_mask = attention_scores_2d > np.nanquantile(attention_scores, high_threshold)
        im.cmap.set_bad(color='black')
        attention_scores_2d_masked = np.ma.masked_where(high_mask, attention_scores_2d)
        im.set_data(attention_scores_2d_masked)
        
        # Function to wrap text
        def wrap_text(text, width=10):
            return "\n".join(textwrap.wrap(text, width=width))
        
        # Add labels to each cell
        for i in range(n_rows):
            for j in range(n_cols):
                text = labels_2d[i, j]
                score = attention_scores_2d[i, j]
                if not np.isnan(score):
                    wrapped_text = wrap_text(text)
                    ax.text(j, i, wrapped_text, ha='center', va='center', fontsize=20, color='white', fontweight='bold')
        
        # plt.title(heatmap_token)

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax, aspect=20)
        
        # Add text for high values
        # plt.text(1.02, 0.5, 'High\nAttention', transform=ax.transAxes, 
        #         va='center', ha='left', bbox=dict(facecolor='black', edgecolor='none', alpha=0.8, pad=5))
        
        plt.tight_layout()
        save_dir = f"./paper/attn/{savename}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'attn_{idx}_{heatmap_token}.pdf'), format='pdf', bbox_inches='tight')
        plt.close()

def save_json(info, json_path):
    try:
        with open(json_path, 'r', encoding='utf8') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    
    data.extend(info)

    with open(json_path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii = False)

class OutputPrinterCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, print_interval=10):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.print_interval = print_interval

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        if state.global_step % self.print_interval == 0:
            # Select a random sample from the dataset
            sample = random.choice(self.dataset)
            inputs = self.tokenizer(sample['text'], return_tensors='pt', truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Perform a forward pass instead of generation
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get the most likely token at each step
            predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
            
            # Decode the predicted tokens
            generated_text = self.tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
            
            print(f"\nStep {state.global_step} - Sample Output:")
            print(f"Input: {sample['text'][:100]}...")  # Print first 100 characters of input
            print(f"Model output: {generated_text}\n")

        return control

class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            log_message = {key: logs[key] for key in ['loss', 'grad_norm', 'learning_rate', 'epoch'] if key in logs}
            logger.info(log_message)
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info(state.global_step)
        return super().on_step_begin(args, state, control, **kwargs)

chat_template_dict = {
    "llama": "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n",
    "mistral": "",
    "qwen1.5": "",
    }