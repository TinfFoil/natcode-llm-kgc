from typing import List
from tqdm.auto import tqdm
import pandas as pd
import json
import re
import logging
import uuid
import time
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

class Runner:
    def __init__(self,
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
                 ) -> None:
        self.type_dict = type_dict
        self.natlang = natlang
        self.chat_model = chat_model
        self.tokenizer = tokenizer
        self.schema_prompt = open(schema_path, 'r', encoding='utf8').read()
        self.rationale = rationale
        self.verbose_train = verbose_train
        self.verbose_test = verbose_test
        self.max_seq_len = max_seq_len
        self.model_name = model_name

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

    def calculate_micro_f1(self, trues: List[List[List[str]]], preds: List[List[List[str]]]) -> List[float]:
        TP = 0
        FP = 0
        FN = 0
        
        for true_triples, pred_triples in zip(trues, preds):
            trues_merged = ['-'.join(trues) for trues in true_triples]
            preds_merged = ['-'.join(preds) for preds in pred_triples]
            true_set = set(trues_merged)
            # if len(true_set) != len(trues_merged):
            #     raise AssertionError('List and set lengths are different')
            pred_set = set(preds_merged)
            
            TP += len(true_set & pred_set)
            FP += len(pred_set - true_set)
            FN += len(true_set - pred_set)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return [precision, recall, f1_score]

    def evaluate(self, model, tokenizer, df_test, icl_prompt):
        trues = []
        preds = []
        
        text_test = df_test['text'].to_list()
        triple_list_test = df_test['triple_list'].to_list()
        
        t_counter = 0
        pbar = tqdm(zip(text_test, triple_list_test), total=len(df_test), desc=f'Model: {self.model_name}')
        # pbar.set_description(f'F1: {0}')
        for text, triple_list in pbar:
            if self.natlang:
                prompt = self.make_natlang_prompt(ICL_prompt=icl_prompt, sample_text=text, triples=[])
            else:
                prompt = self.make_code_prompt(ICL_prompt=icl_prompt, sample_text=text, triples=[])
            
            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            
            t0 = time.time()
            outputs = model.generate(inputs,
                                    num_return_sequences=1,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.eos_token_id,
                                    max_new_tokens = 1000,
                                    )
            t1 = time.time()
            t_diff = t1 - t0
            input_len = inputs.shape[1]
            full_model_output = tokenizer.decode(outputs[0],
                                                skip_special_tokens=True
                                                )
            result = tokenizer.decode(outputs[0][input_len:],
                                                skip_special_tokens=True
                                                )
            timeout = 30
            if t_diff > timeout:
                t_counter += 1
                if t_counter > 2:
                    raise TimeoutError(f'Generation for {self.model_name} is taking longer than {timeout} seconds, exiting...')
            else:
                t_counter -= 1

            trues.append(triple_list)
            pred = self.extract_triples(result)
            preds.append(pred)
            if self.verbose_test:
                metrics_sample = self.calculate_micro_f1([triple_list], [pred])
                metrics_current = self.calculate_micro_f1(trues, preds)
                pbar.set_description(f'F1: {round(metrics_current[-1], 2)}')
                logger.info('\n' + f'result: {result}' + '\n' + f'pred: {pred}' + '\n' + f'trues: {triple_list}')
                logger.info('\n' + f'metrics_sample: {metrics_sample}' + '\n' + f'metrics_current: {metrics_current}')

        precision, recall, f1_score = self.calculate_micro_f1(trues, preds)
        return precision, recall, f1_score

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
                txt_path = f"./prompts/{self.model_name.split('/')[-1]}.txt"
                with open(txt_path, 'w', encoding='utf8') as f:
                    f.write(prompt)
            prompt_token_len = len(tokenizer(prompt).input_ids)
            if prompt_token_len > self.max_seq_len:
                raise Exception(f'prompt is over {self.max_seq_len} tokens')
            text_list.append(prompt)
        
        return text_list

def save_json(info, json_path):
    try:
        with open(json_path, 'r', encoding='utf8') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    
    data.extend(info)

    with open(json_path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii = False)

