from typing import List
from tqdm.auto import tqdm
import pandas as pd
import json
import re
import logging
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
                 ) -> None:
        self.type_dict = type_dict
        self.natlang = natlang
        self.chat_model = chat_model
        self.tokenizer = tokenizer
        self.schema_prompt = open(schema_path, 'r', encoding='utf8').read()
        self.rationale = rationale

    def format_sample(self, item_text: str, triples: List[List[str]], rationale_prompt: str):
        if self.natlang:
            out = f"""text: {item_text}
                    \n
                    {rationale_prompt}
                    {self.natlang_triples(triples)}
                    """
        else:
            out = f"""\"\"\" {item_text} \"\"\"
                    \n
                    {rationale_prompt}
                    {self.pythonize_triples(triples)}
                    """
        return out

    def make_code_prompt(self, ICL_prompt: str, sample_text: str, triples: List[List[str]]):
        prompt = self.schema_prompt + '\n'
        rationale_prompt = self.make_rationale_prompt(triples) if self.rationale else ''
        if self.chat_model:
            text_instruct = f"""Define an instance of Extract from the text below. Only write the definition line.
                        \n{self.format_sample(sample_text, [], rationale_prompt)}"""
            if triples: 
                text_triples = self.pythonize_triples(triples)
                prompt +=  ICL_prompt + '\n' + text_instruct
                prompt = [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": text_triples},]
            else:
                prompt += ICL_prompt + '\n' + text_instruct
                prompt = [{"role": "user", "content": prompt},]
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize = False)
        else:
            text = f"""Define an instance of Extract from the text below.
                        \n{self.format_sample(sample_text, triples, rationale_prompt)}"""
            prompt += ICL_prompt + '\n' + text
        return prompt
    
    def make_natlang_prompt(self, ICL_prompt: str, sample_text: str, triples: List[List[str]]):
        prompt = ''
        rationale_prompt = self.make_rationale_prompt(triples) if self.rationale else ''
        if self.chat_model:
            text_instruct = f"""Extract a list of [entity, relation, entity] triples from the text below. Only write the definition line.
                        \n{self.format_sample(sample_text, [], rationale_prompt)}"""
            if triples: 
                text_triples = self.natlang_triples(triples)
                prompt +=  ICL_prompt + '\n' + text_instruct
                prompt = [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": text_triples},]
            else:
                prompt += ICL_prompt + '\n' + text_instruct
                prompt = [{"role": "user", "content": prompt},]
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize = False)
        else:
            text = f"""Extract a list of [entity, relation, entity] triples from the text below.
                        \n{self.format_sample(sample_text, triples, rationale_prompt)}""" # 
            prompt += ICL_prompt + '\n' + text
        return prompt

    def extract_triples(self, response: str) -> List[List[str]]:
        if self.natlang:
            # Extract triples from natural language response
            pattern = r'\["([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\]'
            matches = re.findall(pattern, response)
            return list(matches)
        else:
            # Extract triples from code-like response
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
        if self.natlang:
            rels = '\n'.join([triple[1] for triple in triple_list])
            ents = '\n'.join(['\n'.join([triple[0], triple[2]]) for triple in triple_list])
        else:
            rels = '\n'.join([f"Rel('{triple[1]}')" for triple in triple_list])
            ents = '\n'.join([f"{self.type_dict[triple[0]]}('{triple[0]}')\n{self.type_dict[triple[2]]}('{triple[2]}')" for triple in triple_list])
        rationale_prompt = f'''
        The candidate relations for this text are:
        \n
        {rels}
        The candidate entities for this text are:
        \n
        {ents}
        \n
        '''
        return rationale_prompt

    def make_icl_prompt(self, data: pd.DataFrame) -> str:
        text_list = data['text'].to_list()
        triple_list = data['triple_list'].to_list()
        prompt = ''
        
        if self.natlang:
            for text, triples in zip(text_list, triple_list):
                rationale_prompt = '' if not self.rationale else self.make_rationale_prompt(triples)
                prompt += f"""text: {text}
                \n
                {rationale_prompt}
                {self.natlang_triples(triple_list=triples)}
                \n
                """
        else:
            for text, triples in zip(text_list, triple_list):
                rationale_prompt = '' if not self.rationale else self.make_rationale_prompt(triples)
                prompt += f"""\"\"\" {text} \"\"\"
                \n
                {rationale_prompt}
                {self.pythonize_triples(triple_list=triples)}
                \n
                """
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

        for text, triple_list in tqdm(zip(text_test, triple_list_test), total=len(df_test)):
            if self.natlang:
                prompt = self.make_natlang_prompt(ICL_prompt=icl_prompt, sample_text=text, triples=[])
            else:
                prompt = self.make_code_prompt(ICL_prompt=icl_prompt, sample_text=text, triples=[])
            
            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            
            outputs = model.generate(inputs,
                                    num_return_sequences=1,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.eos_token_id,
                                    max_new_tokens = 1000,
                                    )
            input_len = inputs.shape[1]
            result = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            trues.append(triple_list)
            pred = self.extract_triples(result)
            preds.append(pred)
            logger.info('\n' + f'result: {result}' + '\n' + f'pred: {pred}')

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
            if self.natlang:
                prompt = self.make_natlang_prompt(icl_prompt, sample_text, sample_triples) + EOS_TOKEN
            else:
                prompt = self.make_code_prompt(icl_prompt, sample_text, sample_triples) + EOS_TOKEN
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

