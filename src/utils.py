from typing import List
from tqdm.auto import tqdm
import pandas as pd
import json
import re

class Prompter:
    def __init__(self,
                 type_dict,
                 natlang
                 ) -> None:
        self.type_dict = type_dict
        self.natlang = natlang

    def make_prompt(self, schema_prompt: str, ICL_prompt: str, item_text: str):
        text_instruct = f'Define an instance of Extract from the text below. Only write the definition line.\n\"\"\" {item_text} \"\"\"'
        prompt = schema_prompt + '\n' + ICL_prompt + '\n' + text_instruct
        return prompt

    def extract_triples(self, code_triples: str) -> List[List[str]]:
        # Regex pattern to match triples in the format Triple(any alphanumerical head('X'), Rel('Y'), any alphanumerical tail('Z'))
        pattern = re.compile(r"Triple\(\s*(\w+)\('([^']+)'\),\s*Rel\('([^']+)'\),\s*(\w+)\('([^']+)'\)\)")
        
        # Find all matches in the string
        matches = pattern.findall(code_triples)
        
        # Convert matches to a list of lists
        triples = [[match[1], match[2], match[4]] for match in matches]
        
        return triples

    def pythonize_triples(self, triple_list: List[List[str]]) -> str:
        # out = ''
        # for triple_list in data:
        pythonic_triples = "extract = Extract(["
        for i, triple in enumerate(triple_list):
            head, edge, tail = triple
            pythonic_triples += f"Triple({self.type_dict[head]}('{head}'), Rel('{edge}'), {self.type_dict[tail]}('{tail}')"
            
            if i < len(triple_list) - 1:
                pythonic_triples += "), "
            else:
                pythonic_triples += ")])"
            # out += pythonic_triples + '\n'
        return pythonic_triples

    def natlang_triples(self, triple_list: List[List[str]]) -> str:
        natlang_triples = "["
        for i, triple in enumerate(triple_list):
            head, edge, tail = triple
            natlang_triples += f"['{head}', '{edge}', '{tail}']"
            
            if i < len(triple_list) - 1:
                natlang_triples += ", "
            else:
                natlang_triples += "]"
        return natlang_triples
    
    def make_icl_prompt(self, data: pd.DataFrame) -> str:
        text_list = data['text'].to_list()
        triple_list = data['triple_list'].to_list()
        prompt = ''
        if self.natlang:
            raise NotImplementedError
        else:
            for text, triples in zip(text_list, triple_list):
                prompt += f"\"\"\" {text} \"\"\"\n {self.pythonize_triples(triple_list=triples)}\n"
        return prompt

def calculate_micro_f1(trues: List[List[List[str]]], preds: List[List[List[str]]]) -> List[float]:
    TP = 0
    FP = 0
    FN = 0
    
    for true_triples, pred_triples in zip(trues, preds):
        trues_merged = ['-'.join(trues) for trues in true_triples]
        preds_merged = ['-'.join(preds) for preds in pred_triples]
        # print('trues:', trues_merged, 'preds:', preds_merged)
        true_set = set(trues_merged)
        pred_set = set(preds_merged)
        
        TP += len(true_set & pred_set)
        FP += len(pred_set - true_set)
        FN += len(true_set - pred_set)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return [precision, recall, f1_score]

def evaluate(model, tokenizer, df_test, schema_prompt, icl_prompt, prompter, chat_model = True):
    trues = []
    preds = []
    
    text_test = df_test['text'].to_list()
    triple_list_test = df_test['triple_list'].to_list()

    for text, triple_list in tqdm(zip(text_test, triple_list_test), total=len(df_test)):
        prompt = prompter.make_prompt(schema_prompt=schema_prompt, ICL_prompt=icl_prompt, item_text=text)
        
        if chat_model:
            messages = [
                {"role": "user", "content": prompt}
            ]

            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        
        outputs = model.generate(inputs,
                                # num_return_sequences=1,
                                # eos_token_id=tokenizer.eos_token_id,
                                # pad_token_id=tokenizer.eos_token_id,
                                # max_new_tokens = 1000,
                                )
        result = tokenizer.decode(outputs[0])#[len(inputs):], skip_special_tokens=True)
        print(result)

        trues.append(triple_list)
        preds.append(prompter.extract_triples(result))

    precision, recall, f1_score = calculate_micro_f1(trues, preds)
    return precision, recall, f1_score

def save_json(info, json_path):
    try:
        with open(json_path, 'r', encoding='utf8') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    
    data.extend(info)

    with open(json_path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii = False)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""