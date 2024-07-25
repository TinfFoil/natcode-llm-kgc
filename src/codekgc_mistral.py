import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from typing import List
import pandas as pd
from tqdm.auto import tqdm

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

def make_prompt(schema_prompt: str, ICL_prompt: str, item_text: str):
    text_instruct = f'Define an instance of Extract from the text below. Only write the definition line.\n\"\"\" {item_text} \"\"\"'
    prompt = schema_prompt + '\n' + ICL_prompt + '\n' + text_instruct
    return prompt

def extract_triples(code_triples: str) -> List[List[str]]:
    # Regex pattern to match triples in the format Triple(person('X'), Rel('Y'), organization('Z'))
    pattern = re.compile(r"Triple\(\s*person\('([^']+)'\),\s*Rel\('([^']+)'\),\s*organization\('([^']+)'\)\)")
    
    # Find all matches in the string
    matches = pattern.findall(code_triples)
    
    # Convert matches to a list of lists
    triples = [[match[0], match[1], match[2]] for match in matches]
    
    return triples

def pythonize_triples(triple_list: List[List[str]]) -> str:
    # out = ''
    # for triple_list in data:
    pythonic_triples = "extract = Extract(["
    for i, triple in enumerate(triple_list):
        person, relation, organization = triple
        pythonic_triples += f"Triple(person('{person}'), Rel('{relation}'), organization('{organization}')"
        
        if i < len(triple_list) - 1:
            pythonic_triples += "), "
        else:
            pythonic_triples += ")])"
        # out += pythonic_triples + '\n'
    return pythonic_triples

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

def make_icl_prompt(data) -> str:
    '''
    input should be a Pandas DataFrame
    '''
    text_list = data['text'].to_list()
    triple_list = data['triple_list'].to_list()
    prompt = ''
    for text, triples in zip(text_list, triple_list):
        prompt += f"\"\"\" {text} \"\"\"\n {pythonize_triples(triple_list=triples)}\n"
    return prompt

def evaluate(model, tokenizer, df_test, schema_prompt, train_json, n_icl_samples):
    trues = []
    preds = []
    
    text_test = df_test['text'].to_list()
    triple_list_test = df_test['triple_list'].to_list()

    for text, triple_list in tqdm(zip(text_test, triple_list_test), total=len(df_test)):
        df_train = pd.read_json(train_json).sample(n=n_icl_samples)
        icl_prompt = make_icl_prompt(df_train)

        prompt = make_prompt(schema_prompt=schema_prompt, ICL_prompt=icl_prompt, item_text=text)
        
        # messages = [
        #     {"role": "user", "content": prompt}
        # ]

        # inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
    
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        
        out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)

        result = tokenizer.decode(out_tokens[0])
        
        # outputs = model.generate(inputs,
        #                         num_return_sequences=1,
        #                         eos_token_id=tokenizer.eos_token_id,
        #                         pad_token_id=tokenizer.eos_token_id,
        #                         max_new_tokens = 1000,
        #                         )
        # result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        print(result)

        trues.append(triple_list)
        preds.append(extract_triples(result))

    precision, recall, f1_score = calculate_micro_f1(trues, preds)
    return precision, recall, f1_score

def main():
    model_name = "NTQAI/Nxcode-CQ-7B-orpo"
    # model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    model_name_simple = model_name.split('/')[-1]   
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
 
    tokenizer = MistralTokenizer.v1()

    mistral_models_path = "./models/mistral-7B-Instruct-v0.3"
    model = Transformer.from_folder(mistral_models_path)
    
    train_json = './data/codekgc-data/ade/train_triples.json'
    test_json = './data/codekgc-data/ade/test_triples_300.json'

    schema_prompt = open('./data/schema_prompt', 'r', encoding='utf8').read()

    df_test = pd.read_json(test_json)
    n_samples_test = len(df_test)
    df_test = df_test.sample(n=n_samples_test)
    n_icl_samples = 15

    precision, recall, f1_score = evaluate(model, tokenizer, df_test, schema_prompt, train_json, n_icl_samples)

    info = {"Model": model_name,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1_score,
            "n_icl_samples": n_icl_samples,
            "n_samples_test": n_samples_test,
            }
    
    print(info)
    
    json_path = f'./results_{model_name_simple}.json'
    
    with open(json_path, 'w', encoding='utf8') as f:
        json.dump(info, f, ensure_ascii = False)

if __name__ == "__main__":
    main()