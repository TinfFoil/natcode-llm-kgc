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
            if not os.path.exists(verbose_output_path):
                os.makedirs(verbose_output_path)
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
    
    def run_model(self, texts, icl_prompt):
        """
        Unified method for both single-sample and batch generation.
        If `texts` is a string, convert it to a list of length 1. 
        Then handle batch generation in one go.
        """
        # 1) If the user passes a single text as a string, convert to list
        if isinstance(texts, str):
            texts = [texts]

        # 2) Construct prompts
        if self.natlang:
            prompts = [self.make_natlang_prompt(icl_prompt, txt, []) for txt in texts]
        else:
            prompts = [self.make_code_prompt(icl_prompt, txt, []) for txt in texts]

        # 3) Tokenize as a batch
        tokenized = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            return_token_type_ids=False,
        ).to(self.model.device)
        
        # 4) Generate outputs in batch
        with torch.no_grad():
            outputs = self.model.generate(
                **tokenized,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=1000
            )

        # 5) Decode each output
        input_lengths = (tokenized["input_ids"].ne(self.tokenizer.pad_token_id).sum(dim=1))
        decoded = []
        for i, seq in enumerate(outputs):
            # Slice out prompt tokens, then decode generation
            in_len = input_lengths[i]
            gen_tokens = seq[in_len:]
            # print(gen_tokens)
            decoded_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            decoded.append(decoded_text)

        # 6) If single text was provided, return just the string; else return a list
        return decoded[0] if len(decoded) == 1 else decoded

    def evaluate(self, df_test, df_train, batch_size=1):
        """
        Evaluate the model on the test set using a configurable `batch_size`.
        This uses a unified run_model(...) method that can handle both
        single-item or multi-item batches.
        """
        trues = []
        preds = []

        text_test = df_test['text'].to_list()
        triple_list_test = df_test['triple_list'].to_list()

        # Gather all possible relation types for eventual per-relation metrics
        all_true_rel_types = set([el[1] for t_list in triple_list_test for el in t_list])
        rel_type_preds_trues = {k: {'trues': [], 'preds': []} for k in all_true_rel_types}

        # Progress bar over sub-batches, rather than item by item
        pbar = tqdm(range(0, len(df_test), batch_size), desc=f'Model: {self.model_name}')
        self.t_counter = 0  # initialize model timeout counter

        for start_idx in pbar:
            end_idx = min(start_idx + batch_size, len(df_test))
            batch_texts = text_test[start_idx:end_idx]
            batch_triples = triple_list_test[start_idx:end_idx]

            # Run the model on the batch. If batch_size=1, this will still work.
            icl_prompt = self.make_icl_prompt(df_train)
            results = self.run_model(batch_texts, icl_prompt)

            # If run_model returned a single string (when batch_size=1), make it a list
            if isinstance(results, str):
                results = [results]

            # Now iterate over each sample in the batch
            for text, true_triple_list, model_output in zip(batch_texts, batch_triples, results):
                pred_list = self.extract_triples(model_output)
                trues.append(true_triple_list)
                preds.append(pred_list)

                # Collect per-relation predictions + gold
                for rel_type in all_true_rel_types:
                    trues_type = [t for t in true_triple_list if t[1] == rel_type]
                    preds_type = [p for p in pred_list if p[1] == rel_type]
                    rel_type_preds_trues[rel_type]['trues'].append(trues_type)
                    rel_type_preds_trues[rel_type]['preds'].append(preds_type)

                # Optionally display partial F1 and log verbose output
                if self.verbose_test:
                    metrics_sample = self.calculate_strict_micro_f1([true_triple_list], [pred_list])
                    metrics_current = self.calculate_strict_micro_f1(trues, preds)
                    pbar.set_description(f'F1: {round(metrics_current[-1], 2)}')

                    output_texts = (
                        '\n' + f'text: {text}' +
                        '\n' + f'result: {model_output}' +
                        '\n' + f'pred: {pred_list}' +
                        '\n' + f'trues: {true_triple_list}'
                    )
                    output_metrics = (
                        '\n' + f'metrics_sample: {metrics_sample}' +
                        '\n' + f'metrics_current: {metrics_current}'
                    )
                    print(output_texts)
                    print(output_metrics)
                    if self.verbose_output_path:
                        self.verbose_output += output_texts + output_metrics

        # If verbose output path is specified, save the accumulated logs
        if self.verbose_output:
            self.verbose_output = self.verbose_output.lstrip()
            with open(self.verbose_output_path, 'w', encoding='utf8') as f:
                f.write(self.verbose_output)
        all_pred_rel_types = set([p[1] for p_list in preds for p in p_list])
        print(f'All types of predicted relations: {all_pred_rel_types}')
        # Final strict micro-averaged metrics
        precision, recall, f1_score = self.calculate_strict_micro_f1(trues, preds)

        # Per-relation metrics
        rel_type_metrics = {k: {} for k in all_true_rel_types}
        for rel_type, pr_dict in rel_type_preds_trues.items():
            p_rel, r_rel, f1_rel = self.calculate_strict_micro_f1(pr_dict['trues'], pr_dict['preds'])
            rel_type_metrics[rel_type]['precision'] = p_rel
            rel_type_metrics[rel_type]['recall']    = r_rel
            rel_type_metrics[rel_type]['f1']        = f1_rel

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1_score,
            'rel_type_metrics': rel_type_metrics
        }


    def make_samples(self, tokenizer, df: pd.DataFrame, n_icl_samples: int) -> List[str]:
        text_list = []
        EOS_TOKEN = tokenizer.eos_token # add EOS_TOKEN to prevent infinite generation
        max_prompt_len = 0
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
            # if prompt_token_len > self.model.max_seq_length:
            #     raise Exception(f'The prompt is longer than the model\'s maximum sequence length ({self.model.max_seq_length}).')
            if prompt_token_len > self.max_seq_len:
                warnings.warn(f'The prompt is longer than the set maximum sequence length ({self.max_seq_len})')
            if prompt_token_len > max_prompt_len:
                max_prompt_len = prompt_token_len
            text_list.append(prompt)
        print('max prompt length:', max_prompt_len)
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

    def heatmap2d(
        self,
        attention_scores,
        token_ids_full,
        idx,
        width=10,         # number of columns in the grid
        height=None,      # maximum number of rows (if provided)
        cell_w=3,       # width of each cell (in figure units)
        cell_h=0.75,       # height of each cell (in figure units)
        cmap='viridis',
        norm='log',
        high_threshold=0.999,
        savename='',
        format='pdf',
        text_mode='truncate',   # 'wrap' or 'truncate'
        wrap_width=10,      # maximum characters per line if wrapping
        truncate_length=5  # maximum total characters if truncating
    ):
        """
        Plots a 2D heatmap of attention scores with token labels in each cell.
        Allows customizing the cell's width and height separately (rather than only squares).
        Also supports wrapping or truncating text within the cells.

        Args:
            attention_scores: 1D array (num_tokens,) of attention values.
            token_ids_full:   The list/array of token IDs.
            idx:              Index or identifier used in the filename.
            width:            Number of columns in the grid.
            height:           Maximum number of rows in the grid (if None, computed automatically).
            cell_w:           Width of each cell in figure coordinates.
            cell_h:           Height of each cell in figure coordinates.
            cmap:             Matplotlib colormap.
            norm:             Normalization type, 'log' or 'linear'.
            high_threshold:   Quantile above which cells will be colored black.
            savename:         Directory or prefix to save the output file.
            format:           Output format for saving the figure (e.g., 'pdf', 'png').
            text_mode:        Either 'wrap' (to wrap text) or 'truncate' (to truncate text).
            wrap_width:       Maximum characters per line if text_mode is 'wrap'.
            truncate_length:  Maximum total characters if text_mode is 'truncate'.
        """
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
        
        # Create the figure and axis using cell dimensions
        fig_width = n_cols * cell_w
        fig_height = n_rows * cell_h
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Choose the normalization
        if norm == 'log':
            color_norm = LogNorm(vmin=np.nanmin(attention_scores), vmax=np.nanmax(attention_scores))
        else:
            color_norm = Normalize(vmin=np.nanmin(attention_scores), vmax=np.nanmax(attention_scores))
        
        # Create the heatmap with non-square cells
        im = ax.imshow(attention_scores_2d, cmap=cmap, aspect='auto', norm=color_norm)
        
        # Apply black color to high values
        high_mask = attention_scores_2d > np.nanquantile(attention_scores, high_threshold)
        im.cmap.set_bad(color='black')
        attention_scores_2d_masked = np.ma.masked_where(high_mask, attention_scores_2d)
        im.set_data(attention_scores_2d_masked)
        
        # Helper function to process text based on mode
        def process_text(text):
            if text_mode == 'wrap':
                return "\n".join(textwrap.wrap(text, width=wrap_width))
            elif text_mode == 'truncate':
                return text[:truncate_length] + "..." if len(text) > truncate_length else text
            else:
                return text
        
        # Add labels to each cell
        for i in range(n_rows):
            for j in range(n_cols):
                text = labels_2d[i, j]
                score = attention_scores_2d[i, j]
                if not np.isnan(score):
                    processed_text = process_text(text)
                    ax.text(j, i, processed_text, ha='center', va='center', fontsize=30, color='white', fontweight='bold')
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, aspect=20)
        cbar.ax.tick_params(labelsize=26)
        
        plt.tight_layout()
        save_dir = f"./paper/attn/{savename}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'attn_{idx}_{heatmap_token}.{format}'), format=format, bbox_inches='tight')
        plt.close()

