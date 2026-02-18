from typing import List, Dict, Union
import argparse
import json
import os

class RelationExtractionEvaluator:
    def __init__(self, mode='SE'):
        self.mode = mode

    def overload_input_structure(fn):
        def wrapper(self, samples: Union[Dict, List], preds: List = None):
            if preds is None:
                trues = [el['true'] for el in samples]
                preds = [el['pred'] for el in samples]
            else:
                trues = samples
            return fn(self, trues, preds)
        return wrapper

    # ---------- NEW: parse partial output.log ----------
    @staticmethod
    def parse_output_log(
        log_path: str,
        require_keys: tuple = ("true", "pred"),
        keep_fields: tuple = ("text", "true", "pred"),
    ) -> List[Dict]:
        """
        Extracts JSON objects from a (possibly truncated) output.log and returns
        a list of dicts containing at least keys in `require_keys`.

        Robustness:
        - Ignores non-JSON python dict prints (single quotes) and other noise.
        - Skips incomplete trailing JSON object (e.g., killed mid-print).
        """
        samples: List[Dict] = []

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            buf_lines: List[str] = []
            depth = 0
            in_json = False

            for line in f:
                # Start capturing on a "{" at any indentation level.
                if not in_json:
                    stripped = line.lstrip()
                    if stripped.startswith("{"):
                        in_json = True
                        buf_lines = [line]
                        # Initialize depth by counting braces on the start line.
                        depth = line.count("{") - line.count("}")
                    continue

                # If we're in a candidate JSON object, accumulate.
                buf_lines.append(line)
                depth += line.count("{") - line.count("}")

                # Object closed
                if depth == 0:
                    raw = "".join(buf_lines).strip()
                    in_json = False
                    buf_lines = []

                    # Try strict JSON parse. If it fails, ignore.
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue

                    # Filter: must contain required keys
                    if not isinstance(obj, dict):
                        continue
                    if not all(k in obj for k in require_keys):
                        continue

                    # Keep only requested fields (optional; avoids bloating memory)
                    kept = {k: obj.get(k) for k in keep_fields if k in obj}
                    # Ensure structure is as expected
                    if isinstance(kept.get("true"), list) and isinstance(kept.get("pred"), list):
                        samples.append(kept)

            # If file ended mid-JSON (OOM kill), we intentionally drop the partial buffer.
        return samples
    # --------------------------------------------------

    def merge(self, triple_list: List):
        if self.mode == 'SE':
            return self.merge_se(triple_list)
        if self.mode == 'EE':
            return self.merge_ee(triple_list)
        if self.mode == 'TE':
            return self.merge_te(triple_list)
        else:
            raise NotImplementedError

    def merge_se(self, triple_list: List[Dict]):
        merged_list = []
        for triple in triple_list:
            if triple is not None:
                head_type = (triple.get('head', {}) or {'type': ''}).get('type', '')
                tail_type = (triple.get('tail', {}) or {'type': ''}).get('type', '')
                rel_text = (triple.get('rel', {}) or {'type': ''}).get('type', '')
                head_text = (triple.get('head', {}) or {'text': ''}).get('text', '')
                tail_text = (triple.get('tail', {}) or {'text': ''}).get('text', '')
                merged_sample = f'{rel_text}_{head_text}_{head_type}_{tail_text}_{tail_type}'
                merged_list.append(merged_sample)
        return merged_list

    def merge_te(self, triple_list: List[Dict]):
        merged_list = []
        for triple in triple_list:
            if triple is not None:
                head_type = (triple.get('head', {}) or {'type': ''}).get('type', '')
                tail_type = (triple.get('tail', {}) or {'type': ''}).get('type', '')
                merged_sample = f'{head_type}_{tail_type}'
                merged_list.append(merged_sample)
        return merged_list

    def merge_ee(self, triple_list: List[Dict]):
        merged_list = []
        for triple in triple_list:
            if triple is not None:
                rel_text = self.extract_triple_element(triple, 'rel', 'type')
                head_text = self.extract_triple_element(triple, 'head', 'text')
                tail_text = self.extract_triple_element(triple, 'tail', 'text')
                merged_sample = f'{rel_text}_{head_text}_{tail_text}'
                merged_list.append(merged_sample)
        return merged_list

    @staticmethod
    def extract_triple_element(triple: dict, triple_type: str = 'head', field_name: str = 'text'):
        triple_type_value = triple.get(triple_type, {})
        if isinstance(triple_type_value, dict):
            return triple_type_value.get(field_name, '')
        else:
            return ''

    def get_results(self, results_dir_path: str):
        preds_test_list = []
        for root, dirs, files in os.walk(results_dir_path):
            for D in dirs:
                dirpath = os.path.join(root, D)
                if 'config.json' in os.listdir(dirpath):
                    config_path = os.path.join(dirpath, 'config.json')
                    try:
                        config = json.load(open(config_path, 'r'))
                        preds_test_path = os.path.join(dirpath, 'preds_test.json')
                        preds_test = json.load(open(preds_test_path, 'r'))
                    except FileNotFoundError:
                        continue
                    preds_test_list.append({'preds': preds_test, 'config': config})
        return preds_test_list

    @overload_input_structure
    def calculate_strict_micro_f1(self, trues, preds):
        TP = 0
        FP = 0
        FN = 0

        for true_triple_list, pred_triple_list in zip(trues, preds):
            trues_sample = self.merge(true_triple_list)
            preds_sample = self.merge(pred_triple_list)
            true_set = set(trues_sample)
            pred_set = set(preds_sample)
            TP += len(true_set & pred_set)
            FP += len(pred_set - true_set)
            FN += len(true_set - pred_set)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

    def collect_metrics(self, results: List[Dict], config: List[Dict]):
        results_list = []
        for res, conf in zip(results, config):
            precision = res['precision']
            recall = res['recall']
            f1_score = res['f1_score']
            results_list.append(
                {
                    'seed': conf['seed'],
                    'model_name': conf['model_name'],
                    'dataset': conf['dataset'],
                    'train_steps': conf['train_steps'],
                    'n_icl_samples': conf['n_icl_samples'],
                    'results_dir': conf['results_dir'],
                    'desc_schema': conf.get('desc_schema', 0),
                    'lora_modules': ''.join([el.replace('_proj', '') for el in conf['lora_modules']]),
                    'do_train': conf['do_train'],
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                }
            )
        return results_list


def main(args):
    """
    Evaluation modes:
      TE: only tags must match
      EE: triples must match exactly (ignoring tag differences vs SE)
      SE: triples and tags must match
    """
    evaluator = RelationExtractionEvaluator(mode=args.m)

    if args.i:
        pair_list = json.load(open(args.i, 'r'))
    elif args.log:
        pair_list = evaluator.parse_output_log(args.log)
    else:
        pair_list = [
            {
                'true': [
                    {"rel": {"text": "Adverse_effect"}, "head": {"text": "Gynecomastia", "type": "disease"},
                     "tail": {"text": "fluoresone", "type": "drug"}},
                    {"rel": {"text": "Adverse_effect"}, "head": {"text": "Gynecomastia", "type": "disease"},
                     "tail": {"text": "phenobarbital", "type": "drug"}},
                    {"rel": {"text": "Adverse_effect"}, "head": {"text": "Gynecomastia", "type": "disease"},
                     "tail": {"text": "phenytoin", "type": "drug"}}
                ],
                'pred': [
                    {"rel": {"text": "Adverse_effect"}, "head": {"text": "Gynecomastia", "type": "disease"},
                     "tail": {"text": "fluoresone", "type": "drug"}},
                    {"rel": {"text": "Adverse_effect"}, "head": {"text": "Gynecomastia", "type": "disease"},
                     "tail": {"text": "phenobarbital", "type": "drug"}},
                    {"rel": {"text": "Adverse_effect"}, "head": {"text": "Gynecomastia", "type": "disease"},
                     "tail": {"text": "phenytoin", "type": "drug"}}
                ]
            }
        ]

    results = evaluator.calculate_strict_micro_f1(pair_list)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relation extraction evaluator")
    parser.add_argument("-i", help="Predictions JSON file (list of {true,pred} dicts)", default="")
    parser.add_argument("--log", help="output.log path to parse partial predictions from", default="")
    parser.add_argument("-m", help="Evaluation mode", default='SE')
    args = parser.parse_args()
    main(args)
