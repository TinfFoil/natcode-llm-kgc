from typing import List, Dict

class RelationExtractionEvaluator:
    def __init__(self, mode = 'SE'):
        self.mode = mode

    def merge(self, triple_list: List):
        if self.mode == 'SE':
            return self.merge_se(triple_list)
        else:
            raise NotImplementedError

    def merge_se(self, triple_list: List[Dict]):
        merged_list = []
        for triple in triple_list:
            rel_text = triple['rel']['text']
            head_text = triple['head']['text']
            head_type = triple['head']['type']
            tail_text = triple['tail']['text']
            tail_type = triple['tail']['type']
            merged_sample = f'{rel_text}_{head_text}_{head_type}_{tail_text}_{tail_type}'
            merged_list.append(merged_sample)
        return merged_list

    def calculate_strict_micro_f1(self, trues, preds):
        TP = 0
        FP = 0
        FN = 0
        
        for true_triple_list, pred_triple_list in zip(trues, preds):
            trues_sample = self.merge(true_triple_list)
            preds_sample = self.merge(pred_triple_list)
            true_set = set(trues_sample)
            pred_set = set(preds_sample)
            print('true_set:', true_set)
            print('pred_set:', pred_set)
            print('#' * 100)
            TP += len(true_set & pred_set)
            FP += len(pred_set - true_set)
            FN += len(true_set - pred_set)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            }

def main():
    # 4 options:
    # type evaluation (TE): only the tags must match to be considered correct
    # partial evaluation (PE): requires the triples to match partially or completely, irrespective of tag, to be considered partially or completely correct
    # exact evaluation (EE): requires the triples to match exactly, irrespective of tag, to be considered correct
    # strict evalution (SE): requires both the triples and tag to match to be considered correct
    trues = [
                [
                    {"rel": {"text": "Adverse_effect"}, "head": {"text": "Gynecomastia", "type": "disease"}, "tail": {"text": "fluoresone", "type": "drug"}},
                    {"rel": {"text": "Adverse_effect"}, "head": {"text": "Gynecomastia", "type": "disease"}, "tail": {"text": "phenobarbital", "type": "drug"}},
                    {"rel": {"text": "Adverse_effect"}, "head": {"text": "Gynecomastia", "type": "disease"}, "tail": {"text": "phenytoin", "type": "drug"}}
                ],
                [
                    {"rel": {"text": "Adverse_effect"}, "head": {"text": "Ischaemia", "type": "disease"}, "tail": {"text": "diamorphine", "type": "drug"}},
                    {"rel": {"text": "Adverse_effect"}, "head": {"text": "Ischaemia", "type": "disease"}, "tail": {"text": "methylphenidate", "type": "drug"}}
                ]
            ]
    pred_dict = {'SE': 
            [
                [
                    {"rel": {"text": "Adverse_effect"}, "head": {"text": "Gynecomastia", "type": "disease"}, "tail": {"text": "fluoresone", "type": "drug"}},
                    {"rel": {"text": "Adverse_effect"}, "head": {"text": "Gynecomastia", "type": "disease"}, "tail": {"text": "phenobarbital", "type": "drug"}},
                    {"rel": {"text": "Adverse_effect"}, "head": {"text": "Gynecomastia", "type": "disease"}, "tail": {"text": "phenytoin", "type": "drug"}}
                ],
                [
                    {"rel": {"text": "Adverse_effect"}, "head": {"text": "Ischaemia", "type": "disease"}, "tail": {"text": "diamorphine", "type": "drug"}},
                    {"rel": {"text": "Adverse_effect"}, "head": {"text": "Ischaemia", "type": "disease"}, "tail": {"text": "methylphenidate", "type": "drug"}}
                ]
            ]
        }
    preds = pred_dict['SE']
    evaluator = RelationExtractionEvaluator(mode='SE')
    results = evaluator.calculate_strict_micro_f1(trues, preds)
    print(results)

if __name__ == "__main__":
    main()

