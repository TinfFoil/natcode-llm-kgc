import os
import json
import numpy as np

results_dir = './results/test/fine-tuned'

steps = [50, 100, 150, 200, 250, 500]
model_name = 'Mistral-7B-Instruct-v0.3'
f1_dict = {}
for step in steps:
    F1_scores = []
    for root, dirs, files in os.walk(results_dir):
        for F in files:
            if F.endswith('.json') and model_name in F and f'steps={step}' in F:
                json_path = os.path.join(root, F)
                
                with open(json_path, 'r', encoding='utf8') as f:
                    data = json.load(f)
                F1_scores.append(np.mean([line['F1_Score'] for line in data]))
    f1_dict[step] = F1_scores
print(f1_dict)