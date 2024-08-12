import os
import json

results_dir = './results'

for root, dirs, files in os.walk(results_dir):
    for F in files:
        file_path = os.path.join(root, F)
        
        with open(file_path, 'r', encoding='utf8') as f:
            data = json.load(f)
        f1_score = round(data[0]['F1_Score'], 2)
        if f1_score < 0.05:
            print(file_path)
            print(f1_score)
            os.rename(file_path, file_path.replace('.json', f'_{f1_score}.json'))