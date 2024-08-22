import os
import json
import re
resdir = './results/test/fine-tuned'
resdirnew = './results/test/fine-tuned-fixed'

for root, dirs, files in os.walk(resdir):
    files = sorted(files)#, key=lambda x: int(re.search(r'steps=(\d+)', x).group(1)))
    for F in files:
        F_no_extension = F.replace('.json', '')
        F_rationale = F_no_extension + '_rationale.json'
        if F_rationale in files:
            F_rationale_path = os.path.join(root, F_rationale)
            F_path = os.path.join(root, F)
            with open(F_rationale_path, 'r', encoding='utf8') as f:
                data_rationale = json.load(f)
            with open(F_path, 'r', encoding='utf8') as f:
                data = json.load(f)
            data_new = data_rationale + data
        
            json_path = os.path.join(resdirnew, F_rationale)
            
            with open(json_path, 'w', encoding='utf8') as f:
                json.dump(data_new, f, ensure_ascii = False)
            new_rationale_path = os.path.join('./results/old', os.path.basename(F_rationale_path))
            new_path = os.path.join('./results/old', os.path.basename(F_path))
            os.rename(F_rationale_path, new_rationale_path)
            os.rename(F_path, new_path)
