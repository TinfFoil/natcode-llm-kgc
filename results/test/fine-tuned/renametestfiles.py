import os

d = './results/test/fine-tuned'

for root, dirs, files in os.walk(d):
    for F in files:
        F_path = os.path.join(root, F)
        if '_rationale_' in F and not F.endswith('_rationale.json'):
            os.rename(F_path, F_path.replace('.json', '_rationale.json'))