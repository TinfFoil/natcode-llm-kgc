import re
import os
import pandas as pd

split = 'test'
model_type = 'fine-tuned'
results_dir = f'./results/{split}/{model_type}'
model_type_mod = 1 if model_type == 'fine-tuned' else 0
df = pd.DataFrame()

training_steps = 250

for root, dirs, files in os.walk(results_dir):
    filename_list = []
    for F in files:
        if f'steps={training_steps}' in F:
            file_path = os.path.join(root, F)
            df_file = pd.read_json(file_path)
            # df_file['index'] = range(len(df_file))
            # df_file.set_index('index')
            # display(df_file)
            df = pd.concat([df, df_file])
            filename_list.append(F.replace('.json', ''))
    df['Model'] = filename_list
print(df)
df['Dataset'] = df['Model'].apply(lambda x: x.split('_')[1 + model_type_mod])
df['Language'] = df['Model'].apply(lambda x: x.split('_')[2 + model_type_mod]).apply(lambda x: 'Code' if x == 'code' else 'Natural')
df['Rationale'] = df['Model'].apply(lambda x: x.split('_')[3 + model_type_mod]).apply(lambda x: 'No' if x == 'base' else 'Yes')
df['Model'] = df['Model'].apply(lambda x: x.split('_')[0])
# df.drop(['Model'], axis=1, inplace=True)
# cols = df.columns.tolist()
# cols = [cols[-1]] + cols[:-1]
# df = df[cols]
df = df.rename(columns={'F1_Score': 'F1'})
df.sort_values(by=['Model', 'Dataset', 'Language', 'Rationale'], inplace=True, )
df = df.round(3)
df.reset_index(inplace=True)
df.drop(['index', 'date', 'n_icl_samples', 'n_samples_test', 'split', 'schema_path', 'dataset', 'rationale', 'natlang', 'chat_model', 'fine-tuned'], axis = 1, inplace=True)
print(df)

paper_dir = '/home/pgajo/DeepKE/example/llm/CodeKGC/paper'

table_path = os.path.join(paper_dir, f'results_{split}_{model_type}_steps={training_steps}.tex')
df.to_latex(table_path, float_format=lambda x: f'{x:.3f}', index=False)