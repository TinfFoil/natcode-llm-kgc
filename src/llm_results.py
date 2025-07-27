import os
from calculate_metrics import RelationExtractionEvaluator
import pandas as pd
# dir_results = '122140'
dir_results = '122356'
dir_path = os.path.join('./results', dir_results)
evaluator = RelationExtractionEvaluator('EE')
preds_list = evaluator.get_results(dir_path)
pred_files = [el['preds'] for el in preds_list]
config_files = [el['config'] for el in preds_list]
results_list = []
for pred_file, config_file in zip(pred_files, config_files):
    trues = [el['true'] for el in pred_file]
    outs = [el['output'] for el in pred_file]
    preds = [evaluator.extract_triples(el, config_file) for el in outs]
    results_list.append(evaluator.calculate_strict_micro_f1(trues, preds))
metrics = evaluator.collect_metrics(results_list, config_files)
df = pd.DataFrame(metrics)
print(df)
df = df[df['do_train'] == 1]
df_grouped = df.groupby(by=['dataset', 'n_icl_samples']).aggregate(['mean', 'std'])
df_grouped = df_grouped.reset_index()
df_grouped['score'] = df_grouped.apply(lambda x: f"{round(x[('f1_score', 'mean')], 3):.3f}" + r'$\pm$ \tiny{' f"{round(x[('f1_score', 'std')], 3):.3f}" + r'}', axis = 1)
df_grouped_pivot = df_grouped.drop(['f1_score', 'seed', 'do_train'], axis = 1)
print(df_grouped_pivot)
df_grouped_pivot = df_grouped_pivot.pivot(
    columns=['dataset'],
    index=['n_icl_samples'],
    values=['score']
)
print(df_grouped_pivot)
df_grouped_pivot.to_latex(
    f'./paper/{dir_results}.tex',
    index=True,
    float_format = "%.3f",
    label = f"tab:llm-results",
    caption = "Micro-F1 (SemDP) and LAS (SynDP) for the labeled edge prediction task, using Mistral-7B-Instruct-v0.3. Best in bold.",
    # escape = 1,
)