import pandas as pd

df = pd.read_csv('./experiments/modeling_input_words.csv', index_col=None)
df_norming_results = pd.read_csv('./experiments/norming_results.csv', index_col=None)

df_combined = df.merge(df_norming_results, left_on=['natural_utterance', 'image'], right_on=['utterance', 'object'], how='left')


