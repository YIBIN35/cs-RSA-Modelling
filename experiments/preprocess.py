import pandas as pd

df1 = pd.read_csv('./parker_stimuli.csv')
df2 = pd.read_excel('./Parker_Modifiers_Trials_May16.xlsx')

df_exptrial = df2[df2['type'] == 'exp']


