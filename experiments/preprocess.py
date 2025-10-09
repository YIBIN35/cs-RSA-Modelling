import pandas as pd
import json



def shuffle_no_adjacent(df, key='noun', random_state=42):
    remaining = df.copy()
    result = []

    prev_noun = None
    while not remaining.empty:
        # candidates that don't share the same noun as previous
        candidates = remaining[remaining[key] != prev_noun]
        row = candidates.sample(1, random_state=random_state)
        result.append(row)
        prev_noun = row.iloc[0][key]
        remaining = remaining.drop(row.index)
    return pd.concat(result, ignore_index=True)

df1 = pd.read_csv('./parker_stimuli.csv')
df2 = pd.read_excel('./Parker_Modifiers_Trials_May16.xlsx')

df_exptrial = df2[df2['type'] == 'exp']
df_exptrial_singleton = df2[df2['type'] == 'exp']

df_singleton_marked = df_exptrial[(df_exptrial['group']=='single') & (df_exptrial['state']=='a')]
df_singleton_unmarked = df_exptrial[(df_exptrial['group']=='single') & (df_exptrial['state']=='b')]

df_barenoun_main = pd.concat([df_singleton_marked[['noun','image1','state','adj']], df_singleton_unmarked[['noun','image1','state','adj']]])
df_barenoun_main = df_barenoun_main.rename(columns={'image1': 'image'})

filler1 = df_singleton_marked[['noun','image2']].rename(columns={'image2': 'image'})
filler2 = df_singleton_marked[['noun','image3']].rename(columns={'image3': 'image'})
filler3 = df_singleton_marked[['noun','image4']].rename(columns={'image4': 'image'})
df_barenoun_filler = pd.concat([filler1,filler2,filler3])

df_barenoun = pd.concat([df_barenoun_main, df_barenoun_filler]).reset_index(drop=True)

df_shuffled_1 = shuffle_no_adjacent(df_barenoun, random_state=1)
df_shuffled_2 = shuffle_no_adjacent(df_barenoun, random_state=2)
df_shuffled_3 = shuffle_no_adjacent(df_barenoun, random_state=3)

for random_number in [1,2,3]:
    df_shuffled = shuffle_no_adjacent(df_barenoun, random_state=random_number)

    # check whether it is a correct shuffle
    assert set(map(tuple, df_shuffled.to_numpy())) == set(map(tuple, df_barenoun.to_numpy()))

    df_shuffled_reversed = df_shuffled.iloc[::-1].reset_index(drop=True)


data = [
    {"label": row["image"], "item": [row["noun"]]}
    for _, row in df_shuffled.iterrows()
]

with open("output.json", "w") as f:
    json.dump(data, f, indent=4)










# label is the name of image, item is the word.
