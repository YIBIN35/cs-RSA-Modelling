import pandas as pd
import json
import os
import re
import numpy as np
import random
import itertools

def copy_target_images():
    src_dir = "images"
    dest_dir = "openAI_images"
    os.makedirs(dest_dir, exist_ok=True)

    for prefix in results['$images_1'].str.split('_').str[0].unique():
        for file in glob.glob(os.path.join(src_dir, f"{prefix}_*")):
            shutil.copy(file, dest_dir)

def nouns_without_overspec():
    df = pd.read_csv("./Modifiers_Data_v3.csv")

    df.columns = ['Row Labels', 'A_singleton', 'A_pair', 'A_total', 'B_singleton', 'B_pair', 'B_total', 'grand_total', 'check']
    df = df.drop(index=range(448,452)) # drop the grand total and so on at the end.
    df = df.drop(index=range(0,4)) # drop the first couple of rows
    df = df.reset_index(drop=True)

    for i in range(len(df) - 1):  # avoid last row overflow
        val = str(df.loc[i, "Row Labels"]).strip()
        if re.fullmatch(r"\d+", val):
            next_label = str(df.loc[i + 1, "Row Labels"])
            noun = next_label.split("_")[0]
            df.loc[i, "Row Labels"] = 'summary ' + noun

    summary_df = df[df['Row Labels'].str.startswith('summary')].copy()
    summary_df['noun'] = summary_df['Row Labels'].str.split(' ').str[1]

    result_nouns = summary_df[summary_df['grand_total'].isna()]['noun'].tolist()
    return summary_df

def shuffle_no_adjacent(df, key="noun", random_state=33):
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

def _pick_two_distractor_rows(group):
    d = group[group['state'].isna()]
    uvals = d['utterance'].unique()
    ivals = d['image'].unique()

    u_pairs = list(itertools.combinations(uvals, 2))
    i_pairs = list(itertools.combinations(ivals, 2))

    # 3C2 × 3C2 = 9 combos
    combos = [((u1, i1), (u2, i2))
              for (u1, u2) in u_pairs
              for (i1, i2) in i_pairs]

    # (u1, i1), (u2, i2) = rng.choice(combos)

    # r1 = d[(d['utterance'] == u1) & (d['image'] == i1)]
    # r2 = d[(d['utterance'] == u2) & (d['image'] == i2)]

    # return pd.concat([r1, r2], ignore_index=False)

    # return all row-pair DataFrames
    result = []
    for (u1, i1), (u2, i2) in combos:
        r1 = d[(d['utterance'] == u1) & (d['image'] == i1)]
        r2 = d[(d['utterance'] == u2) & (d['image'] == i2)]
        result.append(pd.concat([r1, r2], ignore_index=False))
    return result

def pick_normed_per_noun(df, seed=None):
    # rng = random.Random(seed)
    out = []
    for noun, grp in df.groupby('noun', sort=False):
        non_distractor = grp[grp['state'].notna()]
        distractors = _pick_two_distractor_rows(grp)
        distractor = random.choice(distractors)
        out.append(pd.concat([non_distractor, distractor], ignore_index=False))
    return pd.concat(out, ignore_index=True)

df1 = pd.read_csv("./parker_stimuli.csv")
df2 = pd.read_excel("./Parker_Modifiers_Trials_May16.xlsx")

df_exptrial = df2[df2["type"] == "exp"]
df_exptrial_singleton = df2[(df2["type"] == "exp") & (df2["group"] == "single")]

df_singleton_marked = df_exptrial[
    (df_exptrial["group"] == "single") & (df_exptrial["state"] == "a")
]
df_singleton_unmarked = df_exptrial[
    (df_exptrial["group"] == "single") & (df_exptrial["state"] == "b")
]

removed_nouns = nouns_without_overspec()

######################################################################
# create bare noun list
df_barenoun_main = pd.concat(
    [
        df_singleton_marked[["noun", "image1", "state", "adj"]],
        df_singleton_unmarked[["noun", "image1", "state", "adj"]],
    ]
)
df_barenoun_main = df_barenoun_main.rename(columns={"image1": "image"})

# verify unmakred and marked singleton share the objects in the 3 grid positions
for i in range(2,5):
    assert (
        df_singleton_marked[["noun", f"image{i}"]]
        .rename(columns={f"image{i}": "image"})
        .reset_index(drop=True)
        .equals(
            df_singleton_unmarked[["noun", f"image{i}"]]
            .rename(columns={f"image{i}": "image"})
            .reset_index(drop=True)
        )
    )

filler1 = df_singleton_marked[["noun", "image2"]].rename(columns={"image2": "image"})
filler2 = df_singleton_marked[["noun", "image3"]].rename(columns={"image3": "image"})
filler3 = df_singleton_marked[["noun", "image4"]].rename(columns={"image4": "image"})
df_barenoun_filler = pd.concat([filler1, filler2, filler3])

df_barenoun = pd.concat([df_barenoun_main, df_barenoun_filler]).reset_index(drop=True)

######################################################################
# create phrase list
df_expanded = df_barenoun.copy()

# get df for each noun and their two state adjs
pairs = (
    df_barenoun.dropna(subset=["adj"])
    .groupby("noun")["adj"]
    .unique()
    .explode()
    .reset_index()
)

# cross join to duplicate rows for each adjective of the noun
df_phrase = df_expanded.merge(pairs, on="noun", how="left", suffixes=("", "_utterance"))
assert df_phrase["adj_utterance"].isna().any() == False
df_phrase["phrase"] = df_phrase.apply(
    lambda r: f"{r['adj_utterance']} {r['noun']}", axis=1
)

######################################################################
# combine bare and noun lists

df_combined = pd.concat([df_phrase, df_barenoun], ignore_index=True)
df_combined["phrase"] = df_combined["phrase"].fillna(df_combined["noun"])
df_combined = df_combined.rename(columns={"phrase": "utterance"})
df_combined = df_combined.sort_values(["noun", "utterance", "state"])
df_combined = df_combined[
    ["noun", "utterance", "image", "state", "adj", "adj_utterance"]
].reset_index(drop=True)


######################################################################
# test

# test shuffling correctness
test_df = shuffle_no_adjacent(df_combined)
for i in range(len(test_df) - 1):
    assert test_df.iloc[i]["noun"] != test_df.iloc[i + 1]["noun"]

# test whether all images exist
img_dir = "images"
files = set(os.listdir(img_dir))
missing = [f for f in df_combined["image"].unique() if f not in files]
print(f"{len(missing)} missing files")
assert len(missing) == 0

######################################################################
# remove nouns without overspecification and pick two distractors per noun 

filtered_df = df_combined[~df_combined['noun'].isin(removed_nouns)]
df_typicality = pick_normed_per_noun(filtered_df).sort_values('noun')

######################################################################
# shuffling and reversing typicality norming list

for random_number in [1]:
    df_shuffled = shuffle_no_adjacent(df_typicality, random_state=random_number)
    # check whether it is a correct shuffle
    assert set(map(tuple, df_shuffled.to_numpy())) == set(
        map(tuple, df_typicality.to_numpy())
    )
    df_shuffled_reversed = df_shuffled.iloc[::-1].reset_index(drop=True)

# label is the name of image, item is the word.
json_data = [
    {"label": row["image"], "item": [row["utterance"]]}
    for _, row in df_shuffled.iterrows()
]

with open("item_target.json", "w") as f:
    json.dump(json_data, f, indent=4)






















######################################################################
# creating lists for naming experiments

df_naming = df_barenoun.copy()

mask = df_naming['state'].isna() & df_naming['adj'].isna()
df_naming['object'] = np.where(
    mask,
    df_naming['image'].str.replace(r'\.jpe?g$', '', regex=True),
    df_naming['noun']
)
df_naming = df_naming[['object', 'image', 'state', 'adj']]
df_naming_shuffled = shuffle_no_adjacent(df_naming, key='object', random_state=1)

df_naming_shuffled.to_csv("state_overspec_stimuli_naming.csv", index=False)
df_naming_shuffled.head(5).to_csv("test_state_overspec_stimuli_naming.csv", index=False)





