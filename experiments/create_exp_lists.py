import pandas as pd
import json
import os
import numpy as np


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


# test shuffling correctness
xxx = shuffle_no_adjacent(df_combined)
for i in range(len(xxx) - 1):
    assert xxx.iloc[i]["noun"] != xxx.iloc[i + 1]["noun"]

# test whether all images exist
img_dir = "images"
files = set(os.listdir(img_dir))
missing = [f for f in df_combined["image"].unique() if f not in files]
print(f"{len(missing)} missing files")
assert len(missing) == 0


######################################################################
# shuffling and reversing lists

for random_number in [1]:
    df_shuffled = shuffle_no_adjacent(df_combined, random_state=random_number)
    # check whether it is a correct shuffle
    assert set(map(tuple, df_shuffled.to_numpy())) == set(
        map(tuple, df_combined.to_numpy())
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
df_naming_shuffled = shuffle_no_adjacent(df_naming, key='object', random_state=1)

df_naming_shuffled.to_csv("state_overspec_stimuli_naming.csv", index=False)

df_naming_shuffled.head(5).to_csv("test_state_overspec_stimuli_naming.csv", index=False)
