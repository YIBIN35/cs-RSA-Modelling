import pandas as pd
import json
import os
import re
import numpy as np
import random
import itertools
import dtale

def copy_target_images():
    src_dir = "images"
    dest_dir = "openAI_images"
    os.makedirs(dest_dir, exist_ok=True)

    for prefix in results['$images_1'].str.split('_').str[0].unique():
        for file in glob.glob(os.path.join(src_dir, f"{prefix}_*")):
            shutil.copy(file, dest_dir)

def nouns_with_overspec():
    df = pd.read_excel("./Modifiers_Data_v3.xlsx", sheet_name="data")
    df = df[(df["noun_right"] == 1) & (df["state_expected"] == 1) & (df["state_strict"] == 1)]
    df['noun'] = df["$images_1"].str.extract(r"^([a-zA-Z]+)")
    nouns = df['noun'].unique()
    return nouns

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

def pick_distractors_per_noun(df):
    # create N versions of exp list, differ in terms of distractors

    per_noun_data = []
    for noun, grp in df.groupby('noun', sort=False):
        non_distractor = grp[grp['state'].notna()]
        distractors = _pick_two_distractor_rows(grp)
        per_noun_data.append((non_distractor, distractors))

    num_version = len(per_noun_data[0][1])
    all_outputs = []

    for version_id in range(num_version):
        combined_rows = []
        for non_distractor, distractors in per_noun_data:
            distractor = distractors[version_id]
            combined_rows.append(pd.concat([non_distractor, distractor], ignore_index=True))
        all_outputs.append(pd.concat(combined_rows, ignore_index=True))

    return all_outputs

def add_natural_adj_utterance(filtered_df, openai_df):
    """
    For each noun, map Marked→state_a(natural) and Unmarked→state_b(natural) from openai_df,
    Returns a copy of filtered_df with a new column 'natural_adj_utterance'.
    """
    # Build mapping table
    m = openai_df[['item','noun_wt_space', 'Marked', 'Unmarked', 'state_a', 'state_b']].copy()
    for c in ['item','noun_wt_space', 'Marked', 'Unmarked', 'state_a', 'state_b']:
        m[c] = m[c].astype(str).str.strip().str.lower()
    mapping = pd.concat([
        m.set_index(['noun_wt_space', 'Marked'])['state_a'],
        m.set_index(['noun_wt_space', 'Unmarked'])['state_b']
    ])

    # Map adjective using both noun and adj as keys
    df = filtered_df.copy()
    keys = list(zip(df['noun'], df['adj_utterance']))
    mapped = pd.Series(keys).map(mapping)
    df['natural_adj_utterance'] = np.where(mapped.notna(), mapped, df['adj_utterance'])

    # add noun WITHOUT whitespace
    df = (
        df.merge(
            m[['item', 'noun_wt_space']],
            left_on='noun',
            right_on='noun_wt_space',
            how='left'
        )
        .drop(columns=['noun_wt_space'])
    )

    # create natural utterance
    df['natural_utterance'] = np.where(
        df['natural_adj_utterance'].isna(),
        df['item'],
        df['natural_adj_utterance'].str.strip() + ' ' + df['item']
    )

    return df

if __name__ == '__main__':

    df2 = pd.read_excel("./Parker_Modifiers_Trials_May16.xlsx")

    df_exptrial = df2[df2["type"] == "exp"]
    df_exptrial_singleton = df2[(df2["type"] == "exp") & (df2["group"] == "single")]

    df_singleton_marked = df_exptrial[
        (df_exptrial["group"] == "single") & (df_exptrial["state"] == "a")
    ]
    df_singleton_unmarked = df_exptrial[
        (df_exptrial["group"] == "single") & (df_exptrial["state"] == "b")
    ]

    overspec_nouns = nouns_with_overspec()

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
    # remove nouns without overspecification 
    # add openai enhanced natural modifiers
    # pick two distractors per noun for each distractor version

    filtered_df = df_combined[df_combined['noun'].isin(overspec_nouns)]
    openai_df = pd.read_excel('openai_production_adj_selection_manualdone.xlsx')
    filtered_df = add_natural_adj_utterance(filtered_df, openai_df)

    # right now just pick the first distractor_version
    nine_df_typicality = pick_distractors_per_noun(filtered_df)

    ######################################################################
    # shuffling and splitting typicality norming list

    random_numbers = [33,46,7,89,6,17,4489,4,8888]
    nine_df_typicality_shuffled = []

    for df_typicality, random_number in zip(nine_df_typicality,random_numbers):
        df_shuffled = shuffle_no_adjacent(df_typicality, random_state=random_number)
        df_shuffled_reversed = df_shuffled.iloc[::-1].reset_index(drop=True)
        nine_df_typicality_shuffled.append(df_shuffled)


    for version, df_typicality_shuffled in enumerate(nine_df_typicality_shuffled):

        df_len = len(df_typicality_shuffled)
        session_len = int(df_len/4)
        assert df_len == 808

        for session, row_index in enumerate([i for i in range(0, df_len, session_len)]):
            df_session = df_typicality_shuffled.iloc[row_index:row_index+session_len]
            json_data = [
                {"label": row["image"], "item": [row["natural_utterance"]]}
                for _, row in df_session.iterrows()
            ]
            with open(f"typicality_list/item_target_version{version}_session{session}.json", "w") as f:
                json.dump(json_data, f, indent=4)

            # save a test version
            df_session_test = df_typicality_shuffled.iloc[row_index:row_index+5]
            json_data_test = [
                {"label": row["image"], "item": [row["natural_utterance"]]}
                for _, row in df_session_test.iterrows()
            ]
            with open(f"typicality_list_test/item_target_version{version}_session{session}.json", "w") as f:
                json.dump(json_data_test, f, indent=4)


    # create practice list
    practice_df_typicality = shuffle_no_adjacent(
            pick_distractors_per_noun(
                df_combined[~df_combined['noun'].isin(overspec_nouns)]
                )[0]
            ).loc[[0, 18, 30]] # manually pick these three items
    json_data_practice = [
        {"label": row["image"], "item": [row["utterance"]]}
        for _, row in practice_df_typicality.iterrows()
    ]
    with open(f"typicality_list/practice_items.json", "w") as f:
        json.dump(json_data_practice, f, indent=4)

    import ipdb; ipdb.set_trace()





















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





