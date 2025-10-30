import pandas as pd
import re
import dtale
from collections import Counter
import os
import glob
import shutil

df = pd.read_excel("./Modifiers_Data_v3.xlsx", sheet_name="data")

def copy_target_images():
    source_dir = "images"
    destination_dir = "openAI_images"
    os.makedirs(destination_dir, exist_ok=True)

    prefixes = (
        results['$images_1']
        .dropna()
        .astype(str)
        .str.split('_')
        .str[0]
        .add('_')
        .unique()
    )

    matched_filenames = (
        df['$images_1']
        .dropna()
        .astype(str)
        .loc[lambda series: series.str.startswith(tuple(prefixes)) & (series.str.count('_') == 1)]
        .unique()
    )

    for filename in matched_filenames:
        source_path = os.path.join(source_dir, filename)
        if os.path.isfile(source_path):
            shutil.copy2(source_path, destination_dir)
        else:
            print(filename)

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

    return summary_df

def get_all_overspec(group):
    cols = [
        "overspecification1",
        "overspecification2",
        "overspecification3",
        "overspecification4",
        "overspecification5",
    ]
    all_overspec = [v for v in group[cols].values.ravel().tolist() if pd.notna(v)]
    image_val = (
        group["$images_1"].iloc[0]
        if "$images_1" in group and not group["$images_1"].empty
        else None
    )
    trial_id = group['trial_id'].iloc[0] 
    state = group['State_1'].iloc[0] 
    # return pd.Series({"trial_id": trial_id, "state": state, "$images_1": image_val, 'n':len(group), "all_overspec": all_overspec})
    return pd.Series({"$images_1": image_val, 'n':len(group), "all_overspec": all_overspec})

def list_modes(lst):
    if not lst:
        return []
    counts = Counter(lst)
    max_count = max(counts.values())
    return [k for k, v in counts.items() if v == max_count]


overspec_rows = df[
        (df['noun_right'] == 1)
        & (df["state_expected"] == 1)
        & (df["state_strict"] == 1)
        ]
results = overspec_rows.groupby(["trial_id", "State_1"]).apply(get_all_overspec)
results['mode_overspec'] = results['all_overspec'].apply(list_modes)
results['noun'] = results['$images_1'].str.extract(r'^([a-zA-Z]+)')

total_n = (
    df[df['noun_right'] == 1]
    .groupby(['trial_id', 'State_1'])
    .size()
    .reset_index(name='total_n')
)
results = results.merge(total_n, left_on=['trial_id', 'State_1'], right_on=['trial_id', 'State_1'], how='left')











df2 = nouns_without_overspec()
results = results.merge(df2[['noun','grand_total']], how='left', on='noun')
results['grand_total'] = pd.to_numeric(results['grand_total'], errors='coerce').astype('Int64')

print(results)
copy_target_images()













df2['grand_total'] = pd.to_numeric(df2['grand_total'], errors='coerce')

res_sum = results.groupby('noun')['n'].sum().reset_index()
merged = df2[['noun','grand_total']].merge(res_sum, on='noun', how='inner')
diffs = merged[merged['grand_total'] != merged['n']]
