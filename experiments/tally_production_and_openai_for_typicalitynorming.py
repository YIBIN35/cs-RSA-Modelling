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
        results["$images_1"]
        .dropna()
        .astype(str)
        .str.split("_")
        .str[0]
        .add("_")
        .unique()
    )

    matched_filenames = (
        df["$images_1"]
        .dropna()
        .astype(str)
        .loc[
            lambda series: series.str.startswith(tuple(prefixes))
            & (series.str.count("_") == 1)
        ]
        .unique()
    )

    for filename in matched_filenames:
        source_path = os.path.join(source_dir, filename)
        if os.path.isfile(source_path):
            shutil.copy2(source_path, destination_dir)
        else:
            print(filename)


def middle_sheet_result():
    df = pd.read_csv("./Modifiers_Data_v3.csv")

    df.columns = [
        "Row Labels",
        "A_singleton",
        "A_pair",
        "A_total",
        "B_singleton",
        "B_pair",
        "B_total",
        "grand_total",
        "check",
    ]
    df = df.drop(index=range(448, 452))  # drop the grand total and so on at the end.
    df = df.drop(index=range(0, 4))  # drop the first couple of rows
    df = df.reset_index(drop=True)

    for i in range(len(df) - 1):  # avoid last row overflow
        val = str(df.loc[i, "Row Labels"]).strip()
        if re.fullmatch(r"\d+", val):
            next_label = str(df.loc[i + 1, "Row Labels"])
            noun = next_label.split("_")[0]
            df.loc[i, "Row Labels"] = "summary " + noun

    summary_df = df[df["Row Labels"].str.startswith("summary")].copy()
    summary_df["noun"] = summary_df["Row Labels"].str.split(" ").str[1]

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
    trial_id = group["trial_id"].iloc[0]
    state = group["State_1"].iloc[0]
    # return pd.Series({"trial_id": trial_id, "state": state, "$images_1": image_val, 'n':len(group), "all_overspec": all_overspec})
    return pd.Series(
        {"$images_1": image_val, "n": len(group), "all_overspec": all_overspec}
    )


def list_modes(lst):
    if not lst:
        return []
    counts = Counter(lst)
    max_count = max(counts.values())
    return [k for k, v in counts.items() if v == max_count]


# copy_target_images()

overspec_rows = df[
    (df["noun_right"] == 1) & (df["state_expected"] == 1) & (df["state_strict"] == 1)
]
results = overspec_rows.groupby(["trial_id", "State_1"]).apply(get_all_overspec)
results["mode_overspec"] = results["all_overspec"].apply(list_modes)
results["noun"] = results["$images_1"].str.extract(r"^([a-zA-Z]+)")

total_n = (
    df[df["noun_right"] == 1]
    .groupby(["trial_id", "State_1"])
    .size()
    .reset_index(name="total_n")
)
results = results.merge(
    total_n,
    left_on=["trial_id", "State_1"],
    right_on=["trial_id", "State_1"],
    how="left",
)

df2 = middle_sheet_result()
import ipdb; ipdb.set_trace()
results = results.merge(df2[["noun", "grand_total"]], how="left", on="noun")
results["grand_total"] = pd.to_numeric(results["grand_total"], errors="coerce").astype(
    "Int64"
)

print(results)

# create a list for manual selecting open

df_temp = pd.read_csv("./parker_stimuli.csv")
df_temp.loc[74, "Item "] = 'shoelace'
df_temp["noun_wt_space"] = (
    df_temp["Item "].str.strip().str.replace(r"\s+", "", regex=True)
)

df_manual = pd.DataFrame()
df_manual["noun"] = results["noun"].unique()
df_manual = df_manual.merge(
    df_temp, left_on="noun", right_on="noun_wt_space", how="left"
)
df_manual = df_manual.drop(columns="noun")
df_manual = df_manual.rename(columns={"Item ": "item"})

wide = (
    results.pivot(index="noun", columns="State_1", values="mode_overspec")
    .rename(
        columns={
            "state_A": "state_a_mode_production",
            "state_B": "state_b_mode_production",
        }
    )
    .reset_index()
)
df_manual = df_manual.merge(
    wide[["noun", "state_a_mode_production", "state_b_mode_production"]],
    left_on="noun_wt_space",
    right_on="noun",
    how="left",
).drop(columns=["noun"])

df_openai = pd.read_csv("openai_natural_adjective.csv")
df_manual = df_manual.merge(
    df_openai[["noun", "files", "prompt", "answer", "adjectives", "state_A_adj", "state_B_adj"]],
    left_on="noun_wt_space",
    right_on="noun",
    how="left",
).drop(columns=["noun"])
cols_to_prefix = [
    "files",
    "prompt",
    "answer",
    "adjectives",
    "state_A_adj",
    "state_B_adj",
]
df_manual = df_manual.rename(columns={c: f"openai_{c}" for c in cols_to_prefix})
df_manual = df_manual.rename(columns={'openai_state_A_adj':'openai_state_1', 'openai_state_B_adj':'openai_state_2'})
df_manual = df_manual[[
    "item",
    "Marked",
    "Unmarked",
    "noun_wt_space",
    "openai_files",
    "openai_prompt",
    "openai_answer",
    "openai_adjectives",
    "openai_state_1",
    "openai_state_2",
    "state_a_mode_production",
    "state_b_mode_production",
]]

df_manual.to_csv('openai_production_adj_selection.csv')


# df2['grand_total'] = pd.to_numeric(df2['grand_total'], errors='coerce')

# res_sum = results.groupby('noun')['n'].sum().reset_index()
# merged = df2[['noun','grand_total']].merge(res_sum, on='noun', how='inner')
# diffs = merged[merged['grand_total'] != merged['n']]
