import pandas as pd
import dtale
import re

def middle_sheet_result():
    df = pd.read_csv("./experiments/Modifiers_Data_v3.csv")

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
    n_singleton = len(group[group['Context'] == 'singleton'])
    n_pair = len(group[group['Context'] == 'pair'])

    return pd.Series(
        {
            "$images_1": image_val, 
            "over_spec_n": len(group), 
            "overspec_n_singleton": n_singleton, 
            "overspec_n_pair": n_pair, 
            "all_overspec": all_overspec
            }
        )

df = pd.read_excel("./experiments/Modifiers_Data_v3.xlsx", sheet_name="data")

overspec_rows = df[
    (df["noun_right"] == 1) & (df["state_expected"] == 1) & (df["state_strict"] == 1)
]
results = overspec_rows.groupby(["trial_id", "State_1"]).apply(get_all_overspec)
results["noun"] = results["$images_1"].str.extract(r"^([a-zA-Z]+)")

total_n = (
    df[df["noun_right"] == 1]
    .groupby(["trial_id", "State_1"])
    .size()
    .reset_index(name="total_n")
)
singleton_total_n = (
    df[
        (df["noun_right"] == 1)
        & (df["Context"] == 'singleton')
        ]
    .groupby(["trial_id", "State_1"])
    .size()
    .reset_index(name="singleton_total_n")
)
pair_total_n = (
    df[
        (df["noun_right"] == 1)
        & (df["Context"] == 'pair')
        ]
    .groupby(["trial_id", "State_1"])
    .size()
    .reset_index(name="pair_total_n")
)
agg = (
    total_n
    .merge(singleton_total_n, on=["trial_id", "State_1"], how="left")
    .merge(pair_total_n, on=["trial_id", "State_1"], how="left")
)

results = results.merge(agg, on=["trial_id", "State_1"], how="left")


results[
        (results['noun'] == 'apple')
        & (results['State_1'] == 'state_A') 
        ]

results.to_csv('overspec_rate_result.csv')


results_A = results[results['State_1'] == 'state_A']
results_B = results[results['State_1'] == 'state_B']

middle_sheet = middle_sheet_result()
middle_sheet_normed = middle_sheet[middle_sheet['noun'].isin(results['noun'].unique())]
