import dtale
import numpy as np
import pandas as pd
import gensim.downloader as api
from itertools import combinations, combinations_with_replacement
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt


def get_vec(w, wv):
    parts = w.split()  # works for single words too
    vecs = [wv[p] for p in parts if p in wv]
    return np.mean(vecs, axis=0)

wv = api.load("word2vec-google-news-300")
df_naming = pd.read_csv('naming_master_list.csv')


format_words = {
                'cdplayer': 'cd player',
                'chestofdrawers': 'chest of drawers',
                'chipbag': 'chip bag',
                'chocolatebar': 'chocolate bar',
                'eggcarton': 'egg carton',
                'filefolder': 'file folder',
                'garbagecan': 'garbage can',
                'juicebottle': 'juice bottle',
                'milkcarton': 'milk carton',
                'safetypin': 'safety pin',
                'shoppingcart': 'shopping cart',
                'sleepingbag': 'sleeping bag',
                'sugarpacket': 'sugar packet',
                'treasurechest': 'treasure chest',
                'yogamat': 'yoga mat'
                }

df_naming["words_w2v"] = (
    df_naming["object"]
    .map(format_words)
    .fillna(df_naming["object"])
)

words = df_naming["words_w2v"].astype(str).unique().tolist()

rows = []
for w1, w2 in combinations_with_replacement(words, 2):
    v1 = get_vec(w1, wv)
    v2 = get_vec(w2, wv)

    sim = 1 - cosine(v1, v2)
    rows.append((w1, w2, sim))

pairwise_df = pd.DataFrame(rows, columns=["word1", "word2", "cosine_similarity"])
pairwise_df["pair"] = pairwise_df[["word1","word2"]].apply(
            lambda r: "|".join(sorted(r.astype(str))), axis=1
            )


def shuffle_no_adjacent(
        df, 
        pairwise_df=pairwise_df,
        ):
    remaining = df.copy()
    result = []

    threshold = pairwise_df["cosine_similarity"].quantile(0.75)
    prev_noun = remaining["words_w2v"].sample(n=1, random_state=np.random.randint(0, 2**32 - 1)).iloc[0]
    while not remaining.empty:
        high_similarity = True

        tries = 0
        max_tries = len(remaining) * 20   # enough to detect hanging

        while high_similarity:
            tries += 1
            if tries > max_tries:
                print("HANG WARNING: no candidate < threshold",
                      "; prev_noun =", prev_noun,
                      "; remaining =", len(remaining),
                      "; threshold =", threshold,
                      "; current sim =", current_similarity)
                # break
                return None

            row = remaining.sample(1, random_state=np.random.randint(0, 2**32 - 1))
            current_noun = row['words_w2v'].iloc[0]
            current_pair = "|".join(sorted((prev_noun, current_noun)))
            current_similarity = pairwise_df[pairwise_df["pair"] == current_pair]['cosine_similarity'].iloc[0]

            if current_similarity < threshold:
                high_similarity = False

        result.append(row)
        prev_noun = row.iloc[0]['words_w2v']
        remaining = remaining.drop(row.index)
    print("done!\n")
    return pd.concat(result, ignore_index=True)

np.random.seed(123)
dfs_shuffled = []
while len(dfs_shuffled) < 50:
    df_shuffled = shuffle_no_adjacent(df_naming)
    if df_shuffled is not None:
        df_shuffled_reversed = df_shuffled.iloc[::-1].reset_index(drop=True)
        dfs_shuffled.append(df_shuffled)
        dfs_shuffled.append(df_shuffled_reversed)


for index, df in enumerate(dfs_shuffled):
    df.to_csv(f'./naming_list/naming_list_{index}.csv')
