import pandas as pd
from model_with_all_words import create_word_world
import matplotlib.pyplot as plt
plt.ion()


df_words = pd.read_csv("./norming_results.csv")  # created in norming_exp repo
df_results = pd.read_csv("./overspec_rate_result.csv")  # created in norming_exp repo
df_results_a = df_results[df_results['State_1'] == 'state_A']
df_results_b = df_results[df_results['State_1'] == 'state_B']

print(df_results_a['overspec_n_singleton'].sum()/df_results_a['singleton_total_n'].sum())
print(df_results_b['overspec_n_singleton'].sum()/df_results_b['singleton_total_n'].sum())

noncomp_data = {}
for word in df_words["noun"].unique():
    utterances, marked_state, unmarked_state, parker_world, noncomp_semvalue = (
        create_word_world(word)
    )
    noncomp_data[word] = {
        "marked_state": marked_state,
        "unmarked_state": unmarked_state,
        "bare_marked": noncomp_semvalue['bare_marked'],
        "bare_unmarked": noncomp_semvalue['bare_unmarked'],
        "modified_marked_T": noncomp_semvalue['modified_marked_T'],
        "modified_marked_F": noncomp_semvalue['modified_marked_F'],
        "modified_unmarked_T": noncomp_semvalue['modified_unmarked_T'],
        "modified_unmarked_F": noncomp_semvalue['modified_unmarked_F'],
    }

noncomp_df = pd.DataFrame(noncomp_data).T
noncomp_df = noncomp_df.sort_values('marked_state')
noncomp_df['bare_diff'] = noncomp_df['bare_unmarked'] - noncomp_df['bare_marked']


df = noncomp_df.reset_index().rename(columns={'index': 'noun'})
states = df['marked_state'].unique()
plt.figure(figsize=(10, 5))

for state in states:
    sub = df[df['marked_state'] == state]
    plt.scatter(sub.index, sub['bare_diff'], label=state)

words_to_annotate = ["eye", "dish", "glasses", "jar", "apple", "gate"]
words_to_annotate_2 = ['wallet', 'book', 'scissors']
for w in words_to_annotate:
    row = df[df['noun'] == w].iloc[0]
    x = row.name
    y = row["bare_diff"]

    plt.annotate(
        w,
        xy=(x, y),
        xytext=(x + 3, y + 0.03),
        arrowprops=dict(arrowstyle="->", linewidth=1)
    )
for w in words_to_annotate_2:
    row = df[df['noun'] == w].iloc[0]
    x = row.name
    y = row["bare_diff"]

    plt.annotate(
        w,
        xy=(x, y),
        xytext=(x -1.5 , y - 0.06),
        arrowprops=dict(arrowstyle="->", linewidth=1)
    )
plt.axhline(0, color='grey', linewidth=1)
plt.xticks([])
plt.gca().tick_params(axis='x', which='both', length=0)
# plt.xlabel("Item index")
plt.ylabel("typicality difference between marked and unmarked state")
plt.legend(title="marked state")
plt.tight_layout()
plt.savefig('./analysis/typicality_diff.pdf')
plt.show()

# plt.figure(figsize=(7, 6))
# states = df['marked_state'].unique()
# for state in states:
#     sub = df[df['marked_state'] == state]
#     plt.scatter(sub['bare_marked'], sub['bare_unmarked'], label=state)

# # annotate selected words
# words_to_annotate = ["eye"]
# for w in words_to_annotate:
#     row = df[df['noun'] == w].iloc[0]
#     x = row["bare_marked"]
#     y = row["bare_unmarked"]

#     plt.annotate(
#         w,
#         xy=(x, y),
#         xytext=(x + 0.02, y + 0.02),
#         arrowprops=dict(arrowstyle="->", linewidth=1)
#     )

# plt.xlabel("bare_marked")
# plt.ylabel("bare_unmarked")
# plt.legend(title="marked_state")
# plt.tight_layout()
# plt.show()
