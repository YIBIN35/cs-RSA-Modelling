import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing, minimize, brute
import pandas as pd
import dtale
from pprint import pprint

plt.ion()

df_words = pd.read_csv("./norming_results.csv") # created in norming_exp repo

# parker_world = {
#     "singleton_marked": [
#         {"size": None, "state": "open", "nominal": "door"},
#         {"size": None, "state": None, "nominal": "other1"},
#         {"size": None, "state": None, "nominal": "other2"},
#         {"size": None, "state": None, "nominal": "other3"},
#     ],
#     "singleton_unmarked": [
#         {"size": None, "state": "closed", "nominal": "door"},
#         {"size": None, "state": None, "nominal": "other1"},
#         {"size": None, "state": None, "nominal": "other2"},
#         {"size": None, "state": None, "nominal": "other3"},
#     ],
#     "pair_marked": [
#         {"size": "big", "state": "open", "nominal": "door"},
#         {"size": "small", "state": "open", "nominal": "door"},
#         {"size": None, "state": None, "nominal": "other2"},
#         {"size": None, "state": None, "nominal": "other3"},
#     ],
#     "pair_unmarked": [
#         {"size": "big", "state": "closed", "nominal": "door"},
#         {"size": "small", "state": "closed", "nominal": "door"},
#         {"size": None, "state": None, "nominal": "other2"},
#         {"size": None, "state": None, "nominal": "other3"},
#     ],
# }


def create_word_world(word, df_words=df_words):

    df_word = df_words[df_words["noun"] == word]
    assert len(df_word[df_word["state"] == "a"]["adj"].unique()) == 1

    marked_state = df_word[df_word["state"] == "a"]["adj"].unique()[0]
    unmarked_state = df_word[df_word["state"] == "b"]["adj"].unique()[0]

    parker_world = {
        "singleton_marked": [
            {"size": None, "state": marked_state, "nominal": word},
            {"size": None, "state": None, "nominal": "other1"},
            {"size": None, "state": None, "nominal": "other2"},
            {"size": None, "state": None, "nominal": "other3"},
        ],
        "singleton_unmarked": [
            {"size": None, "state": unmarked_state, "nominal": word},
            {"size": None, "state": None, "nominal": "other1"},
            {"size": None, "state": None, "nominal": "other2"},
            {"size": None, "state": None, "nominal": "other3"},
        ],
        "pair_marked": [
            {"size": "big", "state": marked_state, "nominal": word},
            {"size": "small", "state": marked_state, "nominal": word},
            {"size": None, "state": None, "nominal": "other2"},
            {"size": None, "state": None, "nominal": "other3"},
        ],
        "pair_unmarked": [
            {"size": "big", "state": unmarked_state, "nominal": word},
            {"size": "small", "state": unmarked_state, "nominal": word},
            {"size": None, "state": None, "nominal": "other2"},
            {"size": None, "state": None, "nominal": "other3"},
        ],
    }

    noncomp_semvalue = {}
    noncomp_semvalue["bare_marked"] = (
        df_word[(df_word["adj_utterance"].isna()) & (df_word["adj"] == marked_state)][
            "mean_response"
        ].values
        / 7
    ).item()
    noncomp_semvalue["bare_unmarked"] = (
        df_word[(df_word["adj_utterance"].isna()) & (df_word["adj"] == unmarked_state)][
            "mean_response"
        ].values
        / 7
    ).item()

    noncomp_semvalue["modified_marked_T"] = (
        df_word[
            (df_word["adj_utterance"] == marked_state)
            & (df_word["adj"] == marked_state)
        ]["mean_response"].values
        / 7
    ).item()
    noncomp_semvalue["modified_marked_F"] = (
        df_word[
            (df_word["adj_utterance"] == marked_state)
            & (df_word["adj"] == unmarked_state)
        ]["mean_response"].values
        / 7
    ).item()

    noncomp_semvalue["modified_unmarked_T"] = (
        df_word[
            (df_word["adj_utterance"] == unmarked_state)
            & (df_word["adj"] == unmarked_state)
        ]["mean_response"].values
        / 7
    ).item()
    noncomp_semvalue["modified_unmarked_F"] = (
        df_word[
            (df_word["adj_utterance"] == unmarked_state)
            & (df_word["adj"] == marked_state)
        ]["mean_response"].values
        / 7
    ).item()

    utterances = [
        word,
        f'{marked_state} {word}',
        f'{unmarked_state} {word}',
        "other1",
        "other2",
        "other3",
    ]

    return utterances, marked_state, unmarked_state, parker_world, noncomp_semvalue


class cs_rsa:

    def __init__(
        self,
        world,
        marked_state,
        unmarked_state,
        noncomp_semvalue_dict,
        model_type='mixture',
        alpha=13.7,
        beta_fixed=0.69,
        costWeight=0,
        typicalityWeight=1.34,
        size_semvalue=0.8,
        state_semvalue_marked=0.9,
        state_semvalue_unmarked=0.9,
        nominal_semvalue=0.99,
    ):

        # model type
        self.model_type = model_type


        # parameters
        self.alpha = alpha
        self.beta_fixed = beta_fixed
        self.costWeight = costWeight
        self.typicalityWeight = typicalityWeight

        # semantic values
        self.size_semvalue = size_semvalue
        self.state_semvalue_marked = state_semvalue_marked
        self.state_semvalue_unmarked = state_semvalue_unmarked
        self.nominal_semvalue = nominal_semvalue
        self.noncomp_semvalue_dict = noncomp_semvalue_dict

        # world
        self.world = world

        # states
        self.marked_state = marked_state
        self.unmarked_state = unmarked_state

        # vocab
        self.sizes = ["big", "small"]
        self.states = [self.marked_state, self.unmarked_state]
        self.nominals = [
            world[0]["nominal"],
            "other1",
            "other2",
            "other3",
        ]

    def _parse_utterance(self, utt):
        """Return a parsed dict {'size': token|None, 'state': token|None, 'nominal': token|None}."""
        parts = utt.split(" ")
        out = {"size": None, "state": None, "nominal": None}

        cat_map = {}
        for s in self.sizes:
            cat_map[s] = ("size", s)
        for s in self.states:
            cat_map[s] = ("state", s)
        for n in self.nominals:
            cat_map[n] = ("nominal", n)

        for w in parts:
            if w not in cat_map:
                raise ValueError(f"Unknown token in utterance: {w}")
            cat, val = cat_map[w]
            if out[cat] is not None:
                # You can choose to allow multiple-of-a-kind; here we forbid to avoid silent overwrite
                raise ValueError(
                    f"Utterance has multiple {cat} tokens: {out[cat]} and {val}"
                )
            out[cat] = val
        return out

    def _comp_semvalue(self, parsed, obj, print_value=False):
        """Compositional/fixed semantics as a product of per-feature scores."""

        size_word = parsed["size"]
        state_word = parsed["state"]
        nominal_word = parsed["nominal"]

        # Defaults; if size/state/nominal_word == None: sem_val = 1
        size_val = 1.0
        state_val = 1.0
        nominal_val = 1.0

        # SIZE
        if size_word is not None:
            # this condition may not matter as it won't appear.
            if obj["size"] is None:
                size_val = self.size_semvalue  # "big" -> None | "small" -> None
            else:
                if size_word == obj["size"]:
                    size_val = self.size_semvalue  # "big" -> big | "small" -> small
                else:
                    size_val = (
                        1.0 - self.size_semvalue
                    )  # "small" -> big | "big" -> small
        # NOMINAL
        if nominal_word is not None:
            if obj["nominal"] is None:
                raise Exception(
                    "Nominal category of the object is none"
                )  # door -> None | other -> None
            else:
                if nominal_word == obj["nominal"]:
                    nominal_val = self.nominal_semvalue  # "door" -> door
                else:
                    nominal_val = (
                        1.0 - self.nominal_semvalue
                    )  # "door" -> None | "other" -> None
        # STATE
        if state_word is not None:
            if state_word == self.marked_state:
                if obj["state"] == self.marked_state:
                    state_val = self.state_semvalue_marked  # "open" -> open
                else:
                    state_val = (
                        1.0 - self.state_semvalue_marked
                    )  # "open" -> None | "open" -> closed
            elif state_word == self.unmarked_state:
                if obj["state"] == self.unmarked_state:
                    state_val = self.state_semvalue_unmarked  # "closed" -> closed
                else:
                    state_val = (
                        1.0 - self.state_semvalue_unmarked
                    )  # "closed" -> open | "closed" -> None
            else:
                raise ValueError(f"Unexpected state token: {state_word}")

        if print_value == True:
            print(
                f"beta_fixed:{self.beta_fixed:.2f}, size:{size_val:.2f}, state:{state_val:.2f}, nominal:{nominal_val:.2f}"
            )
        return size_val * state_val * nominal_val

    def _noncomp_semvalue(self, parsed, obj, print_value=False):
        """
        Empirical/non-compositional semantic value function
        check table 7 to understand the code here. right now everything is hardcoded!!!
        Returns a number in [0,1].
        """
        nominal = parsed["nominal"]
        state = parsed["state"]

        if nominal is None or obj["nominal"] is None:
            # If utterance lacks a nominal, you need a policy. Here: rely on other parts (neutral).
            raise Exception(
                "Nominal category of the object is none"
            )  # door -> None | other -> None

        elif nominal != obj["nominal"]:
            noncomp_semval = 1.0 - self.nominal_semvalue

        elif nominal == obj["nominal"]:
            if nominal in ["other1", "other2", "other3"]:
                noncomp_semval = self.nominal_semvalue

            elif state is None:  # "door"
                if obj["state"] == self.unmarked_state:
                    noncomp_semval = self.noncomp_semvalue_dict[
                        "bare_unmarked"
                    ]  # "door" -> closed door
                elif obj["state"] == self.marked_state:
                    noncomp_semval = self.noncomp_semvalue_dict[
                        "bare_marked"
                    ]  # "door" -> open door 0.66
                else:
                    raise Exception("something is wrong")

            elif state == self.unmarked_state:  # 'closed_door'
                if obj["state"] == self.unmarked_state:
                    noncomp_semval = self.noncomp_semvalue_dict[
                        "modified_unmarked_T"
                    ]  # "closed_door" -> closed door
                else:
                    noncomp_semval = self.noncomp_semvalue_dict[
                        "modified_unmarked_F"
                    ]  # "closed_door" -> open door

            elif state == self.marked_state:  # 'open_door'
                if obj["state"] == self.marked_state:
                    noncomp_semval = self.noncomp_semvalue_dict[
                        "modified_marked_T"
                    ]  # "open_door" -> open door
                else:
                    noncomp_semval = self.noncomp_semvalue_dict[
                        "modified_marked_F"
                    ]  # "open_door" -> closed door
        else:
            raise Exception("something is wrong")
        if print_value == True:
            print(f"1-beta_fixed:{1-self.beta_fixed:.2f}, emp:{noncomp_semval:.2f}")
        return noncomp_semval

    def meaning(self, utt, obj, print_value=False):
        parsed = self._parse_utterance(utt)

        if print_value == True:
            print("utterance:", parsed)
            print("object:", obj)

        fixed_sem_value = self._comp_semvalue(parsed, obj, print_value=print_value)
        empirical_sem_value = self._noncomp_semvalue(
            parsed, obj, print_value=print_value
        )

        sem_value = (
            self.beta_fixed * fixed_sem_value
            + (1.0 - self.beta_fixed) * empirical_sem_value
        )

        if print_value == True:
            print(f"sem_val: {sem_value:.2f}")
            print("")

        if self.model_type == 'compositional':
            return fixed_sem_value
        elif self.model_type == 'non-compositional':
            return empirical_sem_value
        elif self.model_type == 'mixture':
            return sem_value
        else:
            raise Exception

    def literal_listener(self, utterance):
        obj_keys = [tuple(sorted(obj.items())) for obj in self.world]
        sem_vals = np.array(
            [self.meaning(utterance, obj) for obj in self.world],
            dtype=float
        )
        exp_sem_vals = np.exp(self.typicalityWeight * sem_vals)
        probs = exp_sem_vals / exp_sem_vals.sum()

        return {item: prob for item, prob in zip(obj_keys, probs)}

    def cost(self, utt):
        return len(utt.split(" "))

    def pragmatic_speaker(self, obj, utterances):
        obj_key = tuple(sorted(obj.items()))
        informativeness = []
        costs = []

        for utt in utterances:
            literal_listener_prob = self.literal_listener(utt)
            meaning_prob = literal_listener_prob.get(obj_key)

            informativeness.append(self.alpha * math.log(meaning_prob))
            costs.append(self.cost(utt))

        utilities = np.array(informativeness) - self.costWeight * np.array(costs)
        exp_utilities = np.exp(utilities)
        probs = exp_utilities / exp_utilities.sum()

        return {utt: prob for utt, prob in zip(utterances, probs)}

######################################################################
# functions to run singleton and paired conditions, and function to get the best semantic values and other parameters


def singleton_overspecification_rate(
    word,
    model_type='mixture',
    alpha=13.7,
    beta_fixed=0.69,
    state_semvalue_marked=0.9,
    state_semvalue_unmarked=0.9,
    nominal_semvalue=0.99,
    costWeight=0,
    typicalityWeight=1.34,
):

    utterances, marked_state, unmarked_state, world, noncomp_semvalue_dict = create_word_world(word)
    conditions = ["singleton_marked", "singleton_unmarked"]
    overspecified_utts = [[utterances[1]],[utterances[2]]]
    correct_utts = [[utterances[0], utterances[1]], [utterances[0], utterances[2]]]

    overspecification_rates = []

    for i, condition in enumerate(conditions):
        this_world = world[condition]

        model = cs_rsa(
            this_world,
            marked_state,
            unmarked_state,
            noncomp_semvalue_dict,
            model_type=model_type,
            alpha=alpha,
            beta_fixed=beta_fixed,
            costWeight=costWeight,
            typicalityWeight=typicalityWeight,
            state_semvalue_marked=state_semvalue_marked,
            state_semvalue_unmarked=state_semvalue_unmarked,
            nominal_semvalue=nominal_semvalue,
        )
        results = model.pragmatic_speaker(this_world[0], utterances)
        # print(results)

        numer = sum(results[u] for u in overspecified_utts[i])
        denom = sum(results[u] for u in correct_utts[i])
        overspecification_rates.append(numer / denom)

    return overspecification_rates


def pair_overspecification_rate(
    parker_world,
    alpha=13.7,
    beta_fixed=0.69,
    state_semvalue_marked=0.95,
    state_semvalue_unmarked=0.9,
    costWeight=0,
    noncomp_semvalue_bare_unmarked=0.66,
    typicalityWeight=1.34,
):

    utterances = [
        "door",
        "open_door",
        "closed_door",
        "big_door",
        "big_open_door",
        "big_closed_door",
        "small_door",
        "small_open_door",
        "small_closed_door",
        "other2",
        "other3",
    ]
    conditions = ["pair_marked", "pair_unmarked"]
    overspecified_utts = [["big_open_door"], ["big_closed_door"]]
    correct_utts = [["big_door", "big_open_door"], ["big_door", "big_closed_door"]]

    overspecification_rates = []
    for i, condition in enumerate(conditions):
        this_world = parker_world[condition]

        model = cs_rsa(
            this_world,
            alpha=alpha,
            beta_fixed=beta_fixed,
            state_semvalue_marked=state_semvalue_marked,
            state_semvalue_unmarked=state_semvalue_unmarked,
            noncomp_semvalue_bare_unmarked=noncomp_semvalue_bare_unmarked,
            costWeight=costWeight,
            typicalityWeight=typicalityWeight,
        )
        results = model.pragmatic_speaker(this_world[0], utterances)

        numer = sum(results[u] for u in overspecified_utts[i])
        denom = sum(results[u] for u in correct_utts[i])
        overspecification_rates.append(numer / denom)

    return overspecification_rates

def compute_targets(source='raw'):
    if source == 'raw':
        df = pd.read_csv('./overspec_rate_result.csv')
    elif source == 'middle':
        df = pd.read_csv('./overspec_rate_result_middle.csv')
    else:
        raise Exception

    targets = {}
    counts = {}

    for noun, sub in df.groupby("noun"):
        rates = []
        nums = []
        dens = []

        for state in ["state_A", "state_B"]:
            sub_state = sub[sub["State_1"] == state]

            if len(sub_state) == 0:
                num = 0.0
                den = 0.0
                rate = 0.0
            else:
                num = float(sub_state["overspec_n_singleton"].sum())
                den = float(sub_state["singleton_total_n"].sum())
                rate = num / den if den > 0 else 0.0

            nums.append(num)
            dens.append(den)
            rates.append(rate)

        targets[noun] = np.array(rates, float)
        counts[noun] = np.array([[nums[0], dens[0]],
                                 [nums[1], dens[1]]], float)

    return targets, counts


MODEL_SPECS = {
    "mixture": {
        "param_names": ["alpha", "beta_fixed", "state_sem", "n_sem", "costWeight"],
        "bounds": [(0, 50), (0, 1), (0.6, 1), (0.6, 1), (0, 10)],
        "to_kwargs": lambda p: dict(
            model_type='mixture',
            alpha=float(p[0]),
            beta_fixed=float(p[1]),
            state_semvalue_marked=float(p[2]),
            state_semvalue_unmarked=float(p[2]),
            nominal_semvalue=float(p[3]),
            costWeight=float(p[4]),
        ),
    },

    "non-compositional": {
        "param_names": ["alpha", "costWeight"],
        "bounds": [(0, 50), (0, 10)],
        "to_kwargs": lambda p: dict(
            model_type='non-compositional',
            alpha=float(p[0]),
            costWeight=float(p[1]),
        ),
    },

    "compositional": {
        "param_names": ["alpha", "state_sem", "n_sem", "costWeight"],
        "bounds": [(0, 50), (0.6, 1), (0.6, 1), (0, 10)],
        "to_kwargs": lambda p: dict(
            model_type='compositional',
            alpha=float(p[0]),
            state_semvalue_marked=float(p[1]),
            state_semvalue_unmarked=float(p[1]),
            nominal_semvalue=float(p[2]),
            costWeight=float(p[3]),
        ),
    },

}

def optimization(model_type='compositional'):

    if model_type not in MODEL_SPECS:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         f"Available: {list(MODEL_SPECS.keys())}")

    spec = MODEL_SPECS[model_type]
    param_names = spec["param_names"]
    bounds = spec["bounds"]
    to_kwargs = spec["to_kwargs"]

    targets, counts = compute_targets(source='middle')
    words = list(targets.keys())

    def predict_for_word(params, word):
        """Shared prediction wrapper."""
        kwargs = to_kwargs(params)
        r_marked, r_unmarked = singleton_overspecification_rate(word=word, **kwargs)
        return np.array([r_marked, r_unmarked], float)

    def objective(params):
        losses = []
        for w in words:
            r = predict_for_word(params, w)
            t = np.array(targets[w], float)
            losses.append(np.sum((r - t) ** 2))

        # average loss across words
        L = float(np.sum(losses))
        return L if np.isfinite(L) else 1e9


    res_da = dual_annealing(objective, bounds=bounds, maxiter=200)  # smaller
    res_g  = minimize(objective, x0=res_da.x, method="L-BFGS-B", bounds=bounds)

    # best params
    best_params = res_g.x

    # recompute predictions at optimum
    preds = []
    t_list = []
    for w in words:
        preds.append(predict_for_word(best_params, w))
        t_list.append(targets[w])

    preds = np.array(preds, float)
    t_arr = np.array(t_list, float)

    avg_pred = preds.mean(axis=0)
    avg_target = t_arr.mean(axis=0)

    print("=== Result ===")
    print(f"  model_type           : {model_type}")
    for name, val in zip(param_names, best_params):
        print(f"  {name:20s} : {val:.4f}")
    print(f"  avg predicted rates  : [{avg_pred[0]:.4f}, {avg_pred[1]:.4f}]")
    print(f"  avg target rates     : [{avg_target[0]:.4f}, {avg_target[1]:.4f}]")
    print(f"  final avg loss       : {res_g.fun:.6e}")

    return res_g, preds, t_arr, words

if __name__ == "__main__":
    print(singleton_overspecification_rate('apple'))
    # optimized_params, predictions, targets, words = optimization('compositional')
    # optimized_params, predictions, targets, words = optimization('non-compositional')
    # optimized_params, predictions, targets, words = optimization('mixture')


    # utterance, world, noncomp_semvalue_dict = create_word_world("apple")
    # model = cs_rsa(world, noncomp_semvalue_dict)
    # singleton_overspecification_rate('apple')
    # words = df_words['noun'].unique()
    # optimization(words)

# aggregate the norming values
# choose different nouns: intuitive, dirty soccer ball
# plot noncompositional values for 100 nouns, 

# make the point that compositional modeling requires a different semantic values
# marked - unmarked, plot for 100 nouns, group based on modifiers, arrow point to 5 examples

# try compositional modeling first, doesn't work out because the relative markedness (lession from color doesn't generalize)

# 1. pure composition with same values
# 1.1 pure compositional with different sem for markedness, but fail for eye.

# 2. non-compositional, 
# degen works because they focus on modifers with contrasting typicality
# this cannot be generalized
# degen's framework suggests overspecification occurs for objects that are less typical exemplar of nouns. This is the not the correct overspecification. Hence we need a different generalization (which is not from language not typicality, not linguistic cost)

#part3
# Degen's goal was to arrive at a better fit for the empirical data. We believe however, Degen's mixture potentially provide insight about meaning. some meaning is know the meaning of the words (compositional), other meaning reflects the knowledge of the world (non-compositional). It could be a good idea but does not work.

# ELM: what overspecification of state modifers tell us about compositionality
# ask Harrison
