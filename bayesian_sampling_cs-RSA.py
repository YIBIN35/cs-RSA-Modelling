from model_with_all_words import singleton_overspecification_rate, compute_targets
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import arviz as az

WORDS = []

@as_op(
    itypes=[
        pt.dscalar,  # alpha
        pt.dscalar,  # beta_fixed
        pt.dscalar,  # state_sem
        pt.dscalar,  # nominal_sem
        pt.dscalar,  # costWeight
    ],
    otypes=[pt.dmatrix],
)
def forward_all_words(alpha, beta_fixed, state_sem, n_sem, costWeight):
    """
    Given shared parameters, compute [p_marked, p_unmarked] for all WORDS.
    Returns an array of shape (W, 2), where W = len(WORDS).
    """
    probs = []
    for w in WORDS:
        r_marked, r_unmarked = singleton_overspecification_rate(
            word=str(w),
            alpha=float(alpha),
            beta_fixed=float(beta_fixed),
            state_semvalue_marked=float(state_sem),
            state_semvalue_unmarked=float(state_sem),
            nominal_semvalue=float(n_sem),
            costWeight=float(costWeight),
        )
        probs.append([r_marked, r_unmarked])
    return np.array(probs, dtype=float)  # shape (W, 2)


# ---------- convert targets (probabilities) to counts ----------
def counts_from_targets(targets, n_per_word=8):
    """
    targets: dict[word] -> np.array([rate_state_A, rate_state_B])
    Converts each rate into Binomial counts with n = n_per_word.
    """
    words = list(targets.keys())
    target_mat = np.array([targets[w] for w in words], dtype=float)  # (W, 2)

    y_marked = np.round(target_mat[:, 0] * n_per_word).astype("int64")
    y_unmarked = np.round(target_mat[:, 1] * n_per_word).astype("int64")

    return words, y_marked, y_unmarked


# ---------- main model ----------
def fit_multiword_model(targets, n_per_word=8, random_seed=42):
    """
    targets: dict[word] -> np.array([rate_state_A, rate_state_B])
    n_per_word: Binomial n per word (assume 8 Bernoulli trials)
    """
    global WORDS

    WORDS, y_marked, y_unmarked = counts_from_targets(targets, n_per_word=n_per_word)

    with pm.Model() as model_counts_multi:

        # ----- Priors for shared parameters -----
        alpha = pm.HalfNormal("alpha", sigma=20.0)  # choose prior to taste
        beta_fixed = pm.Beta("beta_fixed", alpha=1.0, beta=1.0)
        state_semvalue = pm.Beta("state_semvalue", alpha=1.0, beta=1.0)
        nominal_semvalue = pm.Beta("nominal_semvalue", alpha=1.0, beta=1.0)

        costWeight01 = pm.Beta("costWeight01", alpha=2.0, beta=2.0)
        costWeight = pm.Deterministic("costWeight", 3.0 * costWeight01)

        # ----- forward model: one op for all words -----
        rates = forward_all_words(
            alpha,
            beta_fixed,
            state_semvalue,
            nominal_semvalue,
            costWeight,
        )  # shape (W, 2)

        p_marked = pm.Deterministic("p_marked", rates[:, 0])
        p_unmarked = pm.Deterministic("p_unmarked", rates[:, 1])

        # ----- Binomial likelihoods (one per word) -----
        pm.Binomial(
            "y_marked",
            n=n_per_word,
            p=p_marked,
            observed=y_marked,
        )
        pm.Binomial(
            "y_unmarked",
            n=n_per_word,
            p=p_unmarked,
            observed=y_unmarked,
        )

        # ----- sampling -----
        trace = pm.sample(
            draws=2000,
            tune=1000,
            chains=8,
            cores=8,
            step=pm.Slice(),
            random_seed=random_seed,
            return_inferencedata=True,
        )

    return model_counts_multi, trace, WORDS, y_marked, y_unmarked




if __name__ == "__main__":

    targets = compute_targets()
    model, trace, words, y_marked, y_unmarked = fit_multiword_model(
        targets,
        n_per_word=8,
        random_seed=42,
    )

    print(az.summary(trace, var_names=["alpha", "beta_fixed", "state_semvalue",
                                       "nominal_semvalue", "costWeight"]))


































# n_marked_total = 2000
# n_unmarked_total = 2000
# y_marked_open = int(round(0.24 * n_marked_total))
# y_unmarked_close = int(round(0.01 * n_unmarked_total))

# with pm.Model() as model_counts:

#     # Priors
#     beta_fixed = pm.Beta("beta_fixed", alpha=1, beta=1)
#     state_semvalue = pm.Beta("state_semvalue", alpha=1.0, beta=1.0)
#     # state_semvalue_unmarked = pm.Beta("state_semvalue_unmarked", alpha=9.0,  beta=2.0)
#     costWeight01 = pm.Beta("costWeight01", alpha=2, beta=2)
#     costWeight = pm.Deterministic("costWeight", 3.0 * costWeight01)
#     noncomp_semvalue_bare_unmarked = pm.Beta("noncomp_semvalue_bare_unmarked", alpha=1.0, beta=1.0)
#     typicalityWeight01 = pm.Beta("typicalityWeight01", alpha=2, beta=2)
#     typicalityWeight = pm.Deterministic("typicalityWeight", 3.0 * typicalityWeight01)

#     # Calculating Rates and Binomial likelihood on counts based on rates
#     rates = forward_rates(
#         beta_fixed, 
#         state_semvalue, 
#         costWeight,
#         noncomp_semvalue_bare_unmarked,
#         typicalityWeight,
#     )
#     p_marked_open = pm.Deterministic("p_marked_open", rates[0])
#     p_unmarked_closed = pm.Deterministic("p_unmarked_closed", rates[1])

#     pm.Binomial("y_marked", n=n_marked_total, p=p_marked_open, observed=y_marked_open)
#     pm.Binomial(
#         "y_unmarked", n=n_unmarked_total, p=p_unmarked_closed, observed=y_unmarked_close
#     )

#     # Sampling
#     trace = pm.sample(
#         draws=10000,
#         tune=3000,
#         chains=4,
#         cores=4,
#         step=pm.Slice(),
#         random_seed=42,
#         return_inferencedata=True,
#     )

# print(
#     az.summary(
#         trace,
#         var_names=[
#             "beta_fixed",
#             "state_semvalue",
#             "costWeight",
#             "noncomp_semvalue_bare_unmarked",
#             "typicalityWeight",
#             "p_marked_open",
#             "p_unmarked_closed",
#         ],
#     )
# )

# print(
#     az.plot_posterior(
#         trace,
#         var_names=[
#             "beta_fixed",
#             "state_semvalue",
#             "costWeight",
#             "noncomp_semvalue_bare_unmarked",
#             "typicalityWeight",
#             "p_marked_open",
#             "p_unmarked_closed",
#         ],
#     )
# )
