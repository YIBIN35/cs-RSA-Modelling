import arviz as az
import numpy as np
import pymc as pm
from model_with_all_words import singleton_overspecification_rate, compute_targets, MODEL_SPECS
import matplotlib.pyplot as plt

def summarize_sample(sample):
    mean = sample.mean()
    lo, hi = np.percentile(sample, [2.5, 97.5])
    return mean, lo, hi

def bayes_r2_model_based(p_draws, n_trials, ddof=1):
    """
    model-based Bayesian R^2 for Binomial counts:
      y_i ~ Binomial(n_i, p_i)

    p_draws: (S, W) posterior draws of p_i
    n_trials:   (W,)  trials n_i
    Returns: (S,)  posterior draws of R^2

    Chose model based r2 because the n in each word trial is different.
    """
    p_draws = np.asarray(p_draws, dtype=float)     # (S, W)
    n_trials   = np.asarray(n_trials, dtype=float)       # (W,)

    mu = n_trials[None, :] * p_draws                  # (S, W)

    # explained variance across observations i (words)
    # ddof is the denom of n-1
    var_mu = np.var(mu, axis=1, ddof=ddof)         # (S,)

    # binomial based residual variance: E_i[ Var(y_i | theta) ] (np(1-p))
    var_y_given_theta = n_trials[None, :] * p_draws * (1.0 - p_draws)  # (S, W)
    var_res = np.mean(var_y_given_theta, axis=1)    # (S,)

    return var_mu / (var_mu + var_res) # (S,)

def r2_residual_based(y_true, y_pred, ddof=1):
    y_true = np.asarray(y_true, dtype=float)    # (W,)
    y_pred = np.asarray(y_pred, dtype=float)    # (S, W)

    var_y_pred = np.var(y_pred, axis=1, ddof=ddof)                  # (S,)
    var_resid  = np.var(y_true[None, :] - y_pred, axis=1, ddof=ddof) # (S,)
    return var_y_pred / (var_y_pred + var_resid)

# load observed data
targets, counts = compute_targets()
words = list(counts.keys())

# load posterior sampling data
model = 'mixture'
# model = 'non-compositional'
# model = 'compositional'
print(model)

idata = az.from_netcdf(f"trace_{model}.nc")
# idata = az.from_netcdf(f"trace_multiword.nc")
print(az.summary(idata, var_names=MODEL_SPECS[model]['param_names'], hdi_prob=0.95))

if model == 'mixture':
    beta_samples = idata.posterior["beta_fixed"].values.flatten()
    beta_mean, beta_lo, beta_hi = summarize_sample(beta_samples)
    print(f"beta_fixed mean = {beta_mean:.3f}, 95% CrI = [{beta_lo:.3f}, {beta_hi:.3f}]")


######################################################################
# posterior distributions

# observed overspec counts per word
y_marked   = idata.observed_data["y_marked"].values.astype(int)   # (W,)
y_unmarked = idata.observed_data["y_unmarked"].values.astype(int) # (W,)

# observed total counts of production
n_marked   = np.array([counts[w][0, 1] for w in words], dtype=int)
n_unmarked = np.array([counts[w][1, 1] for w in words], dtype=int)

# posterior p_marked: (chain, draw, word)
p_marked = (
    idata.posterior["p_marked"]
    .stack(sample=("chain", "draw"))
    .transpose("sample", "p_marked_dim_0")
    .values
)  # shape (S, W)

p_unmarked = (
    idata.posterior["p_unmarked"]
    .stack(sample=("chain", "draw"))
    .transpose("sample", "p_unmarked_dim_0")
    .values
)  # shape (S, W)

# predicted total counts across words for each posterior draw
y_marked_estimated   = (p_marked  * n_marked[None, :]).sum(axis=1)   # (S,)
y_unmarked_estimated = (p_unmarked * n_unmarked[None, :]).sum(axis=1) # (S,)

# total denominators across words
total_n_marked   = n_marked.sum()
total_n_unmarked = n_unmarked.sum()

# posterior over rates across all words
rate_marked   = y_marked_estimated   / total_n_marked
rate_unmarked = y_unmarked_estimated / total_n_unmarked

rm_mean, rm_lo, rm_hi = summarize_sample(rate_marked)
ru_mean, ru_lo, ru_hi = summarize_sample(rate_unmarked)

print("Posterior Marked:   mean {:.2f}, 95% CrI [{:.2f}, {:.2f}]"
      .format(rm_mean, rm_lo, rm_hi))
print("Posterior Unmarked: mean {:.2f}, 95% CrI [{:.2f}, {:.2f}]"
      .format(ru_mean, ru_lo, ru_hi))

print("\nPosterior means (per word):")
for i, w in enumerate(words):
    word_marked_mean   = p_marked[:, i].mean()
    word_unmarked_mean = p_unmarked[:, i].mean()

    print(
        f"{w:15s}  "
        f"p_marked={word_marked_mean:.3f}  "
        f"p_unmarked={word_unmarked_mean:.3f}"
    )

######################################################################
# posterior predictive
rng = np.random.default_rng(42)

y_marked_pp = rng.binomial(n_marked[None, :],   p_marked)
y_unmarked_pp = rng.binomial(n_unmarked[None, :], p_unmarked)

rate_marked_pp = y_marked_pp.sum(axis=1) / total_n_marked
rate_unmarked_pp = y_unmarked_pp.sum(axis=1) / total_n_unmarked

rm_pp_mean, rm_pp_lo, rm_pp_hi = summarize_sample(rate_marked_pp)
ru_pp_mean, ru_pp_lo, ru_pp_hi = summarize_sample(rate_unmarked_pp)

print("Posterior predictive — Marked:   mean {:.2f}, 95% CrI [{:.2f}, {:.2f}]"
      .format(rm_pp_mean, rm_pp_lo, rm_pp_hi))
print("Posterior predictive — Unmarked: mean {:.2f}, 95% CrI [{:.2f}, {:.2f}]"
      .format(ru_pp_mean, ru_pp_lo, ru_pp_hi))


# for each word
# posterior predictive counts per (sample, word): (S, W)
y_marked_pp_word   = rng.binomial(n=n_marked[None, :],   p=p_marked)
y_unmarked_pp_word = rng.binomial(n=n_unmarked[None, :], p=p_unmarked)

# posterior predictive rates per (sample, word): (S, W)
rate_marked_pp_word   = y_marked_pp_word   / n_marked[None, :]
rate_unmarked_pp_word = y_unmarked_pp_word / n_unmarked[None, :]

print("\nPosterior predictive (per word):")
for i, w in enumerate(words):  # or WORDS, but be consistent with how you built n_marked/n_unmarked
    rm_mean, rm_lo, rm_hi = summarize_sample(rate_marked_pp_word[:, i])
    ru_mean, ru_lo, ru_hi = summarize_sample(rate_unmarked_pp_word[:, i])

    print(
        f"{w:15s}  "
        f"pp_marked={rm_mean:.3f} [{rm_lo:.3f}, {rm_hi:.3f}]   "
        f"pp_unmarked={ru_mean:.3f} [{ru_lo:.3f}, {ru_hi:.3f}]"
    )


######################################################################
# R2


R2_marked   = bayes_r2_model_based(p_marked,   n_marked)
R2_unmarked = bayes_r2_model_based(p_unmarked, n_unmarked)

m, lo, hi = summarize_sample(R2_marked)
print(f"Bayesian R2 (marked):   mean={m:.3f}, 95% CrI=[{lo:.3f}, {hi:.3f}]")

m, lo, hi = summarize_sample(R2_unmarked)
print(f"Bayesian R2 (unmarked): mean={m:.3f}, 95% CrI=[{lo:.3f}, {hi:.3f}]")

