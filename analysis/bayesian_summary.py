import arviz as az
import numpy as np
import pymc as pm
from model_with_all_words import singleton_overspecification_rate, compute_targets, MODEL_SPECS
import matplotlib.pyplot as plt

def summarize_sample(sample):
    mean = sample.mean()
    lo, hi = np.percentile(sample, [2.5, 97.5])
    return mean, lo, hi

# load observed data
targets, counts = compute_targets()
words = list(counts.keys())

# load posterior sampling data
# model = 'mixture'
model = 'non-compositional'
# model = 'compositional'
print(model)

idata = az.from_netcdf(f"trace_{model}.nc")
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

print("Posterior Marked:   mean {:.3f}, 95% CrI [{:.3f}, {:.3f}]"
      .format(rm_mean, rm_lo, rm_hi))
print("Posterior Unmarked: mean {:.3f}, 95% CrI [{:.3f}, {:.3f}]"
      .format(ru_mean, ru_lo, ru_hi))


######################################################################
# posterior predictive
rng = np.random.default_rng(42)

y_marked_pp = rng.binomial(n_marked[None, :],   p_marked)
y_unmarked_pp = rng.binomial(n_unmarked[None, :], p_unmarked)

rate_marked_pp = y_marked_pp.sum(axis=1) / total_n_marked
rate_unmarked_pp = y_unmarked_pp.sum(axis=1) / total_n_unmarked

rm_pp_mean, rm_pp_lo, rm_pp_hi = summarize_sample(rate_marked_pp)
ru_pp_mean, ru_pp_lo, ru_pp_hi = summarize_sample(rate_unmarked_pp)

print("Posterior predictive — Marked:   mean {:.3f}, 95% CrI [{:.3f}, {:.3f}]"
      .format(rm_pp_mean, rm_pp_lo, rm_pp_hi))
print("Posterior predictive — Unmarked: mean {:.3f}, 95% CrI [{:.3f}, {:.3f}]"
      .format(ru_pp_mean, ru_pp_lo, ru_pp_hi))
