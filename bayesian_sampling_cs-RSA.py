from updated_model import singleton_overspecification_rate
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import arviz as az


@as_op(
    itypes=[
        pt.dscalar,
        pt.dscalar,
        pt.dscalar,
        pt.dscalar,
        pt.dscalar,
    ],
    otypes=[pt.dvector],
)
def forward_rates(
    beta_fixed, s_marked, s_unmarked, costWeight, noncomp_semvalue_bare_unmarked
):
    r_marked, r_unmarked = singleton_overspecification_rate(
        beta_fixed=float(beta_fixed),
        state_semvalue_marked=float(s_marked),
        state_semvalue_unmarked=float(s_unmarked),
        costWeight=float(costWeight),
        noncomp_semvalue_bare_unmarked=float(noncomp_semvalue_bare_unmarked),
    )
    return np.array([r_marked, r_unmarked], dtype=float)


n_marked_total = 100
n_unmarked_total = 100
y_marked_open = int(round(0.24 * n_marked_total))
y_unmarked_close = int(round(0.01 * n_unmarked_total))

with pm.Model() as model_counts:

    # priors
    beta_fixed = pm.Beta("beta_fixed", alpha=1, beta=1)
    state_semvalue_marked = pm.Beta("state_semvalue_marked", alpha=1.0, beta=1.0)
    # state_semvalue_unmarked = pm.Beta("state_semvalue_unmarked", alpha=9.0,  beta=2.0)
    costWeight01 = pm.Beta("costWeight01", alpha=2, beta=2)
    costWeight = pm.Deterministic("costWeight", 3.0 * costWeight01)
    noncomp_semvalue_bare_unmarked = pm.Beta(
        "noncomp_semvalue_bare_unmarked", alpha=1.0, beta=1.0
    )

    # Binomial likelihood on counts based on rates
    rates = forward_rates(
        beta_fixed, 
        state_semvalue_marked, 
        state_semvalue_marked, 
        costWeight,
        noncomp_semvalue_bare_unmarked
    )
    p_marked_open = pm.Deterministic("p_marked_open", rates[0])
    p_unmarked_closed = pm.Deterministic("p_unmarked_closed", rates[1])

    pm.Binomial("y_marked", n=n_marked_total, p=p_marked_open, observed=y_marked_open)
    pm.Binomial(
        "y_unmarked", n=n_unmarked_total, p=p_unmarked_closed, observed=y_unmarked_close
    )

    # Sampling
    trace = pm.sample(
        draws=40000,
        tune=3000,
        chains=4,
        cores=4,
        step=pm.Slice(),
        random_seed=42,
        return_inferencedata=True,
    )

print(
    az.summary(
        trace,
        var_names=[
            "beta_fixed",
            "state_semvalue_marked",
            "costWeight",
            "noncomp_semvalue_bare_unmarked",
            "p_marked_open",
            "p_unmarked_closed",
        ],
    )
)

print(
    az.plot_posterior(
        trace,
        var_names=[
            "beta_fixed",
            "state_semvalue_marked",
            "costWeight",
            "noncomp_semvalue_bare_unmarked",
            "p_marked_open",
            "p_unmarked_closed",
        ],
    )
)
