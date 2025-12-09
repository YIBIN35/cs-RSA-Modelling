from model_with_all_words import singleton_overspecification_rate, compute_targets, MODEL_SPECS
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import arviz as az

WORDS = []

def make_forward_all_words_op(model_type: str):
    spec = MODEL_SPECS[model_type]
    param_names = spec["param_names"]
    to_kwargs = spec["to_kwargs"]
    itypes = [pt.dscalar] * len(param_names)

    @as_op(itypes=itypes, otypes=[pt.dmatrix])
    def forward_all_words(*params):
        p = [float(v) for v in params]
        probs = []
        for w in WORDS:
            kwargs = to_kwargs(p)
            kwargs["word"] = str(w)
            r_marked, r_unmarked = singleton_overspecification_rate(**kwargs)
            probs.append([r_marked, r_unmarked])
        return np.array(probs, dtype=float)  # shape (W, 2)

    return forward_all_words

def fit_multiword_model(targets, counts, model_type="mixture", random_seed=42):
    global WORDS
    WORDS = list(targets.keys())

    y_marked   = np.array([counts[w][0, 0] for w in WORDS], dtype="int64")
    n_marked   = np.array([counts[w][0, 1] for w in WORDS], dtype="int64")
    y_unmarked = np.array([counts[w][1, 0] for w in WORDS], dtype="int64")
    n_unmarked = np.array([counts[w][1, 1] for w in WORDS], dtype="int64")

    spec = MODEL_SPECS[model_type]
    param_names = spec["param_names"]
    forward_all_words = make_forward_all_words_op(model_type=model_type)

    with pm.Model() as model_counts_multi:
        params = []

        for name in param_names:
            if name == "alpha":
                rv = pm.Uniform("alpha", lower=0.0, upper=50.0)

            elif name == "beta_fixed":
                rv = pm.Beta("beta_fixed", alpha=1.0, beta=1.0)

            elif name in ("state_sem", "n_sem"):
                rv = pm.Beta(name, alpha=1.0, beta=1.0)

            elif name == "costWeight":
                cw01 = pm.Beta("costWeight01", alpha=2.0, beta=2.0)
                rv = pm.Deterministic("costWeight", 3.0 * cw01)

            else:
                raise ValueError(f"Unknown parameter name in MODEL_SPECS: {name}")

            params.append(rv)

        rates = forward_all_words(*params)  # shape (W, 2)

        p_marked   = pm.Deterministic("p_marked",   rates[:, 0])
        p_unmarked = pm.Deterministic("p_unmarked", rates[:, 1])

        pm.Binomial("y_marked",   n=n_marked,   p=p_marked,   observed=y_marked)
        pm.Binomial("y_unmarked", n=n_unmarked, p=p_unmarked, observed=y_unmarked)

        trace = pm.sample(
            draws=50,
            tune=50,
            chains=4,
            cores=4,
            step=pm.Slice(),
            random_seed=random_seed,
            return_inferencedata=True,
        )

    return model_counts_multi, trace, WORDS, y_marked, y_unmarked


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="mixture",
        help="model_type passed into singleton_overspecification_rate (default: mixture)",
    )
    args = parser.parse_args()

    targets, counts = compute_targets()
    model, trace, words, y_marked, y_unmarked = fit_multiword_model(
        targets,
        counts,
        model_type=args.model_type,
        random_seed=42,
    )

    trace.to_netcdf(f"trace_{args.model_type}.nc")
    trace.to_netcdf("trace_multiword.nc")


    print(az.summary(trace, var_names=MODEL_SPECS[args.model_type]['param_names']))

    p_marked_post = trace.posterior["p_marked"]      # (chain, draw, word)
    p_unmarked_post = trace.posterior["p_unmarked"]  # (chain, draw, word)

    # mean over chains and draws
    p_marked_mean = p_marked_post.mean(dim=("chain", "draw")).values  # (word,)
    p_unmarked_mean = p_unmarked_post.mean(dim=("chain", "draw")).values

    for w, pmu, uu in zip(words, p_marked_mean, p_unmarked_mean):
        print(f"{w:15s}  p_marked={pmu:.3f}  p_unmarked={uu:.3f}")


    # posterior prediction
    with model:
        ppc = pm.sample_posterior_predictive(
            trace,
            var_names=["y_marked", "y_unmarked"],
            return_inferencedata=True,  # make it explicit
        )

    # xarray DataArrays
    y_marked_pp_da   = ppc.posterior_predictive["y_marked"]
    y_unmarked_pp_da = ppc.posterior_predictive["y_unmarked"]

    # total denominators from counts
    total_den_marked   = np.sum([counts[w][0, 1] for w in words])
    total_den_unmarked = np.sum([counts[w][1, 1] for w in words])

    # sum predictive counts across words
    dim_word_marked   = y_marked_pp_da.dims[-1]
    dim_word_unmarked = y_unmarked_pp_da.dims[-1]

    y_marked_sum   = y_marked_pp_da.sum(dim=dim_word_marked)
    y_unmarked_sum = y_unmarked_pp_da.sum(dim=dim_word_unmarked)

    # posterior predictive RATES
    rate_marked_pp   = y_marked_sum   / total_den_marked
    rate_unmarked_pp = y_unmarked_sum / total_den_unmarked

    # posterior means
    rate_marked_mean   = rate_marked_pp.mean(dim=("chain", "draw"))
    rate_unmarked_mean = rate_unmarked_pp.mean(dim=("chain", "draw"))

    # 95% HDIs
    rate_marked_hdi   = az.hdi(rate_marked_pp, hdi_prob=0.95)
    rate_unmarked_hdi = az.hdi(rate_unmarked_pp, hdi_prob=0.95)

    print("Marked rate mean:", float(rate_marked_mean))
    print("Marked rate HDI:", rate_marked_hdi)

    print("Unmarked rate mean:", float(rate_unmarked_mean))
    print("Unmarked rate HDI:", rate_unmarked_hdi)


















