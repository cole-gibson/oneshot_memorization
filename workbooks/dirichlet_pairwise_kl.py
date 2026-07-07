import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return np, pd


@app.cell
def _():
    num_distributions = 10_000
    num_categories = 1000
    alpha = 0.1
    random_seed = 1
    return alpha, num_categories, num_distributions, random_seed


@app.cell
def _(alpha, np, num_categories, num_distributions, random_seed):
    rng = np.random.default_rng(random_seed)
    concentration = np.full(num_categories, alpha)
    distributions = rng.dirichlet(concentration, size=num_distributions)
    return (distributions,)


@app.cell
def _(distributions, np):
    log_distributions = np.log(distributions)
    expected_log_self = (distributions * log_distributions).sum(axis=1)
    expected_log_other = distributions @ log_distributions.T
    pairwise_kl_matrix = expected_log_self[:, None] - expected_log_other
    off_diagonal_mask = ~np.eye(len(distributions), dtype=bool)
    directed_pairwise_kl_values = pairwise_kl_matrix[off_diagonal_mask]
    return (directed_pairwise_kl_values,)


@app.cell
def _(directed_pairwise_kl_values, np, pd):
    kl_statistics = pd.Series(
        {
            "count": directed_pairwise_kl_values.size,
            "mean": directed_pairwise_kl_values.mean(),
            "std": directed_pairwise_kl_values.std(ddof=1),
            "min": directed_pairwise_kl_values.min(),
            "p25": np.percentile(directed_pairwise_kl_values, 25),
            "median": np.median(directed_pairwise_kl_values),
            "p75": np.percentile(directed_pairwise_kl_values, 75),
            "max": directed_pairwise_kl_values.max(),
        },
        name="directed_pairwise_kl",
    )
    print(kl_statistics)
    return


if __name__ == "__main__":
    app.run()
