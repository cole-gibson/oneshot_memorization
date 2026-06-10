import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import marimo as mo
    import torch

    from base.data_generator import DirichletZipfSequenceGenerator

    return DirichletZipfSequenceGenerator, mo, plt, torch


@app.cell
def _(mo):
    num_distributions = mo.ui.slider(
        start=5,
        stop=200,
        step=5,
        value=50,
        label="components",
    )
    batch_size = mo.ui.slider(
        start=100,
        stop=100_000,
        step=100,
        value=10_000,
        label="batch size",
    )
    zipf_exponent = mo.ui.slider(
        start=0.0,
        stop=3.0,
        step=0.1,
        value=1.2,
        label="zipf exponent",
    )
    seed = mo.ui.number(value=0, start=0, stop=1_000_000, step=1, label="seed")

    mo.vstack(
        [
            mo.md("## Zipf component sampling check"),
            mo.hstack([num_distributions, batch_size]),
            mo.hstack([zipf_exponent, seed]),
        ]
    )
    return batch_size, num_distributions, seed, zipf_exponent


@app.cell
def _(
    DirichletZipfSequenceGenerator,
    batch_size,
    num_distributions,
    seed,
    torch,
    zipf_exponent,
):
    rng = torch.Generator().manual_seed(int(seed.value))
    data_generator = DirichletZipfSequenceGenerator(
        num_distributions=int(num_distributions.value),
        num_states=32,
        alpha=0.5,
        zipf_exponent=float(zipf_exponent.value),
        generator=rng,
    )

    _, component_ids = data_generator.sample(
        batch_size=int(batch_size.value),
        sequence_length=8,
        return_distribution_ids=True,
    )
    empirical = torch.bincount(
        component_ids,
        minlength=data_generator.num_distributions,
    ).to(torch.float32)
    empirical /= empirical.sum()

    expected = data_generator.distribution_weights.cpu()
    ranks = torch.arange(1, data_generator.num_distributions + 1)
    return empirical, expected, ranks


@app.cell
def _(empirical, expected, mo, plt, ranks):
    max_abs_error = (empirical - expected).abs().max().item()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(
        ranks.numpy(),
        empirical.numpy(),
        color="#4f7cac",
        alpha=0.75,
        label="empirical",
    )
    ax.plot(
        ranks.numpy(),
        expected.numpy(),
        color="#c2410c",
        linewidth=2.25,
        label="expected Zipf",
    )
    ax.set_xlabel("component rank")
    ax.set_ylabel("probability")
    ax.set_title("Empirical distribution over sampled components")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()

    mo.vstack(
        [
            mo.md(f"Max absolute error: `{max_abs_error:.4f}`"),
            fig,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
