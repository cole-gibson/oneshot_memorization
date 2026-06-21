import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    from scipy.special import digamma

    import sys
    sys.path.append("..")

    from base.data_generator import DirichletZipfSequenceGenerator
    from estimators import BayesOptimalEstimator, DirichletEmpiricalEstimator

    return (
        BayesOptimalEstimator,
        DirichletEmpiricalEstimator,
        DirichletZipfSequenceGenerator,
        mo,
        torch,
    )


@app.cell
def _(torch):
    import torch.nn.functional as F
    import torch.nn as nn

    class UniMem(nn.Module):
        """Retrieval with likelihood computed from unigram statistics of the context (vectorized)."""
        def __init__(self, distributions: torch.Tensor, alpha: float, zipf_exponent: float, n_states: int):
            super().__init__()
            self.n_states = n_states
            self.log_probs = torch.log(distributions)   # K, C
            self.distributions = distributions
            self.n_distributions = distributions.shape[0]
            self.log_prior = -1 * zipf_exponent * torch.log(torch.arange(1, self.n_distributions + 1, device=distributions.device))  # K

        def forward(self, idx, targets, return_hatT = False, reduction = 'mean'):
            B, T = idx.shape
            seq = F.one_hot(idx, self.n_states).float()  # B, T, C
            cum_counts = torch.cumsum(seq, dim = 1) # B, T, C
            cum_loglikelihood = cum_counts @ self.log_probs.T     # B, T, C @ C, K -> B, T, K
            post = F.softmax(cum_loglikelihood + self.log_prior, dim = -1)   # constant prior can be omitted
            hatp = torch.einsum('ijk,kp-> ijp', post, self.distributions) # B, T, K @ K, C -> B, T, C
            logits = torch.log(hatp)
            loss = F.cross_entropy(logits.flatten(0,1), targets.flatten(0,1), reduction = reduction)
            if reduction == 'none':
                loss = loss.reshape(B, T)

            return logits, loss

    return (UniMem,)


@app.cell
def _(mo):
    mo.md("""
    ## Optimal estimators

    `BayesOptimalEstimator` uses the known component distributions and Zipf
    prior to compute the posterior mixture prediction for the next state.

    `DirichletEmpiricalEstimator` is a baseline that ignores the component
    set and uses the Dirichlet posterior predictive distribution from the
    empirical state counts.
    """)
    return


@app.cell
def _(
    BayesOptimalEstimator,
    DirichletEmpiricalEstimator,
    DirichletZipfSequenceGenerator,
    UniMem,
):
    alpha = 0.1
    num_states = 100
    sequence_length = 257
    num_distributions = 10_000

    generator = DirichletZipfSequenceGenerator(
        num_distributions=num_distributions,
        num_states=num_states,
        alpha=alpha,
        zipf_exponent=1.0,
    )
    tokens = generator.sample(batch_size=2**12, sequence_length=sequence_length)

    uni_mem = UniMem(
        distributions=generator.distributions,
        alpha=alpha,
        zipf_exponent=1.0,
        n_states=num_states,
    )

    bayes = BayesOptimalEstimator.from_generator(generator)
    empirical = DirichletEmpiricalEstimator(num_states=num_states, alpha=alpha)

    # track runtimes for each estimator
    from time import time

    start_time = time()
    bayes_loss = bayes.autoregressive_loss(tokens, component_chunk_size=1024)
    bayes_time = time() - start_time
    empirical_loss = empirical.autoregressive_loss(tokens)
    empirical_time = time() - start_time - bayes_time
    # _, my_loss = uni_mem(tokens[:, :-1], tokens[:, 1:])
    # uni_mem_time = time() - start_time - bayes_time - empirical_time

    print("Bayes optimal loss:", bayes_loss.item(), "computed in", bayes_time, "seconds")
    print("Dirichlet empirical loss:", empirical_loss.item(), "computed in", empirical_time, "seconds")
    # print("UniMem loss:", my_loss.mean().item(), "computed in", uni_mem_time, "seconds")
    return


if __name__ == "__main__":
    app.run()
