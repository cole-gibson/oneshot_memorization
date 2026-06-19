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

    return DirichletZipfSequenceGenerator, mo, torch


@app.cell
def _(torch):
    def as_2d_tokens(input_ids):
        if input_ids.ndim == 1:
            return input_ids.unsqueeze(0)
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape (seq_len,) or (batch_size, seq_len)"
            )
        return input_ids

    def counts_from_tokens(input_ids, num_states):
        input_ids = as_2d_tokens(input_ids)
        if input_ids.numel() == 0:
            return torch.zeros(
                input_ids.shape[0],
                num_states,
                device=input_ids.device,
                dtype=torch.float32,
            )
        if input_ids.min() < 0 or input_ids.max() >= num_states:
            raise ValueError("input_ids contain states outside [0, num_states)")

        counts = torch.zeros(
            input_ids.shape[0],
            num_states,
            device=input_ids.device,
            dtype=torch.float32,
        )
        return counts.scatter_add(
            dim=1,
            index=input_ids,
            src=torch.ones_like(input_ids, dtype=counts.dtype),
        )

    def make_alpha(alpha, num_states, device, dtype):
        alpha = torch.as_tensor(alpha, device=device, dtype=dtype)
        if torch.any(alpha <= 0):
            raise ValueError("alpha must be positive")
        if alpha.ndim == 0:
            return alpha.expand(num_states)
        if alpha.shape != (num_states,):
            raise ValueError(
                "alpha must be a scalar or a tensor with shape (num_states,)"
            )
        return alpha

    return as_2d_tokens, counts_from_tokens, make_alpha


@app.cell
def _(as_2d_tokens, counts_from_tokens, torch):
    class BayesOptimalEstimator:
        """Bayes optimal next-state predictor for the finite mixture generator.

        Given a prefix sequence and the known component distributions, this
        infers the posterior over components and returns the posterior
        predictive distribution over the next state.
        """

        def __init__(
            self,
            distributions,
            distribution_weights=None,
            ranks=None,
            zipf_exponent=None,
            eps=1e-30,
        ):
            if distributions.ndim != 2:
                raise ValueError(
                    "distributions must have shape "
                    "(num_distributions, num_states)"
                )
            if torch.any(distributions < 0):
                raise ValueError("distributions must be non-negative")

            row_sums = distributions.sum(dim=1)
            if torch.any(row_sums <= 0):
                raise ValueError("each distribution must have positive mass")

            self.distributions = distributions / row_sums[:, None]
            self.num_distributions, self.num_states = self.distributions.shape
            self.eps = eps

            prior = self._make_prior(distribution_weights, ranks, zipf_exponent)
            self.distribution_weights = prior / prior.sum()
            self.log_prior = self.distribution_weights.clamp_min(eps).log()
            self.log_distributions = self.distributions.clamp_min(eps).log()

        @classmethod
        def from_generator(cls, data_generator):
            return cls(
                distributions=data_generator.distributions,
                distribution_weights=data_generator.distribution_weights,
            )

        def _make_prior(self, distribution_weights, ranks, zipf_exponent):
            if distribution_weights is not None:
                prior = torch.as_tensor(
                    distribution_weights,
                    device=self.distributions.device,
                    dtype=self.distributions.dtype,
                )
                if prior.shape != (self.num_distributions,):
                    raise ValueError(
                        "distribution_weights must have shape "
                        "(num_distributions,)"
                    )
            else:
                if zipf_exponent is None:
                    raise ValueError(
                        "provide either distribution_weights or ranks with "
                        "zipf_exponent"
                    )
                if ranks is None:
                    ranks = torch.arange(
                        1,
                        self.num_distributions + 1,
                        device=self.distributions.device,
                        dtype=self.distributions.dtype,
                    )
                else:
                    ranks = torch.as_tensor(
                        ranks,
                        device=self.distributions.device,
                        dtype=self.distributions.dtype,
                    )
                if ranks.shape != (self.num_distributions,):
                    raise ValueError("ranks must have shape (num_distributions,)")
                if torch.any(ranks <= 0):
                    raise ValueError("ranks must be positive")
                prior = ranks.pow(-float(zipf_exponent))

            if torch.any(prior < 0) or prior.sum() <= 0:
                raise ValueError(
                    "prior weights must be non-negative with positive total mass"
                )
            return prior

        @torch.no_grad()
        def posterior_over_components(self, input_ids):
            counts = counts_from_tokens(input_ids, self.num_states).to(
                device=self.distributions.device,
                dtype=self.distributions.dtype,
            )
            log_posterior = counts @ self.log_distributions.T
            log_posterior = log_posterior + self.log_prior
            return torch.softmax(log_posterior, dim=-1)

        @torch.no_grad()
        def predict_proba(self, input_ids):
            posterior = self.posterior_over_components(input_ids)
            return posterior @ self.distributions

        @torch.no_grad()
        def predict_log_proba(self, input_ids):
            return self.predict_proba(input_ids).clamp_min(self.eps).log()

        @torch.no_grad()
        def autoregressive_losses(self, tokens, component_chunk_size=None):
            tokens = as_2d_tokens(tokens).to(self.distributions.device)
            if tokens.shape[1] < 2:
                raise ValueError("tokens must contain at least two positions")

            input_ids = tokens[:, :-1]
            targets = tokens[:, 1:]
            batch_size, input_len = input_ids.shape
            if component_chunk_size is None:
                component_chunk_size = self.num_distributions
            component_chunk_size = int(component_chunk_size)
            if component_chunk_size < 1:
                raise ValueError("component_chunk_size must be at least 1")

            counts = torch.zeros(
                batch_size,
                self.num_states,
                device=self.distributions.device,
                dtype=self.distributions.dtype,
            )
            losses = []

            for pos in range(input_len):
                counts.scatter_add_(
                    dim=1,
                    index=input_ids[:, pos : pos + 1],
                    src=torch.ones(
                        batch_size,
                        1,
                        device=self.distributions.device,
                        dtype=self.distributions.dtype,
                    ),
                )
                target = targets[:, pos]
                log_denominator = torch.full(
                    (batch_size,),
                    float("-inf"),
                    device=self.distributions.device,
                    dtype=self.distributions.dtype,
                )
                log_numerator = torch.full_like(log_denominator, float("-inf"))

                for chunk_start in range(
                    0,
                    self.num_distributions,
                    component_chunk_size,
                ):
                    chunk_stop = min(
                        self.num_distributions,
                        chunk_start + component_chunk_size,
                    )
                    log_distributions = self.log_distributions[
                        chunk_start:chunk_stop
                    ]
                    component_scores = (
                        counts @ log_distributions.T
                        + self.log_prior[chunk_start:chunk_stop]
                    )
                    target_log_probs = log_distributions.T[target]
                    log_denominator = torch.logaddexp(
                        log_denominator,
                        torch.logsumexp(component_scores, dim=1),
                    )
                    log_numerator = torch.logaddexp(
                        log_numerator,
                        torch.logsumexp(component_scores + target_log_probs, dim=1),
                    )

                losses.append(-(log_numerator - log_denominator))

            return torch.stack(losses, dim=1).mean(dim=1)

        @torch.no_grad()
        def autoregressive_loss(self, tokens, component_chunk_size=None):
            return self.autoregressive_losses(
                tokens,
                component_chunk_size=component_chunk_size,
            ).mean()

        __call__ = predict_proba

        def to(self, device):
            device = torch.device(device)
            self.distributions = self.distributions.to(device)
            self.distribution_weights = self.distribution_weights.to(device)
            self.log_prior = self.log_prior.to(device)
            self.log_distributions = self.log_distributions.to(device)
            return self

    return (BayesOptimalEstimator,)


@app.cell
def _(as_2d_tokens, make_alpha, torch):
    class DirichletEmpiricalEstimator:
        """Posterior predictive baseline using only empirical state counts.

        This ignores the known finite component set. It treats each observed
        prefix as samples from an unknown categorical distribution with a
        Dirichlet prior concentration ``alpha``.
        """

        def __init__(self, num_states, alpha, device=None, dtype=torch.float32):
            if num_states < 1:
                raise ValueError("num_states must be at least 1")
            self.num_states = num_states
            self.device = (
                torch.device(device) if device is not None else torch.device("cpu")
            )
            self.dtype = dtype
            self.alpha = make_alpha(alpha, num_states, self.device, dtype)

        @torch.no_grad()
        def predict_proba(self, input_ids):
            input_ids = as_2d_tokens(input_ids)
            if input_ids.numel() > 0:
                if input_ids.min() < 0 or input_ids.max() >= self.num_states:
                    raise ValueError("input_ids contain states outside [0, num_states)")

            counts = torch.zeros(
                input_ids.shape[0],
                self.num_states,
                device=self.device,
                dtype=self.dtype,
            )
            if input_ids.numel() > 0:
                counts.scatter_add_(
                    dim=1,
                    index=input_ids.to(self.device),
                    src=torch.ones(
                        input_ids.shape,
                        device=self.device,
                        dtype=self.dtype,
                    ),
                )

            posterior_counts = counts + self.alpha
            return posterior_counts / posterior_counts.sum(dim=-1, keepdim=True)

        @torch.no_grad()
        def predict_log_proba(self, input_ids):
            return self.predict_proba(input_ids).log()

        @torch.no_grad()
        def autoregressive_losses(self, tokens, eps=1e-30):
            tokens = as_2d_tokens(tokens).to(self.device)
            if tokens.shape[1] < 2:
                raise ValueError("tokens must contain at least two positions")

            input_ids = tokens[:, :-1]
            targets = tokens[:, 1:]
            batch_size, input_len = input_ids.shape
            counts = torch.zeros(
                batch_size,
                self.num_states,
                device=self.device,
                dtype=self.dtype,
            )
            alpha = self.alpha.unsqueeze(0)
            alpha_sum = self.alpha.sum()
            losses = []

            for pos in range(input_len):
                counts.scatter_add_(
                    dim=1,
                    index=input_ids[:, pos : pos + 1],
                    src=torch.ones(
                        batch_size,
                        1,
                        device=self.device,
                        dtype=self.dtype,
                    ),
                )
                target_counts = (counts + alpha).gather(
                    1,
                    targets[:, pos : pos + 1],
                )
                denominator = alpha_sum + float(pos + 1)
                losses.append(
                    -target_counts.div(denominator).clamp_min(eps).log().squeeze(1)
                )

            return torch.stack(losses, dim=1).mean(dim=1)

        @torch.no_grad()
        def autoregressive_loss(self, tokens, eps=1e-30):
            return self.autoregressive_losses(tokens, eps=eps).mean()

        __call__ = predict_proba

        def to(self, device):
            self.device = torch.device(device)
            self.alpha = self.alpha.to(self.device)
            return self

    return (DirichletEmpiricalEstimator,)


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
