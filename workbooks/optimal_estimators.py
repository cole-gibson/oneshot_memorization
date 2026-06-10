import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch

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
def _(counts_from_tokens, torch):
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

        __call__ = predict_proba

        def to(self, device):
            self.device = torch.device(device)
            self.alpha = self.alpha.to(self.device)
            return self

    return (DirichletEmpiricalEstimator,)


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
    mo,
):
    generator = DirichletZipfSequenceGenerator(
        num_distributions=10_000,
        num_states=100,
        alpha=0.1,
        zipf_exponent=1.0,
    )
    tokens = generator.sample(batch_size=256, sequence_length=129)

    bayes = BayesOptimalEstimator.from_generator(generator)
    empirical = DirichletEmpiricalEstimator(num_states=100, alpha=1.0)

    bayes_probs = bayes.predict_proba(tokens[:, :-1])
    empirical_probs = empirical.predict_proba(tokens[:, :-1])

    targets = tokens[:, -1]
    bayes_loss = -bayes_probs.gather(1, targets[:, None]).clamp_min(1e-30).log().mean()
    empirical_loss = (
        -empirical_probs.gather(1, targets[:, None]).clamp_min(1e-30).log().mean()
    )

    mo.md(
        f"""
        Next-state loss on the sampled batch:
        - Bayes optimal: `{bayes_loss.item():.6f}`
        - Dirichlet empirical baseline: `{empirical_loss.item():.6f}`
        """
    )
    return


if __name__ == "__main__":
    app.run()
