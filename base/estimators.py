import torch


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
        raise ValueError("alpha must be a scalar or a tensor with shape (num_states,)")
    return alpha


class BayesOptimalEstimator:
    """Bayes optimal next-state predictor for the finite mixture generator."""

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
                "distributions must have shape (num_distributions, num_states)"
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
                    "distribution_weights must have shape (num_distributions,)"
                )
        else:
            if zipf_exponent is None:
                raise ValueError(
                    "provide either distribution_weights or ranks with zipf_exponent"
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
                src=counts.new_ones(batch_size, 1),
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
                log_distributions = self.log_distributions[chunk_start:chunk_stop]
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


class DirichletEmpiricalEstimator:
    """Posterior predictive baseline using only empirical state counts."""

    def __init__(self, num_states, alpha, device=None, dtype=torch.float32):
        if num_states < 1:
            raise ValueError("num_states must be at least 1")
        self.num_states = num_states
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.alpha = make_alpha(alpha, num_states, self.device, dtype)

    @torch.no_grad()
    def predict_proba(self, input_ids):
        counts = counts_from_tokens(
            as_2d_tokens(input_ids).to(self.device),
            self.num_states,
        ).to(dtype=self.dtype)
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
                src=counts.new_ones(batch_size, 1),
            )
            target_counts = (counts + alpha).gather(1, targets[:, pos : pos + 1])
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
