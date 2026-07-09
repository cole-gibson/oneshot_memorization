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


class DistributionPosterior:
    """Posterior over Dirichlet-Zipf mixture components."""

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

    def to(self, device):
        device = torch.device(device)
        self.distributions = self.distributions.to(device)
        self.distribution_weights = self.distribution_weights.to(device)
        self.log_prior = self.log_prior.to(device)
        self.log_distributions = self.log_distributions.to(device)
        return self


class BayesOptimalDistributionLabelClassifier(DistributionPosterior):
    """Bayes optimal classifier for labels attached to components."""

    def __init__(
        self,
        distributions,
        distribution_labels,
        distribution_weights=None,
        ranks=None,
        zipf_exponent=None,
        num_classes=None,
        eps=1e-30,
    ):
        super().__init__(
            distributions=distributions,
            distribution_weights=distribution_weights,
            ranks=ranks,
            zipf_exponent=zipf_exponent,
            eps=eps,
        )
        distribution_labels = torch.as_tensor(
            distribution_labels,
            device=self.distributions.device,
        )
        if distribution_labels.shape != (self.num_distributions,):
            raise ValueError(
                "distribution_labels must have shape (num_distributions,)"
            )
        if distribution_labels.min() < 0:
            raise ValueError("distribution_labels must be non-negative")
        self.distribution_labels = distribution_labels.to(dtype=torch.long)
        if num_classes is None:
            num_classes = int(self.distribution_labels.max().item()) + 1
        if num_classes < 1:
            raise ValueError("num_classes must be at least 1")
        if self.distribution_labels.max() >= num_classes:
            raise ValueError("distribution_labels must be less than num_classes")
        self.num_classes = int(num_classes)

    @classmethod
    def from_generator(cls, data_generator):
        return cls(
            distributions=data_generator.distributions,
            distribution_labels=data_generator.distribution_labels,
            distribution_weights=data_generator.distribution_weights,
            num_classes=2,
        )

    @torch.no_grad()
    def predict_proba(self, input_ids):
        posterior = self.posterior_over_components(input_ids)
        labels = self.distribution_labels.unsqueeze(0).expand(posterior.shape[0], -1)
        probabilities = posterior.new_zeros(posterior.shape[0], self.num_classes)
        return probabilities.scatter_add(1, labels, posterior)

    @torch.no_grad()
    def predict(self, input_ids):
        return self.predict_proba(input_ids).argmax(dim=1)

    @torch.no_grad()
    def losses(self, tokens, targets):
        log_proba = self.predict_proba(tokens).clamp_min(self.eps).log()
        targets = targets.to(device=log_proba.device, dtype=torch.long)
        return -log_proba.gather(1, targets[:, None]).squeeze(1)

    def to(self, device):
        super().to(device)
        self.distribution_labels = self.distribution_labels.to(self.distributions.device)
        return self
