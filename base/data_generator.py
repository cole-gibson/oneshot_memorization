import torch


class DirichletZipfSequenceGenerator:
    """Generate token sequences from Zipf-selected Dirichlet components.

    Each component is a categorical distribution over ``num_states`` states.
    For every sequence in a batch, the component id is sampled from a Zipf-like
    distribution over component ranks, then all states in that sequence are
    sampled independently from the chosen component.
    """

    def __init__(
        self,
        num_distributions,
        num_states,
        alpha,
        zipf_exponent,
        device=None,
        dtype=torch.float32,
        generator=None,
        distributions=None,
    ):
        if num_distributions < 1:
            raise ValueError("num_distributions must be at least 1")
        if num_states < 1:
            raise ValueError("num_states must be at least 1")
        if zipf_exponent < 0:
            raise ValueError("zipf_exponent must be non-negative")

        self.num_distributions = num_distributions
        self.num_states = num_states
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.generator = generator

        if distributions is None:
            concentration = self._make_concentration(alpha)
            self.distributions = self._sample_dirichlet(concentration)
        else:
            self.distributions = self._make_distributions(distributions)

        ranks = torch.arange(1, num_distributions + 1, device=self.device, dtype=dtype)
        self.distribution_weights = ranks.pow(-zipf_exponent)
        self.distribution_weights /= self.distribution_weights.sum()

    def _sample_dirichlet(self, concentration):
        gamma_samples = torch._standard_gamma(
            concentration.expand(self.num_distributions, self.num_states),
            generator=self.generator,
        )
        return gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)

    def _make_concentration(self, alpha):
        alpha = torch.as_tensor(alpha, device=self.device, dtype=self.dtype)
        if torch.any(alpha <= 0):
            raise ValueError("alpha must be positive")

        if alpha.ndim == 0:
            return alpha.expand(self.num_states)
        if alpha.shape != (self.num_states,):
            raise ValueError("alpha must be a scalar or a tensor with shape (num_states,)")
        return alpha

    def _make_distributions(self, distributions):
        distributions = torch.as_tensor(
            distributions,
            device=self.device,
            dtype=self.dtype,
        )
        if distributions.shape != (self.num_distributions, self.num_states):
            raise ValueError(
                "distributions must have shape "
                "(num_distributions, num_states)"
            )
        if torch.any(distributions < 0):
            raise ValueError("distributions must be non-negative")
        row_sums = distributions.sum(dim=1, keepdim=True)
        if torch.any(row_sums <= 0):
            raise ValueError("each distribution must have positive mass")
        return distributions / row_sums

    @torch.no_grad()
    def sample(self, batch_size, sequence_length, return_distribution_ids=False):
        """Return a LongTensor of shape ``(batch_size, sequence_length)``.

        If ``return_distribution_ids`` is true, also returns the component ids
        used for each sequence as a LongTensor of shape ``(batch_size,)``.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")

        distribution_ids = torch.multinomial(
            self.distribution_weights,
            num_samples=batch_size,
            replacement=True,
            generator=self.generator,
        )
        selected_distributions = self.distributions[distribution_ids]
        flat_samples = torch.multinomial(
            selected_distributions,
            num_samples=sequence_length,
            replacement=True,
            generator=self.generator,
        )

        if return_distribution_ids:
            return flat_samples, distribution_ids
        return flat_samples

    @torch.no_grad()
    def sample_from_distribution_ids(self, distribution_ids, sequence_length):
        """Sample sequences from explicitly supplied component ids.

        ``distribution_ids`` is a one-dimensional tensor-like object with one
        component id per requested sequence. The returned LongTensor has shape
        ``(len(distribution_ids), sequence_length)``.
        """
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")

        distribution_ids = torch.as_tensor(distribution_ids, device=self.device)
        if distribution_ids.ndim != 1:
            raise ValueError("distribution_ids must have shape (batch_size,)")
        if distribution_ids.numel() == 0:
            raise ValueError("distribution_ids must contain at least one id")
        if not torch.is_floating_point(distribution_ids):
            distribution_ids = distribution_ids.to(dtype=torch.long)
        else:
            if not torch.all(distribution_ids == distribution_ids.long()):
                raise ValueError("distribution_ids must contain integer ids")
            distribution_ids = distribution_ids.to(dtype=torch.long)
        if (
            distribution_ids.min() < 0
            or distribution_ids.max() >= self.num_distributions
        ):
            raise ValueError(
                "distribution_ids contain ids outside "
                f"[0, {self.num_distributions})"
            )

        selected_distributions = self.distributions[distribution_ids]
        return torch.multinomial(
            selected_distributions,
            num_samples=sequence_length,
            replacement=True,
            generator=self.generator,
        )

    __call__ = sample

    def to(self, device):
        """Move the generator's cached tensors to ``device``."""
        device = torch.device(device)
        self.device = device
        self.distributions = self.distributions.to(device)
        self.distribution_weights = self.distribution_weights.to(device)
        return self


if __name__ == "__main__":
    generator = DirichletZipfSequenceGenerator(
        num_distributions=100,
        num_states=128,
        alpha=0.1,
        zipf_exponent=1.2,
    )
    sequences, distribution_ids = generator.sample(
        batch_size=32,
        sequence_length=64,
        return_distribution_ids=True,
    )
    print(sequences.shape)
    print(distribution_ids.shape)
