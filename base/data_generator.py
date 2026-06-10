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

        concentration = self._make_concentration(alpha)
        self.distributions = torch.distributions.Dirichlet(concentration).sample(
            (num_distributions,)
        )

        ranks = torch.arange(1, num_distributions + 1, device=self.device, dtype=dtype)
        self.distribution_weights = ranks.pow(-zipf_exponent)
        self.distribution_weights /= self.distribution_weights.sum()

    def _make_concentration(self, alpha):
        alpha = torch.as_tensor(alpha, device=self.device, dtype=self.dtype)
        if torch.any(alpha <= 0):
            raise ValueError("alpha must be positive")

        if alpha.ndim == 0:
            return alpha.expand(self.num_states)
        if alpha.shape != (self.num_states,):
            raise ValueError("alpha must be a scalar or a tensor with shape (num_states,)")
        return alpha

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
