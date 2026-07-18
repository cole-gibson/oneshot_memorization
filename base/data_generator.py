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


class DirichletZipfBinaryClassificationGenerator(DirichletZipfSequenceGenerator):
    """Generate sequence-label pairs from Zipf-selected Dirichlet components.

    Each component receives a fixed binary label. A sampled sequence inherits
    the label of the component that generated its states.
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
        distribution_labels=None,
    ):
        super().__init__(
            num_distributions=num_distributions,
            num_states=num_states,
            alpha=alpha,
            zipf_exponent=zipf_exponent,
            device=device,
            dtype=dtype,
            generator=generator,
            distributions=distributions,
        )
        if distribution_labels is None:
            self.distribution_labels = torch.randint(
                0,
                2,
                (self.num_distributions,),
                device=self.device,
                generator=self.generator,
            )
        else:
            self.distribution_labels = self._make_distribution_labels(
                distribution_labels
            )

    def _make_distribution_labels(self, distribution_labels):
        distribution_labels = torch.as_tensor(
            distribution_labels,
            device=self.device,
        )
        if distribution_labels.shape != (self.num_distributions,):
            raise ValueError(
                "distribution_labels must have shape (num_distributions,)"
            )
        if distribution_labels.min() < 0 or distribution_labels.max() > 1:
            raise ValueError("distribution_labels must contain only binary labels")
        return distribution_labels.to(dtype=torch.long)

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        sequence_length,
        return_distribution_ids=False,
        return_labels=False,
    ):
        tokens, distribution_ids = super().sample(
            batch_size=batch_size,
            sequence_length=sequence_length,
            return_distribution_ids=True,
        )
        labels = self.distribution_labels[distribution_ids]

        if return_distribution_ids and return_labels:
            return tokens, distribution_ids, labels
        if return_distribution_ids:
            return tokens, distribution_ids
        if return_labels:
            return tokens, labels
        return tokens

    @torch.no_grad()
    def sample_from_distribution_ids(
        self,
        distribution_ids,
        sequence_length,
        return_labels=False,
    ):
        tokens = super().sample_from_distribution_ids(
            distribution_ids=distribution_ids,
            sequence_length=sequence_length,
        )
        if return_labels:
            distribution_ids = torch.as_tensor(
                distribution_ids,
                device=self.device,
                dtype=torch.long,
            )
            return tokens, self.distribution_labels[distribution_ids]
        return tokens

    def to(self, device):
        super().to(device)
        self.distribution_labels = self.distribution_labels.to(self.device)
        return self


class DirichletZipfBinaryProbabilityVectorGenerator(
    DirichletZipfBinaryClassificationGenerator
):
    """Generate probability-vector/label pairs from Zipf-selected components."""

    def __init__(self, *args, noise_enabled=False, noise_intensity=1.0, **kwargs):
        if not isinstance(noise_enabled, bool):
            raise ValueError("noise_enabled must be a boolean")
        if (
            not isinstance(noise_intensity, (int, float))
            or isinstance(noise_intensity, bool)
            or noise_intensity < 0
        ):
            raise ValueError("noise_intensity must be a nonnegative number")
        self.noise_enabled = noise_enabled
        self.noise_intensity = float(noise_intensity)
        super().__init__(*args, **kwargs)

    def _add_noise(self, probabilities):
        standard_normal = torch.randn(
            probabilities.shape,
            device=self.device,
            dtype=self.dtype,
            generator=self.generator,
        )
        scaled_normal = probabilities.sqrt() * standard_normal
        noise = scaled_normal - probabilities * scaled_normal.sum(
            dim=-1, keepdim=True
        )
        return probabilities + self.noise_intensity * noise

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        return_distribution_ids=False,
        return_labels=False,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        distribution_ids = torch.multinomial(
            self.distribution_weights,
            num_samples=batch_size,
            replacement=True,
            generator=self.generator,
        )
        probabilities = self.distributions[distribution_ids]
        if self.noise_enabled:
            probabilities = self._add_noise(probabilities)
        labels = self.distribution_labels[distribution_ids]

        if return_distribution_ids and return_labels:
            return probabilities, distribution_ids, labels
        if return_distribution_ids:
            return probabilities, distribution_ids
        if return_labels:
            return probabilities, labels
        return probabilities

    @torch.no_grad()
    def sample_from_distribution_ids(
        self,
        distribution_ids,
        return_labels=False,
    ):
        distribution_ids = torch.as_tensor(distribution_ids, device=self.device)
        if distribution_ids.ndim != 1:
            raise ValueError("distribution_ids must have shape (batch_size,)")
        if distribution_ids.numel() == 0:
            raise ValueError("distribution_ids must contain at least one id")
        if torch.is_floating_point(distribution_ids):
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

        probabilities = self.distributions[distribution_ids]
        if self.noise_enabled:
            probabilities = self._add_noise(probabilities)
        if return_labels:
            return probabilities, self.distribution_labels[distribution_ids]
        return probabilities


class DirichletZipfBinaryVectorProbabilityVectorGenerator(
    DirichletZipfBinaryProbabilityVectorGenerator
):
    """Generate probability vectors with fixed random signed-vector labels."""

    def __init__(
        self,
        num_distributions,
        num_states,
        d_label,
        alpha,
        zipf_exponent,
        device=None,
        dtype=torch.float32,
        generator=None,
        distributions=None,
        distribution_labels=None,
        noise_enabled=False,
        noise_intensity=1.0,
    ):
        if (
            not isinstance(d_label, int)
            or isinstance(d_label, bool)
            or d_label < 1
        ):
            raise ValueError("d_label must be a positive integer")
        self.d_label = d_label
        if distribution_labels is None:
            distribution_labels = 2 * torch.randint(
                0,
                2,
                (num_distributions, self.d_label),
                device=device,
                generator=generator,
            ) - 1
        super().__init__(
            num_distributions=num_distributions,
            num_states=num_states,
            alpha=alpha,
            zipf_exponent=zipf_exponent,
            device=device,
            dtype=dtype,
            generator=generator,
            distributions=distributions,
            distribution_labels=distribution_labels,
            noise_enabled=noise_enabled,
            noise_intensity=noise_intensity,
        )

    def _make_distribution_labels(self, distribution_labels):
        distribution_labels = torch.as_tensor(
            distribution_labels,
            device=self.device,
        )
        expected_shape = (self.num_distributions, self.d_label)
        if distribution_labels.shape != expected_shape:
            raise ValueError(
                "distribution_labels must have shape "
                "(num_distributions, d_label)"
            )
        if not torch.all((distribution_labels == -1) | (distribution_labels == 1)):
            raise ValueError("distribution_labels must contain only -1 and +1")
        return distribution_labels.to(dtype=self.dtype)


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
