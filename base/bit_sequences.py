import torch
import torch.nn as nn
import torch.nn.functional as F


class ZipfBitSequenceGenerator:
    """Sample fixed bit sequences from a Zipf prior over sequence ranks."""

    def __init__(
        self,
        num_sequences,
        sequence_length,
        zipf_exponent,
        device=None,
        generator=None,
        sequences=None,
        labels=None,
        unique=True,
    ):
        if num_sequences < 1:
            raise ValueError("num_sequences must be at least 1")
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if zipf_exponent < 0:
            raise ValueError("zipf_exponent must be non-negative")

        self.num_sequences = int(num_sequences)
        self.sequence_length = int(sequence_length)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.generator = generator

        if sequences is None:
            self.sequences = (
                self._sample_unique_sequences()
                if unique
                else torch.randint(
                    0,
                    2,
                    (self.num_sequences, self.sequence_length),
                    device=self.device,
                    generator=self.generator,
                )
            )
        else:
            self.sequences = self._make_sequences(sequences)

        if labels is None:
            self.labels = torch.randint(
                0,
                2,
                (self.num_sequences,),
                device=self.device,
                generator=self.generator,
            )
        else:
            self.labels = self._make_labels(labels)

        ranks = torch.arange(
            1,
            self.num_sequences + 1,
            device=self.device,
            dtype=torch.float32,
        )
        self.sequence_weights = ranks.pow(-float(zipf_exponent))
        self.sequence_weights /= self.sequence_weights.sum()

    def _sample_unique_sequences(self):
        if self.num_sequences > 2**self.sequence_length:
            raise ValueError(
                "num_sequences cannot exceed the number of possible bit sequences"
            )

        rows = []
        seen = set()
        batch_size = max(1024, self.num_sequences)
        while len(rows) < self.num_sequences:
            candidates = torch.randint(
                0,
                2,
                (batch_size, self.sequence_length),
                device=self.device,
                generator=self.generator,
            )
            for row in candidates.tolist():
                key = tuple(row)
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
                    if len(rows) == self.num_sequences:
                        break
        return torch.tensor(rows, device=self.device, dtype=torch.long)

    def _make_sequences(self, sequences):
        sequences = torch.as_tensor(sequences, device=self.device)
        if sequences.shape != (self.num_sequences, self.sequence_length):
            raise ValueError(
                "sequences must have shape (num_sequences, sequence_length)"
            )
        if sequences.min() < 0 or sequences.max() > 1:
            raise ValueError("sequences must contain only bits")
        return sequences.to(dtype=torch.long)

    def _make_labels(self, labels):
        labels = torch.as_tensor(labels, device=self.device)
        if labels.shape != (self.num_sequences,):
            raise ValueError("labels must have shape (num_sequences,)")
        if labels.min() < 0 or labels.max() > 1:
            raise ValueError("labels must contain only binary labels")
        return labels.to(dtype=torch.long)

    @torch.no_grad()
    def sample(self, batch_size, return_sequence_ids=False):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        sequence_ids = torch.multinomial(
            self.sequence_weights,
            num_samples=batch_size,
            replacement=True,
            generator=self.generator,
        )
        tokens = self.sequences[sequence_ids]
        if return_sequence_ids:
            return tokens, sequence_ids
        return tokens

    @torch.no_grad()
    def sample_from_sequence_ids(self, sequence_ids):
        sequence_ids = torch.as_tensor(sequence_ids, device=self.device)
        if sequence_ids.ndim != 1:
            raise ValueError("sequence_ids must have shape (batch_size,)")
        if sequence_ids.numel() == 0:
            raise ValueError("sequence_ids must contain at least one id")
        sequence_ids = sequence_ids.to(dtype=torch.long)
        if sequence_ids.min() < 0 or sequence_ids.max() >= self.num_sequences:
            raise ValueError("sequence_ids contain ids outside the sequence range")
        return self.sequences[sequence_ids]

    __call__ = sample

    def to(self, device):
        device = torch.device(device)
        self.device = device
        self.sequences = self.sequences.to(device)
        self.labels = self.labels.to(device)
        self.sequence_weights = self.sequence_weights.to(device)
        return self


class BayesOptimalNextBitPredictor:
    """Bayes optimal next-bit predictor for a Zipf prior over fixed sequences."""

    def __init__(self, sequences, sequence_weights, eps=1e-30):
        sequences = torch.as_tensor(sequences)
        sequence_weights = torch.as_tensor(
            sequence_weights,
            device=sequences.device,
            dtype=torch.float32,
        )
        if sequences.ndim != 2:
            raise ValueError("sequences must have shape (num_sequences, sequence_length)")
        if sequences.min() < 0 or sequences.max() > 1:
            raise ValueError("sequences must contain only bits")
        if sequence_weights.shape != (sequences.shape[0],):
            raise ValueError("sequence_weights must have shape (num_sequences,)")
        if torch.any(sequence_weights < 0) or sequence_weights.sum() <= 0:
            raise ValueError("sequence_weights must be non-negative with positive mass")

        self.sequences = sequences.to(dtype=torch.long)
        self.sequence_weights = sequence_weights / sequence_weights.sum()
        self.num_sequences, self.sequence_length = self.sequences.shape
        self.eps = eps

    @classmethod
    def from_generator(cls, data_generator):
        return cls(data_generator.sequences, data_generator.sequence_weights)

    @torch.no_grad()
    def predict_proba(self, prefixes):
        prefixes = torch.as_tensor(prefixes, device=self.sequences.device)
        if prefixes.ndim == 1:
            prefixes = prefixes.unsqueeze(0)
        if prefixes.ndim != 2:
            raise ValueError("prefixes must have shape (batch_size, prefix_length)")
        prefix_length = prefixes.shape[1]
        if prefix_length < 1 or prefix_length >= self.sequence_length:
            raise ValueError("prefix_length must be in [1, sequence_length)")
        if prefixes.min() < 0 or prefixes.max() > 1:
            raise ValueError("prefixes must contain only bits")

        matches = self.sequences[:, :prefix_length].unsqueeze(0) == prefixes[:, None, :]
        matching_weights = matches.all(dim=-1) * self.sequence_weights.unsqueeze(0)
        denominator = matching_weights.sum(dim=1).clamp_min(self.eps)
        next_bits = self.sequences[:, prefix_length].float()
        probability_one = (matching_weights * next_bits.unsqueeze(0)).sum(dim=1)
        probability_one = probability_one / denominator
        return torch.stack([1.0 - probability_one, probability_one], dim=1)

    @torch.no_grad()
    def autoregressive_accuracies(self, tokens):
        tokens = torch.as_tensor(tokens, device=self.sequences.device)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 2 or tokens.shape[1] != self.sequence_length:
            raise ValueError("tokens must have shape (batch_size, sequence_length)")

        correct = []
        for prefix_length in range(1, self.sequence_length):
            proba = self.predict_proba(tokens[:, :prefix_length])
            predictions = proba.argmax(dim=1)
            correct.append(predictions.eq(tokens[:, prefix_length]).float())
        return torch.stack(correct, dim=1).mean(dim=1)

    def to(self, device):
        device = torch.device(device)
        self.sequences = self.sequences.to(device)
        self.sequence_weights = self.sequence_weights.to(device)
        return self


class BayesOptimalSequenceClassifier:
    """Bayes optimal binary-label classifier for exact fixed bit sequences."""

    def __init__(self, sequences, labels, sequence_weights):
        sequences = torch.as_tensor(sequences)
        labels = torch.as_tensor(labels, device=sequences.device)
        sequence_weights = torch.as_tensor(
            sequence_weights,
            device=sequences.device,
            dtype=torch.float32,
        )
        if sequences.ndim != 2:
            raise ValueError("sequences must have shape (num_sequences, sequence_length)")
        if labels.shape != (sequences.shape[0],):
            raise ValueError("labels must have shape (num_sequences,)")
        if labels.min() < 0 or labels.max() > 1:
            raise ValueError("labels must contain only binary labels")
        if sequence_weights.shape != (sequences.shape[0],):
            raise ValueError("sequence_weights must have shape (num_sequences,)")

        self.sequences = sequences.to(dtype=torch.long)
        self.labels = labels.to(dtype=torch.long)
        self.sequence_weights = sequence_weights / sequence_weights.sum()
        self.num_sequences, self.sequence_length = self.sequences.shape

    @classmethod
    def from_generator(cls, data_generator):
        return cls(
            data_generator.sequences,
            data_generator.labels,
            data_generator.sequence_weights,
        )

    @torch.no_grad()
    def predict(self, tokens):
        tokens = torch.as_tensor(tokens, device=self.sequences.device)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 2 or tokens.shape[1] != self.sequence_length:
            raise ValueError("tokens must have shape (batch_size, sequence_length)")

        matches = self.sequences.unsqueeze(0).eq(tokens[:, None, :]).all(dim=-1)
        scores = matches * self.sequence_weights.unsqueeze(0)
        sequence_ids = scores.argmax(dim=1)
        return self.labels[sequence_ids]

    @torch.no_grad()
    def losses(self, tokens, targets, eps=1e-30):
        predictions = self.predict(tokens)
        return -predictions.eq(targets.to(predictions.device)).float().clamp_min(eps).log()

    def to(self, device):
        device = torch.device(device)
        self.sequences = self.sequences.to(device)
        self.labels = self.labels.to(device)
        self.sequence_weights = self.sequence_weights.to(device)
        return self


class SequenceClassifierMLP(nn.Module):
    def __init__(
        self,
        sequence_length,
        num_classes,
        hidden_dim=256,
        num_hidden_layers=2,
        dropout=0.0,
    ):
        super().__init__()
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        if num_classes < 1:
            raise ValueError("num_classes must be at least 1")
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be nonnegative")

        layers = []
        input_dim = sequence_length
        for layer_index in range(num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(input_dim if layer_index == 0 else hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(
            nn.Linear(hidden_dim if num_hidden_layers else input_dim, num_classes)
        )
        self.net = nn.Sequential(*layers)

    def forward(self, tokens, targets=None):
        x = tokens.float()
        logits = self.net(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return {"logits": logits, "loss": loss}


class SummarySequenceClassifierMLP(nn.Module):
    def __init__(
        self,
        vocab_size,
        sequence_length,
        num_classes,
        embed_dim=256,
        mlp_ratio=4,
        mlp_num_layers=2,
        dropout=0.0,
    ):
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be at least 1")
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        if num_classes < 1:
            raise ValueError("num_classes must be at least 1")
        if mlp_num_layers < 0:
            raise ValueError("mlp_num_layers must be nonnegative")

        self.max_seq_len = sequence_length
        self.embed_dim = embed_dim
        self.init_std = embed_dim**-0.5
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        hidden_dim = mlp_ratio * embed_dim
        layers = []
        for layer_index in range(mlp_num_layers):
            layers.extend(
                [
                    nn.Linear(
                        embed_dim if layer_index == 0 else hidden_dim,
                        hidden_dim,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(
            nn.Linear(hidden_dim if mlp_num_layers else embed_dim, num_classes)
        )
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)

    def forward(self, tokens, targets=None):
        if tokens.shape[1] > self.max_seq_len:
            raise ValueError("input sequence is longer than max_seq_len")

        x = self.token_embedding(tokens)
        x = x.mean(dim=1)
        logits = self.mlp(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return {"logits": logits, "loss": loss}
