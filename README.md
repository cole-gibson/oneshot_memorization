# oneshot_memorization

Numerical experiments for one-shot memorization.

## Experiment settings

The current focus is sequence classification.

- `base.train_distribution_classifier` trains an embedding MLP to classify
  Dirichlet-Zipf components using either random binary labels or distribution
  identity labels. Use `data.type: dirichlet_zipf_binary` with
  `model.type: summary_mlp` to classify sampled state sequences, or use
  `data.type: dirichlet_zipf_binary_probability_vector` with
  `model.type: probability_mlp` to classify component probability vectors
  directly.
- `workbooks/train_distribution_label_classifier.py` explores the same
  distribution-based sequence classification task interactively.
- `workbooks/train_bit_sequence_classifier.py` trains an MLP to classify fixed
  binary sequences using either random binary labels or sequence identity labels.

The workbooks use Bayes optimal classification baselines in `base/bit_sequences.py`
and `base/estimators.py`.

With `training.compile: true`, the complete parameter update (zeroing gradients,
forward and loss, backward, optional clipping, and optimizer step) is compiled
as one function. Set it to `false` for short runs, debugging, or environments
where `torch.compile` is unsupported.

Evaluation uses a linear stride by default (`evaluation.spacing: linear` and
`evaluation.interval`). To evaluate on a logarithmic schedule instead, set
`evaluation.spacing: logarithmic` and `evaluation.points_per_decade` to the
desired number of evaluations per factor of ten in training iterations. The
logarithmic schedule always includes iterations 1 and `training.max_iters`.
