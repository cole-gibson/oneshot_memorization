# oneshot_memorization

Numerical experiments for one-shot memorization.

## Experiment settings

The current focus is classification under Zipf-distributed task frequencies.

- `base.train_distribution_classifier` trains an embedding MLP to classify
  Dirichlet-Zipf components using either random binary labels or distribution
  identity labels. Use `data.type: dirichlet_zipf_binary` with
  `model.type: summary_mlp` to classify sampled state sequences, or use
  `data.type: dirichlet_zipf_binary_probability_vector` with
  `model.type: probability_mlp` to classify component probability vectors
  directly. Use `model.type: probability_autoencoder_mlp` with the same data
  type to autoencode those vectors with KL-divergence reconstruction loss.
- `data.type: dirichlet_zipf_binary_vector_probability_vector` modifies the
  probability-vector setting by assigning each distribution a fixed
  `d_label`-dimensional label in `{−1,+1}^d_label`. The model is trained with
  mean-squared error and evaluated by the normalized signed overlap
  `mean(sign(prediction) * label)`.
- The same trainer accepts `data.type: zipf_bit_binary` with
  `model.type: bit_sequence_mlp`. This setting samples a fixed collection of
  unique N-bit vectors from a Zipf prior and predicts their random binary labels.
  The model sends signed bits directly through an MLP.
- `workbooks/train_distribution_label_classifier.py` explores the same
  distribution-based sequence classification task interactively.
- `workbooks/train_bit_sequence_classifier.py` trains an MLP to classify fixed
  binary sequences interactively.

Runnable examples are provided in `sample_configs/distribution_classifier.yaml`,
`sample_configs/distribution_vector_classifier.yaml`,
`sample_configs/distribution_vector_autoencoder.yaml`,
`sample_configs/distribution_vector_label_regression.yaml`, and
`sample_configs/bit_sequence_classifier.yaml`. Run one with, for example:

```bash
uv run python -m base.train_distribution_classifier \
  --config sample_configs/bit_sequence_classifier.yaml
```

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

Checkpoint recording also uses a linear stride by default
(`training.checkpoint_spacing: linear` and `training.checkpoint_interval`). To
record checkpoints logarithmically, set `training.checkpoint_spacing` to
`logarithmic` and `training.checkpoint_points_per_decade` to the desired number
of checkpoints per factor of ten in training iterations. The logarithmic
schedule always includes iterations 1 and `training.max_iters`.
