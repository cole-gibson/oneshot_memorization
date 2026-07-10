# oneshot_memorization

Numerical experiments for one-shot memorization.

## Experiment settings

The current focus is sequence classification.

- `base.train_distribution_classifier` trains a summary embedding MLP to
  classify sampled Dirichlet-Zipf state sequences using either random binary
  labels or distribution identity labels.
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
