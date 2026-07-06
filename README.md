# oneshot_memorization

Numerical experiments for one-shot memorization.

## Experiment settings

The original setting studies autoregressive prediction from Zipf-selected
Dirichlet components.

The bit-sequence setting studies a Zipf prior over a fixed dataset of binary
sequences:

- `workbooks/train_next_bit_sequences.py` trains the full Transformer on
  next-bit cross entropy and marks a sequence memorized once next-bit accuracy
  exceeds 90%.
- `workbooks/train_bit_sequence_classifier.py` trains a single MLP to classify
  full bit sequences using either random binary labels or sequence identity
  labels.
- `workbooks/train_distribution_label_classifier.py` trains the same summary
  embedding MLP to classify sampled state sequences using either random binary
  labels or distribution identity labels.

The workbooks use Bayes optimal baselines in `base/bit_sequences.py` and
`base/estimators.py`.
