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

## Training benchmarks

`base.train_distribution_classifier` supports an optional `benchmark` config
section. When enabled on CPU or CUDA, it writes aggregated initialization,
training-step, evaluation, and checkpoint phase measurements to `timing.csv` in
the run directory. `warmup_iters` and `measure_iters` select one steady-state
training window relative to the start (or resume point) of the invocation.

CUDA computational timings use events on the current stream. They measure GPU
execution time and exclude Python dispatch overhead; `training_step_total`
provides the aggregate device time for the measured training steps. CPU and
host-I/O timings use wall time. Benchmark mode is unsupported on MPS.
