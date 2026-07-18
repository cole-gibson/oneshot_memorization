import csv
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from base.fine_tune_unseen import (
    checkpoint_paths,
    evaluation_iterations,
    replace_batch_item,
    run,
    threshold_reached,
)
from base.train_distribution_classifier import build_model
from utils.submit_fine_tune_slurm_array import find_training_runs, write_sbatch


class FineTuneHelpersTest(unittest.TestCase):
    def test_replaces_exactly_one_batch_item(self):
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor([0, 1])

        replaced_inputs, replaced_targets = replace_batch_item(
            inputs,
            targets,
            torch.tensor([[8.0, 9.0]]),
            torch.tensor([1]),
        )

        torch.testing.assert_close(
            replaced_inputs,
            torch.tensor([[8.0, 9.0], [3.0, 4.0]]),
        )
        torch.testing.assert_close(replaced_targets, torch.tensor([1, 1]))
        torch.testing.assert_close(inputs, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_threshold_direction_depends_on_metric(self):
        self.assertTrue(threshold_reached("loss", 0.1, 0.2))
        self.assertFalse(threshold_reached("loss", 0.3, 0.2))
        self.assertTrue(threshold_reached("accuracy", 0.9, 0.8))
        self.assertFalse(threshold_reached("accuracy", 0.7, 0.8))

    def test_evaluation_schedule_matches_training_conventions(self):
        self.assertEqual(
            evaluation_iterations(10, {"spacing": "linear", "interval": 4}),
            {1, 4, 8},
        )
        self.assertEqual(
            evaluation_iterations(
                25,
                {"spacing": "logarithmic", "points_per_decade": 1},
            ),
            {1, 10, 25},
        )

    def test_numbered_checkpoint_discovery_ignores_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir()
            for name in ("checkpoint_000020.pt", "checkpoint_000010.pt", "latest.pt"):
                (checkpoint_dir / name).touch()

            paths = checkpoint_paths(run_dir)

        self.assertEqual([iteration for iteration, _ in paths], [10, 20])


class FineTuneSubmissionTest(unittest.TestCase):
    def test_discovers_only_training_runs_with_numbered_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_run = root / "runs" / "valid"
            (valid_run / "checkpoints").mkdir(parents=True)
            (valid_run / "config.yaml").touch()
            (valid_run / "checkpoints" / "checkpoint_000001.pt").touch()
            invalid_run = root / "runs" / "latest_only"
            (invalid_run / "checkpoints").mkdir(parents=True)
            (invalid_run / "config.yaml").touch()
            (invalid_run / "checkpoints" / "latest.pt").touch()

            runs = find_training_runs(root)

        self.assertEqual(runs, [valid_run.resolve()])

    def test_generated_sbatch_is_valid_shell_and_invokes_runner(self):
        args = SimpleNamespace(
            venv=Path(".venv"),
            partition=None,
            account=None,
            qos=None,
            time="00:10:00",
            mem="1G",
            cpus_per_task=1,
            gpus=None,
            gres="gpu:1",
            constraint=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sbatch_path = root / "submit.sbatch"
            write_sbatch(
                sbatch_path,
                args,
                root / "manifest.tsv",
                root / "config.yaml",
                root / "logs",
                3,
            )
            contents = sbatch_path.read_text()
            subprocess.run(["bash", "-n", str(sbatch_path)], check=True)

        self.assertIn("#SBATCH --array=0-2", contents)
        self.assertIn("-m base.fine_tune_unseen", contents)


class FineTuneIntegrationTest(unittest.TestCase):
    def test_runs_autoencoder_trajectory_and_writes_dynamics(self):
        training_config = {
            "data": {
                "type": "dirichlet_zipf_binary_probability_vector",
                "label_scheme": "binary",
                "num_distributions": 2,
                "num_states": 3,
                "alpha": 1.0,
                "zipf_exponent": 0.0,
                "batch_size": 2,
            },
            "model": {
                "type": "probability_autoencoder_mlp",
                "vocab_size": 3,
                "embed_dim": 4,
                "mlp_num_layers": 0,
            },
            "optimizer": {"type": "adam", "lr": 0.01},
            "training": {
                "max_iters": 2,
                "checkpoint_interval": 1,
                "compile": False,
            },
            "evaluation": {"spacing": "linear", "interval": 1},
        }
        experiment_config = {
            "seed": 3,
            "device": "cpu",
            "fine_tuning": {
                "num_unseen_items": 1,
                "max_iters": 2,
                "log_interval": 1,
                "report_interval": 10,
            },
            "evaluation": {"spacing": "linear", "interval": 1},
            "early_stopping": {"threshold": None},
        }
        distributions = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        labels = torch.tensor([0, 1])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "source"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            (run_dir / "config.yaml").touch()
            checkpoint = {
                "config": training_config,
                "model_state": build_model(training_config).state_dict(),
                "distributions": distributions,
                "distribution_labels": labels,
            }
            torch.save(checkpoint, checkpoint_dir / "checkpoint_000001.pt")
            output_dir = root / "output"

            run(run_dir, experiment_config, output_dir)

            with (output_dir / "train_log.csv").open(newline="") as log_file:
                train_rows = list(csv.DictReader(log_file))
            with (output_dir / "eval_log.csv").open(newline="") as log_file:
                eval_rows = list(csv.DictReader(log_file))
            with (output_dir / "summary.csv").open(newline="") as log_file:
                summary_rows = list(csv.DictReader(log_file))

        self.assertEqual([row["fine_tune_iter"] for row in train_rows], ["1", "2"])
        self.assertEqual([row["fine_tune_iter"] for row in eval_rows], ["0", "1", "2"])
        self.assertTrue(all(row["metric_name"] == "loss" for row in eval_rows))
        self.assertEqual(summary_rows[0]["completed_iters"], "2")


if __name__ == "__main__":
    unittest.main()
