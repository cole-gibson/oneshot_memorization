import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

from base.train_distribution_classifier import (
    PhaseTimingCollector,
    TIMING_FIELDS,
    main,
    validate_benchmark_run,
    validate_config,
)


def make_config(run_dir, benchmark_enabled):
    return {
        "experiment_name": "timing_test",
        "seed": 7,
        "device": "cpu",
        "run": {"run_dir": str(run_dir), "resume_from": None},
        "data": {
            "type": "dirichlet_zipf_binary",
            "label_scheme": "binary",
            "num_distributions": 5,
            "num_states": 7,
            "alpha": 0.5,
            "zipf_exponent": 1.0,
            "sequence_length": 3,
            "batch_size": 4,
        },
        "model": {
            "type": "summary_mlp",
            "vocab_size": 7,
            "sequence_length": 3,
            "num_classes": 2,
            "embed_dim": 8,
            "mlp_ratio": 2,
            "mlp_num_layers": 1,
            "dropout": 0.0,
        },
        "optimizer": {"type": "adam", "lr": 0.001},
        "training": {
            "max_iters": 4,
            "report_interval": 10,
            "log_interval": 10,
            "checkpoint_interval": 2,
            "grad_clip_norm": None,
        },
        "evaluation": {
            "interval": 2,
            "seed": 8,
            "num_distributions": 5,
            "seqs_per_distribution": 2,
            "microbatch_size": 5,
        },
        "benchmark": {
            "enabled": benchmark_enabled,
            "warmup_iters": 1,
            "measure_iters": 2,
        },
    }


def run_training(config, config_path):
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    args = SimpleNamespace(config=config_path, resume_from=None, seed=None)
    with mock.patch(
        "base.train_distribution_classifier.parse_args", return_value=args
    ), redirect_stdout(io.StringIO()):
        main()


class BenchmarkTimingTest(unittest.TestCase):
    def test_cpu_timing_csv_and_training_semantics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            timed_dir = root / "timed"
            plain_dir = root / "plain"
            run_training(
                make_config(timed_dir, True),
                root / "timed_config.yaml",
            )
            run_training(
                make_config(plain_dir, False),
                root / "plain_config.yaml",
            )

            self.assertFalse((plain_dir / "timing.csv").exists())
            self.assertTrue((plain_dir / "train_log.csv").exists())
            self.assertTrue((plain_dir / "eval_by_distribution.csv").exists())

            with (timed_dir / "timing.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                reader = csv.DictReader(file)
                self.assertEqual(tuple(reader.fieldnames), TIMING_FIELDS)
                rows = {row["phase"]: row for row in reader}

            measured_phases = (
                "data_sample",
                "label_and_count_bookkeeping",
                "zero_grad",
                "forward_and_loss",
                "backward",
                "optimizer_step_and_scaler_update",
                "training_step_total",
            )
            for phase in measured_phases:
                self.assertEqual(int(rows[phase]["num_calls"]), 2)
                self.assertEqual(int(rows[phase]["num_examples"]), 8)
                self.assertGreaterEqual(float(rows[phase]["total_ms"]), 0.0)
                self.assertTrue(rows[phase]["mean_ms"])
            self.assertEqual(int(rows["grad_unscale_and_clip"]["num_calls"]), 0)
            self.assertEqual(rows["grad_unscale_and_clip"]["mean_ms"], "")

            subphase_total = sum(
                float(rows[phase]["total_ms"])
                for phase in measured_phases
                if phase != "training_step_total"
            )
            step_total = float(rows["training_step_total"]["total_ms"])
            self.assertLessEqual(subphase_total, step_total * 1.05)

            self.assertEqual(int(rows["evaluation_forward"]["num_calls"]), 3)
            self.assertEqual(int(rows["evaluation_forward"]["num_examples"]), 30)
            self.assertEqual(int(rows["evaluation_csv_write"]["num_calls"]), 3)
            self.assertEqual(int(rows["checkpoint_numbered"]["num_calls"]), 2)
            self.assertEqual(int(rows["checkpoint_latest"]["num_calls"]), 2)

            timed_checkpoint = torch.load(
                timed_dir / "checkpoints" / "latest.pt",
                map_location="cpu",
                weights_only=False,
            )
            plain_checkpoint = torch.load(
                plain_dir / "checkpoints" / "latest.pt",
                map_location="cpu",
                weights_only=False,
            )
            for name, timed_parameter in timed_checkpoint["model_state"].items():
                torch.testing.assert_close(
                    timed_parameter,
                    plain_checkpoint["model_state"][name],
                    rtol=0,
                    atol=0,
                )

    def test_benchmark_validation(self):
        config = make_config("unused", True)
        config["benchmark"]["warmup_iters"] = -1
        with self.assertRaisesRegex(ValueError, "warmup_iters"):
            validate_config(config)

        config = make_config("unused", True)
        config["benchmark"]["measure_iters"] = 0
        with self.assertRaisesRegex(ValueError, "measure_iters"):
            validate_config(config)

        config = make_config("unused", True)
        config["benchmark"]["warmup_iters"] = True
        with self.assertRaisesRegex(ValueError, "warmup_iters"):
            validate_config(config)

        config = make_config("unused", True)
        config["training"]["compile"] = True
        config["benchmark"]["warmup_iters"] = 0
        with self.assertRaisesRegex(ValueError, "warm-up"):
            validate_config(config)

        config = make_config("unused", True)
        with self.assertRaisesRegex(ValueError, "MPS"):
            validate_benchmark_run(config, torch.device("mps"), 0)

        config["training"]["max_iters"] = 2
        with self.assertRaisesRegex(ValueError, "exceed remaining"):
            validate_benchmark_run(config, torch.device("cpu"), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_event_collector(self):
        device = torch.device("cuda")
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = PhaseTimingCollector(
                True,
                device,
                Path(temp_dir) / "timing.csv",
                1,
                1,
            )
            tensor = torch.ones(1024, device=device)
            with mock.patch("torch.cuda.synchronize", wraps=torch.cuda.synchronize) as sync:
                with collector.phase(
                    "forward_and_loss", 1, 1024, use_cuda_events=True
                ):
                    tensor.square_()
                self.assertEqual(sync.call_count, 0)
                collector.resolve_cuda()
                self.assertEqual(sync.call_count, 1)
            record = collector.records["forward_and_loss"]
            self.assertEqual(record["num_calls"], 1)
            self.assertGreater(record["total_ms"], 0.0)
            self.assertEqual(record["backends"], {"cuda_event"})


if __name__ == "__main__":
    unittest.main()
