import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from base.bit_sequences import (
    ProbabilityVectorAutoencoderMLP,
    ProbabilityVectorClassifierMLP,
    SequenceClassifierMLP,
    SummarySequenceClassifierMLP,
)
from base.data_generator import (
    DirichletZipfBinaryProbabilityVectorGenerator,
    DirichletZipfBinaryVectorProbabilityVectorGenerator,
)
from base.train_distribution_classifier import (
    evaluate,
    logarithmic_evaluation_iterations,
    logarithmic_iterations,
    make_compiled_training_step,
    validate_config,
)


def make_config():
    return {
        "model": {
            "type": "summary_mlp",
            "vocab_size": 7,
            "sequence_length": 3,
            "num_classes": 2,
        },
        "data": {
            "type": "dirichlet_zipf_binary",
            "label_scheme": "binary",
            "num_distributions": 5,
            "num_states": 7,
            "sequence_length": 3,
        },
        "training": {
            "max_iters": 2,
            "checkpoint_interval": 2,
            "compile": True,
        },
        "evaluation": {"interval": 1, "seqs_per_distribution": 1},
    }


def make_bit_config():
    return {
        "model": {
            "type": "bit_sequence_mlp",
            "sequence_length": 5,
            "num_classes": 2,
        },
        "data": {
            "type": "zipf_bit_binary",
            "label_scheme": "binary",
            "num_sequences": 8,
            "sequence_length": 5,
        },
        "training": {
            "max_iters": 2,
            "checkpoint_interval": 2,
            "compile": False,
        },
        "evaluation": {"interval": 1},
    }


class CompiledTrainingStepTest(unittest.TestCase):
    def test_compile_must_be_boolean(self):
        config = make_config()
        config["training"]["compile"] = "yes"
        with self.assertRaisesRegex(ValueError, "training.compile"):
            validate_config(config)

    def test_compiled_step_updates_parameters(self):
        torch.manual_seed(0)
        model = SummarySequenceClassifierMLP(
            vocab_size=7,
            sequence_length=3,
            num_classes=2,
            embed_dim=8,
            mlp_ratio=2,
            mlp_num_layers=1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        tokens = torch.randint(0, 7, (4, 3))
        labels = torch.randint(0, 2, (4,))
        before = [parameter.detach().clone() for parameter in model.parameters()]

        with mock.patch(
            "base.train_distribution_classifier.torch.compile",
            side_effect=lambda function: function,
        ) as compile_mock:
            step = make_compiled_training_step(
                model,
                optimizer,
                scaler,
                torch.device("cpu"),
                None,
                None,
            )
            loss = step(tokens, labels)

        compile_mock.assert_called_once()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters())
            )
        )


class EvaluationScheduleTest(unittest.TestCase):
    def test_logarithmic_iterations_have_configured_density(self):
        self.assertEqual(
            logarithmic_evaluation_iterations(100, points_per_decade=2),
            {1, 3, 10, 32, 100},
        )

    def test_logarithmic_iterations_include_non_power_of_ten_final_iteration(self):
        self.assertEqual(
            logarithmic_evaluation_iterations(25, points_per_decade=1),
            {1, 10, 25},
        )

    def test_logarithmic_config_does_not_require_linear_interval(self):
        config = make_config()
        config["evaluation"] = {
            "spacing": "logarithmic",
            "points_per_decade": 4,
            "seqs_per_distribution": 1,
        }

        validate_config(config)

    def test_logarithmic_density_must_be_a_positive_integer(self):
        for value in (0, 1.5, True):
            with self.subTest(value=value):
                config = make_config()
                config["evaluation"]["spacing"] = "logarithmic"
                config["evaluation"]["points_per_decade"] = value
                with self.assertRaisesRegex(ValueError, "points_per_decade"):
                    validate_config(config)

    def test_unknown_spacing_is_rejected(self):
        config = make_config()
        config["evaluation"]["spacing"] = "geometric"

        with self.assertRaisesRegex(ValueError, "evaluation.spacing"):
            validate_config(config)


class CheckpointScheduleTest(unittest.TestCase):
    def test_logarithmic_iterations_have_configured_density(self):
        self.assertEqual(
            logarithmic_iterations(100, points_per_decade=2),
            {1, 3, 10, 32, 100},
        )

    def test_logarithmic_config_does_not_require_linear_interval(self):
        config = make_config()
        config["training"] = {
            "max_iters": 100,
            "checkpoint_spacing": "logarithmic",
            "checkpoint_points_per_decade": 4,
        }

        validate_config(config)

    def test_logarithmic_density_must_be_a_positive_integer(self):
        for value in (0, 1.5, True):
            with self.subTest(value=value):
                config = make_config()
                config["training"]["checkpoint_spacing"] = "logarithmic"
                config["training"]["checkpoint_points_per_decade"] = value
                with self.assertRaisesRegex(
                    ValueError, "checkpoint_points_per_decade"
                ):
                    validate_config(config)

    def test_unknown_spacing_is_rejected(self):
        config = make_config()
        config["training"]["checkpoint_spacing"] = "geometric"

        with self.assertRaisesRegex(ValueError, "checkpoint_spacing"):
            validate_config(config)


class ProbabilityVectorSettingTest(unittest.TestCase):
    def test_generator_returns_selected_probability_vectors_and_labels(self):
        distributions = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        labels = torch.tensor([1, 0])
        generator = DirichletZipfBinaryProbabilityVectorGenerator(
            num_distributions=2,
            num_states=2,
            alpha=1.0,
            zipf_exponent=0.0,
            distributions=distributions,
            distribution_labels=labels,
        )

        probabilities, distribution_ids, sampled_labels = (
            generator.sample(
                batch_size=8,
                return_distribution_ids=True,
                return_labels=True,
            )
        )

        torch.testing.assert_close(probabilities, distributions[distribution_ids])
        torch.testing.assert_close(sampled_labels, labels[distribution_ids])

    def test_model_uses_probability_weighted_state_embedding(self):
        model = ProbabilityVectorClassifierMLP(
            vocab_size=3,
            num_classes=2,
            embed_dim=4,
            mlp_num_layers=0,
        )
        probabilities = torch.tensor([[0.2, 0.3, 0.5]])

        result = model(probabilities)
        expected_embedding = probabilities @ model.state_embedding
        expected_logits = model.mlp(model.input_layer_norm(expected_embedding))

        torch.testing.assert_close(result["logits"], expected_logits)

    def test_probability_vector_config_does_not_require_sequence_length(self):
        config = make_config()
        config["data"]["type"] = "dirichlet_zipf_binary_probability_vector"
        config["data"].pop("sequence_length")
        config["evaluation"].pop("seqs_per_distribution")
        config["model"] = {
            "type": "probability_mlp",
            "vocab_size": 7,
            "num_classes": 2,
        }

        validate_config(config)

    def test_probability_vector_config_rejects_repeated_evaluation_vectors(self):
        config = make_config()
        config["data"]["type"] = "dirichlet_zipf_binary_probability_vector"
        config["model"] = {
            "type": "probability_mlp",
            "vocab_size": 7,
            "num_classes": 2,
        }
        config["evaluation"]["seqs_per_distribution"] = 2

        with self.assertRaisesRegex(ValueError, "must be 1"):
            validate_config(config)

    def test_probability_vector_data_requires_probability_model(self):
        config = make_config()
        config["data"]["type"] = "dirichlet_zipf_binary_probability_vector"

        with self.assertRaisesRegex(ValueError, "requires model.type"):
            validate_config(config)


class ProbabilityVectorAutoencoderSettingTest(unittest.TestCase):
    @staticmethod
    def make_autoencoder_config():
        config = make_config()
        config["data"]["type"] = "dirichlet_zipf_binary_probability_vector"
        config["data"].pop("sequence_length")
        config["model"] = {
            "type": "probability_autoencoder_mlp",
            "vocab_size": 7,
        }
        config["evaluation"] = {"interval": 1}
        return config

    def test_config_accepts_probability_vector_autoencoder(self):
        validate_config(self.make_autoencoder_config())

    def test_model_reconstruction_loss_is_kl_divergence(self):
        model = ProbabilityVectorAutoencoderMLP(
            vocab_size=3,
            embed_dim=4,
            mlp_num_layers=0,
        )
        probabilities = torch.tensor([[0.2, 0.3, 0.5]])

        result = model(probabilities, targets=probabilities)
        expected = torch.nn.functional.kl_div(
            result["probabilities"].log(),
            probabilities,
            reduction="batchmean",
        )

        torch.testing.assert_close(result["loss"], expected)
        self.assertNotIn("logits", result)
        self.assertEqual(result["probabilities"].shape, probabilities.shape)
        torch.testing.assert_close(
            result["probabilities"].sum(dim=-1), torch.ones(1)
        )

    def test_evaluation_logs_only_kl_loss_metric(self):
        class FixedModel(torch.nn.Module):
            def forward(self, inputs, targets=None, loss_reduction="mean"):
                losses = torch.nn.functional.kl_div(
                    inputs.log(), targets, reduction="none"
                ).sum(dim=-1)
                if loss_reduction != "none":
                    losses = losses.mean()
                return {"probabilities": inputs, "loss": losses}

        model = FixedModel()
        model.train()
        probabilities = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        eval_batch = {
            "tokens": probabilities,
            "labels": torch.tensor([1, 0]),
            "distribution_ids": torch.tensor([0, 1]),
        }

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "eval.csv"
            evaluate(
                model=model,
                data_generator=None,
                config=self.make_autoencoder_config(),
                eval_batch=eval_batch,
                presentation_counts=torch.tensor([4, 2]),
                log_path=log_path,
                iteration=7,
                amp_dtype=None,
            )
            with log_path.open(newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(
            list(rows[0]),
            ["iter", "distribution_id", "loss", "training_seen_count"],
        )
        self.assertTrue(all(float(row["loss"]) < 1e-6 for row in rows))
        self.assertTrue(model.training)


class ProbabilityVectorLabelRegressionSettingTest(unittest.TestCase):
    @staticmethod
    def make_vector_label_config():
        config = make_config()
        config["data"] = {
            "type": "dirichlet_zipf_binary_vector_probability_vector",
            "label_scheme": "binary",
            "num_distributions": 5,
            "num_states": 7,
            "d_label": 3,
        }
        config["model"] = {
            "type": "probability_mlp",
            "vocab_size": 7,
            "num_classes": 3,
            "loss": "mse",
        }
        config["evaluation"] = {"interval": 1}
        return config

    def test_generator_returns_selected_signed_vector_labels(self):
        distributions = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        labels = torch.tensor([[1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]])
        generator = DirichletZipfBinaryVectorProbabilityVectorGenerator(
            num_distributions=2,
            num_states=2,
            d_label=3,
            alpha=1.0,
            zipf_exponent=0.0,
            distributions=distributions,
            distribution_labels=labels,
        )

        probabilities, distribution_ids, sampled_labels = generator.sample(
            batch_size=8,
            return_distribution_ids=True,
            return_labels=True,
        )

        torch.testing.assert_close(probabilities, distributions[distribution_ids])
        torch.testing.assert_close(sampled_labels, labels[distribution_ids])

    def test_probability_model_uses_mse_for_vector_targets(self):
        model = ProbabilityVectorClassifierMLP(
            vocab_size=3,
            num_classes=2,
            embed_dim=4,
            mlp_num_layers=0,
            loss="mse",
        )
        probabilities = torch.tensor([[0.2, 0.3, 0.5]])
        targets = torch.tensor([[1.0, -1.0]])

        result = model(probabilities, targets=targets)

        torch.testing.assert_close(
            result["loss"],
            torch.nn.functional.mse_loss(result["logits"], targets),
        )

    def test_config_requires_mse_and_matching_label_dimension(self):
        config = self.make_vector_label_config()
        validate_config(config)

        config["model"]["loss"] = "cross_entropy"
        with self.assertRaisesRegex(ValueError, "model.loss 'mse'"):
            validate_config(config)

        config = self.make_vector_label_config()
        config["model"]["num_classes"] = 2
        with self.assertRaisesRegex(ValueError, "label dimension"):
            validate_config(config)

    def test_evaluation_logs_per_distribution_signed_accuracy(self):
        class FixedModel(torch.nn.Module):
            def forward(self, inputs):
                return {"logits": inputs}

        model = FixedModel()
        model.train()
        eval_batch = {
            "tokens": torch.tensor([[0.2, -0.5, 0.0], [-0.1, 0.7, 0.9]]),
            "labels": torch.tensor([[1.0, -1.0, 1.0], [1.0, 1.0, 1.0]]),
            "distribution_ids": torch.tensor([0, 1]),
        }
        config = self.make_vector_label_config()

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "eval.csv"
            evaluate(
                model=model,
                data_generator=None,
                config=config,
                eval_batch=eval_batch,
                presentation_counts=torch.tensor([4, 2]),
                log_path=log_path,
                iteration=7,
                amp_dtype=None,
            )
            with log_path.open(newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(
            [row["accuracy"] for row in rows],
            ["0.66666669", "0.33333334"],
        )
        self.assertTrue(model.training)


class BitSequenceSettingTest(unittest.TestCase):
    def test_config_does_not_require_distribution_fields(self):
        validate_config(make_bit_config())

    def test_model_processes_signed_bits_directly(self):
        model = SequenceClassifierMLP(
            sequence_length=3,
            num_classes=2,
            hidden_dim=16,
            num_hidden_layers=1,
        )
        tokens = torch.tensor([[0, 1, 1], [1, 0, 1]])

        result = model(tokens)
        expected_logits = model.net(2.0 * tokens.float() - 1.0)

        torch.testing.assert_close(result["logits"], expected_logits)

    def test_identity_labels_require_one_class_per_sequence(self):
        config = make_bit_config()
        config["data"]["label_scheme"] = "identity"

        with self.assertRaisesRegex(ValueError, "number of tasks"):
            validate_config(config)

        config["model"]["num_classes"] = config["data"]["num_sequences"]
        validate_config(config)


if __name__ == "__main__":
    unittest.main()
