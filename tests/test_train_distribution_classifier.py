import unittest
from unittest import mock

import torch

from base.bit_sequences import (
    ProbabilityVectorClassifierMLP,
    SummarySequenceClassifierMLP,
)
from base.data_generator import DirichletZipfBinaryProbabilityVectorGenerator
from base.train_distribution_classifier import (
    logarithmic_evaluation_iterations,
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


if __name__ == "__main__":
    unittest.main()
