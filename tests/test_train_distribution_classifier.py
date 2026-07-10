import unittest
from unittest import mock

import torch

from base.bit_sequences import SummarySequenceClassifierMLP
from base.train_distribution_classifier import (
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


if __name__ == "__main__":
    unittest.main()
