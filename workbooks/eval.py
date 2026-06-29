import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import argparse
    import csv
    import re
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    import yaml

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from base.minimal_model import Transformer as MinimalTransformer
    from base.model import Transformer as FullTransformer
    from base.estimators import (
        BayesOptimalEstimator,
        DirichletEmpiricalEstimator,
    )

    CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.pt$")
    MODEL_TYPES = {
        "full": FullTransformer,
        "minimal": MinimalTransformer,
    }
    return (
        BayesOptimalEstimator,
        CHECKPOINT_RE,
        DirichletEmpiricalEstimator,
        F,
        MODEL_TYPES,
        Path,
        argparse,
        csv,
        sys,
        torch,
        yaml,
    )


@app.cell
def _(CHECKPOINT_RE, MODEL_TYPES, Path, argparse, torch, yaml):
    def load_config(path):
        with path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        return config

    def resolve_device(device_name):
        if device_name == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(device_name)

    def make_torch_generator(device, seed):
        if device.type not in ("cpu", "cuda"):
            return None
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        return generator

    def parse_checkpoint_iteration(path):
        match = CHECKPOINT_RE.match(path.name)
        if match is None:
            return None
        return int(match.group(1))

    def list_checkpoint_paths(checkpoint_dir, file_stride, max_checkpoints=0):
        file_stride = int(file_stride)
        max_checkpoints = int(max_checkpoints)
        if file_stride < 1:
            raise ValueError("checkpoint stride must be at least 1")
        if max_checkpoints < 0:
            raise ValueError("max_checkpoints must be non-negative")

        checkpoint_dir = Path(checkpoint_dir)
        checkpoints = [
            path
            for path in checkpoint_dir.glob("checkpoint_*.pt")
            if parse_checkpoint_iteration(path) is not None
        ]
        checkpoints.sort(key=parse_checkpoint_iteration)
        selected = checkpoints[::file_stride]
        if checkpoints and checkpoints[-1] not in selected:
            selected.append(checkpoints[-1])
        if max_checkpoints:
            selected = selected[:max_checkpoints]
            if checkpoints and checkpoints[-1] not in selected:
                selected[-1] = checkpoints[-1]
        return selected

    def build_model(model_config):
        model_config = dict(model_config)
        model_type = model_config.pop("type", "full")
        if model_type not in MODEL_TYPES:
            raise ValueError(
                "model.type must be one of "
                f"{sorted(MODEL_TYPES)}; got {model_type!r}"
            )
        return MODEL_TYPES[model_type](**model_config)

    def parse_eval_args(argv):
        parser = argparse.ArgumentParser(description="Evaluate a training run.")
        parser.add_argument(
            "experiment_dir",
            nargs="?",
            default=None,
            help="Experiment directory containing config.yaml and checkpoints/.",
        )
        parser.add_argument(
            "--experiment-dir",
            dest="experiment_dir_option",
            default=None,
            help="Experiment directory containing config.yaml and checkpoints/.",
        )
        args, _ = parser.parse_known_args(argv)
        return args.experiment_dir_option or args.experiment_dir

    return (
        build_model,
        list_checkpoint_paths,
        load_config,
        make_torch_generator,
        parse_checkpoint_iteration,
        parse_eval_args,
        resolve_device,
    )


@app.cell
def _(Path, make_torch_generator, torch):
    def load_saved_distributions(experiment_dir, config, device):
        path = Path(experiment_dir) / "distributions.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing saved distributions: {path}")

        distributions = torch.load(path, map_location=device, weights_only=True)
        if distributions.ndim != 2:
            raise ValueError(
                "saved distributions must have shape "
                "(num_distributions, num_states)"
            )

        expected_shape = (
            int(config["data"]["num_distributions"]),
            int(config["data"]["num_states"]),
        )
        if tuple(distributions.shape) != expected_shape:
            raise ValueError(
                "saved distributions shape does not match config data "
                f"({tuple(distributions.shape)} != {expected_shape})"
            )
        return distributions

    def make_balanced_batch(
        distributions,
        batch_sequence_length,
        batch_seqs_per_distribution,
        batch_distribution_start,
        batch_max_distributions,
        eval_seed,
    ):
        seqs_per_distribution_int = int(batch_seqs_per_distribution)
        distribution_start_int = int(batch_distribution_start)
        max_distributions_int = int(batch_max_distributions)
        num_distributions = distributions.shape[0]
        if seqs_per_distribution_int < 1:
            raise ValueError("seqs_per_distribution must be at least 1")
        if (
            distribution_start_int < 0
            or distribution_start_int >= num_distributions
        ):
            raise ValueError("distribution_start must be in the distribution range")

        distribution_stop = (
            num_distributions
            if max_distributions_int == 0
            else min(
                num_distributions,
                distribution_start_int + max_distributions_int,
            )
        )
        if distribution_stop <= distribution_start_int:
            raise ValueError("selected distribution range is empty")

        selected_distribution_ids = torch.arange(
            distribution_start_int,
            distribution_stop,
            device=distributions.device,
            dtype=torch.long,
        )
        distribution_ids = selected_distribution_ids.repeat_interleave(
            seqs_per_distribution_int
        )
        tokens = torch.multinomial(
            distributions[distribution_ids],
            num_samples=batch_sequence_length,
            replacement=True,
            generator=make_torch_generator(distributions.device, eval_seed),
        )
        return tokens.cpu(), distribution_ids.cpu(), selected_distribution_ids.cpu()

    return load_saved_distributions, make_balanced_batch


@app.cell
def _(F, torch):
    @torch.no_grad()
    def model_autoregressive_losses(model, tokens):
        input_ids = tokens[:, :-1]
        targets = tokens[:, 1:]
        logits = model(input_ids)["logits"]
        losses = F.cross_entropy(
            logits.transpose(1, 2),
            targets,
            reduction="none",
        )
        return losses.mean(dim=1)

    return (model_autoregressive_losses,)


@app.cell
def _(
    DirichletEmpiricalEstimator,
    build_model,
    model_autoregressive_losses,
    parse_checkpoint_iteration,
    torch,
):
    def evaluate_checkpoint(
        checkpoint_path,
        config,
        tokens,
        distribution_ids,
        selected_distribution_ids,
        bayes_estimator,
        bayes_component_chunk_size,
        microbatch_size,
        device,
    ):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model(config["model"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        use_counts = checkpoint.get("distribution_use_counts")
        if use_counts is None:
            raise ValueError(
                f"{checkpoint_path} does not contain distribution_use_counts"
            )
        use_counts = use_counts.cpu().to(dtype=torch.long)
        if use_counts.numel() != config["data"]["num_distributions"]:
            raise ValueError(
                "checkpoint distribution_use_counts length does not match "
                "data.num_distributions"
            )

        num_distributions = config["data"]["num_distributions"]
        sum_model = torch.zeros(num_distributions, dtype=torch.float64)
        sum_dirichlet = torch.zeros(num_distributions, dtype=torch.float64)
        sum_dirichlet_gap = torch.zeros(num_distributions, dtype=torch.float64)
        sum_bayes = torch.zeros(num_distributions, dtype=torch.float64)
        sum_bayes_gap = torch.zeros(num_distributions, dtype=torch.float64)
        eval_counts = torch.zeros(num_distributions, dtype=torch.long)

        dirichlet_estimator = DirichletEmpiricalEstimator(
            num_states=config["data"]["num_states"],
            alpha=config["data"]["alpha"],
            device=device,
        )

        microbatch_size = int(microbatch_size)
        if microbatch_size < 1:
            raise ValueError("microbatch_size must be at least 1")

        for start in range(0, tokens.shape[0], microbatch_size):
            stop = min(tokens.shape[0], start + microbatch_size)
            batch_tokens = tokens[start:stop].to(device)
            batch_distribution_ids = distribution_ids[start:stop]

            model_losses = model_autoregressive_losses(model, batch_tokens).cpu()
            dirichlet_losses = dirichlet_estimator.autoregressive_losses(
                batch_tokens
            ).cpu()
            bayes_losses = bayes_estimator.autoregressive_losses(
                batch_tokens,
                component_chunk_size=bayes_component_chunk_size,
            ).cpu()
            dirichlet_gaps = model_losses - dirichlet_losses
            bayes_gaps = model_losses - bayes_losses

            sum_model.scatter_add_(
                0,
                batch_distribution_ids,
                model_losses.to(dtype=torch.float64),
            )
            sum_dirichlet.scatter_add_(
                0,
                batch_distribution_ids,
                dirichlet_losses.to(dtype=torch.float64),
            )
            sum_dirichlet_gap.scatter_add_(
                0,
                batch_distribution_ids,
                dirichlet_gaps.to(dtype=torch.float64),
            )
            sum_bayes.scatter_add_(
                0,
                batch_distribution_ids,
                bayes_losses.to(dtype=torch.float64),
            )
            sum_bayes_gap.scatter_add_(
                0,
                batch_distribution_ids,
                bayes_gaps.to(dtype=torch.float64),
            )
            eval_counts.scatter_add_(
                0,
                batch_distribution_ids,
                torch.ones_like(batch_distribution_ids, dtype=torch.long),
            )

        fallback_iteration = parse_checkpoint_iteration(checkpoint_path)
        iteration = int(checkpoint.get("iteration", fallback_iteration))
        rows = []
        for distribution_id in selected_distribution_ids.tolist():
            count = int(eval_counts[distribution_id].item())
            if count == 0:
                continue
            mean_dirichlet_loss = sum_dirichlet[distribution_id].item() / count
            mean_dirichlet_gap = (
                sum_dirichlet_gap[distribution_id].item() / count
            )
            rows.append(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "iteration": iteration,
                    "distribution_id": distribution_id,
                    "seqs_per_distribution": count,
                    "mean_model_loss": sum_model[distribution_id].item() / count,
                    "mean_dirichlet_loss": mean_dirichlet_loss,
                    "mean_dirichlet_gap": mean_dirichlet_gap,
                    "mean_baseline_loss": mean_dirichlet_loss,
                    "mean_gap": mean_dirichlet_gap,
                    "mean_bayes_loss": sum_bayes[distribution_id].item() / count,
                    "mean_bayes_gap": sum_bayes_gap[distribution_id].item() / count,
                    "training_seen_count": int(use_counts[distribution_id].item()),
                }
            )
        return rows

    return (evaluate_checkpoint,)


@app.cell
def _(
    BayesOptimalEstimator,
    Path,
    csv,
    evaluate_checkpoint,
    list_checkpoint_paths,
    load_config,
    load_saved_distributions,
    make_balanced_batch,
    resolve_device,
):
    def write_rows_csv(rows, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "checkpoint_path",
            "iteration",
            "distribution_id",
            "seqs_per_distribution",
            "mean_model_loss",
            "mean_dirichlet_loss",
            "mean_dirichlet_gap",
            "mean_baseline_loss",
            "mean_gap",
            "mean_bayes_loss",
            "mean_bayes_gap",
            "training_seen_count",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return output_path

    def run_evaluation(
        eval_settings,
    ):
        experiment_dir_path = Path(eval_settings["experiment_dir"]).expanduser()
        config_path = experiment_dir_path / "config.yaml"
        checkpoint_dir = experiment_dir_path / "checkpoints"
        if not config_path.exists():
            raise FileNotFoundError(f"missing config.yaml: {config_path}")
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"missing checkpoints directory: {checkpoint_dir}")

        config = load_config(config_path)
        device = resolve_device(eval_settings["device_name"])
        checkpoints = list_checkpoint_paths(
            checkpoint_dir,
            eval_settings["checkpoint_stride"],
            eval_settings.get("max_checkpoints", 0),
        )
        if not checkpoints:
            raise ValueError(f"no numbered checkpoints found in {checkpoint_dir}")

        sequence_length = int(config["data"]["sequence_length"])
        if sequence_length - 1 > int(config["model"]["max_seq_len"]):
            raise ValueError("data.sequence_length - 1 exceeds model.max_seq_len")

        distributions = load_saved_distributions(experiment_dir_path, config, device)
        distribution_weights = distributions.new_ones(distributions.shape[0]).cumsum(0)
        distribution_weights = distribution_weights.pow(
            -float(config["data"]["zipf_exponent"])
        )
        bayes_estimator = BayesOptimalEstimator(
            distributions=distributions,
            distribution_weights=distribution_weights,
        ).to(device)
        tokens, distribution_ids, selected_distribution_ids = make_balanced_batch(
            distributions=distributions,
            batch_sequence_length=sequence_length,
            batch_seqs_per_distribution=eval_settings["seqs_per_distribution"],
            batch_distribution_start=eval_settings["distribution_start"],
            batch_max_distributions=eval_settings["max_distributions"],
            eval_seed=eval_settings["eval_seed"],
        )

        rows = []
        for checkpoint_path in checkpoints:
            rows.extend(
                evaluate_checkpoint(
                    checkpoint_path=checkpoint_path,
                    config=config,
                    tokens=tokens,
                    distribution_ids=distribution_ids,
                    selected_distribution_ids=selected_distribution_ids,
                    bayes_estimator=bayes_estimator,
                    bayes_component_chunk_size=eval_settings[
                        "bayes_component_chunk_size"
                    ],
                    microbatch_size=eval_settings["microbatch_size"],
                    device=device,
                )
            )

        output_path = Path(eval_settings["output_csv"]).expanduser()
        if not output_path.is_absolute():
            output_path = experiment_dir_path / output_path
        write_rows_csv(rows, output_path)

        return {
            "rows": rows,
            "csv_path": output_path,
            "num_checkpoints": len(checkpoints),
            "num_distributions": len(selected_distribution_ids),
            "num_sequences": tokens.shape[0],
            "device": str(device),
        }

    return (run_evaluation,)


@app.cell
def _(parse_eval_args, sys):
    experiment_dir = (
        parse_eval_args(sys.argv[1:])
        or "/home/cg5763/data/output_oneshot_memorization/less-tasks-private-eel"
    )
    eval_config = {
        "experiment_dir": experiment_dir,
        "checkpoint_stride": 1,
        "max_checkpoints": 1000,
        "seqs_per_distribution": 2**6,
        "distribution_start": 0,
        "max_distributions": 500,
        "microbatch_size": 8e4,
        "bayes_component_chunk_size": 500,
        "eval_seed": 12345,
        "device_name": "auto",
        "output_csv": "balanced_eval.csv",
        "run_now": True,
    }
    return (eval_config,)


@app.cell
def _(eval_config, run_evaluation):
    result = None
    if eval_config["run_now"]:
        result = run_evaluation(eval_config)
        print(f"wrote {len(result['rows'])} rows to {result['csv_path']}")
        print(
            "evaluated "
            f"{result['num_checkpoints']} checkpoints, "
            f"{result['num_sequences']} sequences, "
            f"{result['num_distributions']} distributions, "
            f"device={result['device']}"
        )
    else:
        print("set run_now = True in the parameter cell to run evaluation")
    return


if __name__ == "__main__":
    app.run()
