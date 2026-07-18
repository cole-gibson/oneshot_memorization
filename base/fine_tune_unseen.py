import argparse
import copy
import csv
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from base.train_distribution_classifier import (
    BIT_SEQUENCE_DATA_TYPE,
    DATA_TYPES,
    PROBABILITY_VECTOR_DATA_TYPE,
    VECTOR_LABEL_DATA_TYPE,
    autocast_context,
    build_data_generator,
    build_model,
    build_optimizer,
    is_probability_autoencoder_config,
    is_vector_label_config,
    logarithmic_evaluation_iterations,
    make_grad_scaler,
    resolve_amp_dtype,
    resolve_device,
    validate_config,
)
from base.training_utils import load_config, make_torch_generator, seed_everything


CHECKPOINT_PATTERN = re.compile(r"checkpoint_(\d+)\.pt$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune every checkpoint from one run on unseen items."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_fine_tune_config(config, training_config):
    fine_tuning = config.get("fine_tuning")
    if not isinstance(fine_tuning, dict):
        raise ValueError("fine_tuning must be a YAML mapping")
    for name in ("num_unseen_items", "max_iters"):
        value = fine_tuning.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"fine_tuning.{name} must be a positive integer")
    for name in ("log_interval", "report_interval"):
        value = fine_tuning.get(name, 1)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"fine_tuning.{name} must be a positive integer")
    checkpoint_iterations = fine_tuning.get("checkpoint_iterations")
    if checkpoint_iterations is not None and (
        not isinstance(checkpoint_iterations, list)
        or not checkpoint_iterations
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in checkpoint_iterations
        )
    ):
        raise ValueError(
            "fine_tuning.checkpoint_iterations must be a nonempty list of "
            "nonnegative integers or null"
        )
    evaluation = config.get("evaluation")
    optimizer = config.get("optimizer")
    early_stopping = config.get("early_stopping")
    evaluation = {} if evaluation is None else evaluation
    optimizer = {} if optimizer is None else optimizer
    early_stopping = {} if early_stopping is None else early_stopping
    for name, value in (
        ("evaluation", evaluation),
        ("optimizer", optimizer),
        ("early_stopping", early_stopping),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a YAML mapping")
    threshold = early_stopping.get("threshold")
    if threshold is not None and (
        not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
    ):
        raise ValueError("early_stopping.threshold must be numeric or null")
    if training_config["data"]["type"] == BIT_SEQUENCE_DATA_TYPE:
        raise ValueError("fine-tuning unseen items does not support zipf_bit_binary")
    if training_config["data"].get("label_scheme", "binary") != "binary":
        raise ValueError("fine-tuning unseen items requires binary label_scheme")
    validation_config = copy.deepcopy(training_config)
    validation_config["training"]["max_iters"] = fine_tuning["max_iters"]
    validation_config["training"]["amp_dtype"] = fine_tuning.get(
        "amp_dtype",
        validation_config["training"].get("amp_dtype"),
    )
    validation_config["evaluation"].update(evaluation)
    validate_config(validation_config)


def checkpoint_paths(run_dir, requested_iterations=None):
    paths_by_iteration = {}
    for path in (run_dir / "checkpoints").glob("checkpoint_*.pt"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match:
            paths_by_iteration[int(match.group(1))] = path
    if not paths_by_iteration:
        raise ValueError(f"no numbered checkpoints found in {run_dir / 'checkpoints'}")
    if requested_iterations is None:
        return sorted(paths_by_iteration.items())
    missing = sorted(set(requested_iterations) - paths_by_iteration.keys())
    if missing:
        raise ValueError(f"checkpoint iterations not found: {missing}")
    return [(iteration, paths_by_iteration[iteration]) for iteration in requested_iterations]


def evaluation_iterations(max_iters, evaluation):
    if evaluation.get("spacing", "linear") == "logarithmic":
        return logarithmic_evaluation_iterations(
            max_iters,
            evaluation.get("points_per_decade", 10),
        )
    interval = int(evaluation["interval"])
    return {1, *range(interval, max_iters + 1, interval)}


def threshold_reached(metric_name, metric_value, threshold):
    if threshold is None:
        return False
    if metric_name == "loss":
        return metric_value <= threshold
    if metric_name == "accuracy":
        return metric_value >= threshold
    raise ValueError(f"unknown evaluation metric {metric_name!r}")


def replace_batch_item(inputs, targets, unseen_input, unseen_target):
    inputs = inputs.clone()
    targets = targets.clone()
    inputs[0] = unseen_input[0]
    targets[0] = unseen_target[0]
    return inputs, targets


def _generator_kwargs(training_config, num_distributions, device, generator, items=None):
    data = dict(training_config["data"])
    data_type = data.pop("type")
    data.pop("label_scheme", None)
    data.pop("batch_size", None)
    data.pop("sequence_length", None)
    data["num_distributions"] = num_distributions
    data["device"] = device
    data["generator"] = generator
    if items is not None:
        data["distributions"] = items["distributions"].to(device)
        data["distribution_labels"] = items["distribution_labels"].to(device)
    return DATA_TYPES[data_type](**data)


def make_unseen_items(training_config, count, seed):
    generator = make_torch_generator(torch.device("cpu"), seed)
    data_generator = _generator_kwargs(
        training_config,
        count,
        torch.device("cpu"),
        generator,
    )
    return {
        "distributions": data_generator.distributions.cpu(),
        "distribution_labels": data_generator.distribution_labels.cpu(),
    }


def sample_normal_batch(data_generator, config, batch_size):
    if config["data"]["type"] in (
        PROBABILITY_VECTOR_DATA_TYPE,
        VECTOR_LABEL_DATA_TYPE,
    ):
        inputs, distribution_ids, labels = data_generator.sample(
            batch_size=batch_size,
            return_distribution_ids=True,
            return_labels=True,
        )
    else:
        inputs, _, labels = data_generator.sample(
            batch_size=batch_size,
            sequence_length=config["data"]["sequence_length"],
            return_distribution_ids=True,
            return_labels=True,
        )
    if is_probability_autoencoder_config(config):
        targets = (
            data_generator.distributions[distribution_ids]
            if config["data"].get("noise_enabled", False)
            else inputs
        )
    else:
        targets = labels
    return inputs, targets


def sample_unseen_item(data_generator, config, item_id, count=1):
    ids = torch.full((count,), item_id, device=data_generator.device, dtype=torch.long)
    if config["data"]["type"] in (
        PROBABILITY_VECTOR_DATA_TYPE,
        VECTOR_LABEL_DATA_TYPE,
    ):
        inputs, labels = data_generator.sample_from_distribution_ids(
            ids,
            return_labels=True,
        )
    else:
        inputs, labels = data_generator.sample_from_distribution_ids(
            ids,
            sequence_length=config["data"]["sequence_length"],
            return_labels=True,
        )
    if is_probability_autoencoder_config(config):
        targets = (
            data_generator.distributions[ids]
            if config["data"].get("noise_enabled", False)
            else inputs
        )
    else:
        targets = labels
    return inputs, targets


@torch.no_grad()
def evaluate_unseen(model, inputs, targets, config, amp_dtype):
    was_training = model.training
    model.eval()
    with autocast_context(inputs.device, amp_dtype):
        if is_vector_label_config(config):
            predictions = model(inputs)["logits"]
            value = (predictions.sign() * targets).mean().item()
            metric_name = "accuracy"
        elif is_probability_autoencoder_config(config):
            value = model(inputs, targets=targets)["loss"].item()
            metric_name = "loss"
        else:
            logits = model(inputs)["logits"]
            value = F.cross_entropy(logits, targets).item()
            metric_name = "loss"
    model.train(was_training)
    return metric_name, value


def append_row(path, fieldnames, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _trajectory_seed(seed, checkpoint_index, item_id, num_items):
    return seed + 1 + checkpoint_index * num_items + item_id


def run_trajectory(
    checkpoint,
    checkpoint_iteration,
    checkpoint_index,
    item_id,
    items,
    experiment_config,
    output_dir,
    device,
):
    training_config = copy.deepcopy(checkpoint["config"])
    fine_tuning = experiment_config["fine_tuning"]
    seed = _trajectory_seed(
        int(experiment_config.get("seed", 0)),
        checkpoint_index,
        item_id,
        fine_tuning["num_unseen_items"],
    )
    seed_everything(seed)
    train_generator = make_torch_generator(device, seed)
    eval_generator = make_torch_generator(
        device,
        int(experiment_config.get("seed", 0)) + 10_000 + item_id,
    )
    data_generator = build_data_generator(
        training_config,
        device,
        train_generator,
        checkpoint,
    )
    unseen_train_generator = _generator_kwargs(
        training_config,
        fine_tuning["num_unseen_items"],
        device,
        train_generator,
        items,
    )
    unseen_eval_generator = _generator_kwargs(
        training_config,
        fine_tuning["num_unseen_items"],
        device,
        eval_generator,
        items,
    )
    model = build_model(training_config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    optimizer_config = copy.deepcopy(training_config)
    optimizer_config["optimizer"].update(experiment_config.get("optimizer") or {})
    optimizer_config["training"]["amp_dtype"] = fine_tuning.get(
        "amp_dtype",
        optimizer_config["training"].get("amp_dtype"),
    )
    optimizer = build_optimizer(model, optimizer_config, device)
    amp_dtype = resolve_amp_dtype(optimizer_config, device)
    grad_scaler = make_grad_scaler(optimizer_config, device)
    grad_clip_norm = fine_tuning.get(
        "grad_clip_norm",
        training_config["training"].get("grad_clip_norm"),
    )

    evaluation = copy.deepcopy(training_config["evaluation"])
    evaluation.update(experiment_config.get("evaluation") or {})
    eval_count = evaluation.get("seqs_per_distribution", 1)
    eval_inputs, eval_targets = sample_unseen_item(
        unseen_eval_generator,
        training_config,
        item_id,
        count=eval_count,
    )
    eval_iters = evaluation_iterations(fine_tuning["max_iters"], evaluation)
    threshold = (experiment_config.get("early_stopping") or {}).get("threshold")
    train_log = output_dir / "train_log.csv"
    eval_log = output_dir / "eval_log.csv"
    train_fields = [
        "checkpoint_iter",
        "unseen_item_id",
        "fine_tune_iter",
        "loss",
        "lr",
        "time_sec",
    ]
    eval_fields = [
        "checkpoint_iter",
        "unseen_item_id",
        "fine_tune_iter",
        "metric_name",
        "metric_value",
        "threshold_reached",
    ]
    start_time = time.perf_counter()

    metric_name, metric_value = evaluate_unseen(
        model,
        eval_inputs,
        eval_targets,
        training_config,
        amp_dtype,
    )
    stopped = threshold_reached(metric_name, metric_value, threshold)
    append_row(
        eval_log,
        eval_fields,
        {
            "checkpoint_iter": checkpoint_iteration,
            "unseen_item_id": item_id,
            "fine_tune_iter": 0,
            "metric_name": metric_name,
            "metric_value": f"{metric_value:.8f}",
            "threshold_reached": int(stopped),
        },
    )
    completed_iters = 0
    model.train()
    for iteration in range(1, fine_tuning["max_iters"] + 1):
        if stopped:
            break
        inputs, targets = sample_normal_batch(
            data_generator,
            training_config,
            training_config["data"]["batch_size"],
        )
        unseen_input, unseen_target = sample_unseen_item(
            unseen_train_generator,
            training_config,
            item_id,
        )
        inputs, targets = replace_batch_item(
            inputs,
            targets,
            unseen_input,
            unseen_target,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_dtype):
            loss = model(inputs, targets=targets)["loss"]
        grad_scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        grad_scaler.step(optimizer)
        grad_scaler.update()
        completed_iters = iteration

        if (
            iteration == 1
            or iteration % fine_tuning.get("log_interval", 1) == 0
            or iteration == fine_tuning["max_iters"]
        ):
            append_row(
                train_log,
                train_fields,
                {
                    "checkpoint_iter": checkpoint_iteration,
                    "unseen_item_id": item_id,
                    "fine_tune_iter": iteration,
                    "loss": f"{loss.item():.8f}",
                    "lr": optimizer.param_groups[0]["lr"],
                    "time_sec": f"{time.perf_counter() - start_time:.4f}",
                },
            )
        if iteration in eval_iters:
            metric_name, metric_value = evaluate_unseen(
                model,
                eval_inputs,
                eval_targets,
                training_config,
                amp_dtype,
            )
            stopped = threshold_reached(metric_name, metric_value, threshold)
            append_row(
                eval_log,
                eval_fields,
                {
                    "checkpoint_iter": checkpoint_iteration,
                    "unseen_item_id": item_id,
                    "fine_tune_iter": iteration,
                    "metric_name": metric_name,
                    "metric_value": f"{metric_value:.8f}",
                    "threshold_reached": int(stopped),
                },
            )
        if (
            iteration == 1
            or iteration % fine_tuning.get("report_interval", 1) == 0
            or iteration == fine_tuning["max_iters"]
            or stopped
        ):
            print(
                f"checkpoint {checkpoint_iteration} item {item_id} "
                f"iter {iteration}/{fine_tuning['max_iters']} loss {loss.item():.6f}",
                flush=True,
            )
    return {
        "checkpoint_iter": checkpoint_iteration,
        "unseen_item_id": item_id,
        "completed_iters": completed_iters,
        "metric_name": metric_name,
        "final_metric_value": f"{metric_value:.8f}",
        "threshold_reached": int(stopped),
    }


def run(run_dir, experiment_config, output_dir):
    fine_tuning = experiment_config.get("fine_tuning") or {}
    checkpoint_specs = checkpoint_paths(
        run_dir,
        fine_tuning.get("checkpoint_iterations"),
    )
    first_checkpoint = torch.load(
        checkpoint_specs[0][1],
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    training_config = copy.deepcopy(first_checkpoint["config"])
    validate_config(training_config)
    validate_fine_tune_config(experiment_config, training_config)

    output_dir.mkdir(parents=True, exist_ok=False)
    saved_config = copy.deepcopy(experiment_config)
    saved_config["source_run_dir"] = str(run_dir.resolve())
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(saved_config, config_file, sort_keys=False)
    items = make_unseen_items(
        training_config,
        experiment_config["fine_tuning"]["num_unseen_items"],
        int(experiment_config.get("seed", 0)),
    )
    torch.save(items, output_dir / "unseen_items.pt")

    device = resolve_device(experiment_config.get("device", "auto"))
    summary_path = output_dir / "summary.csv"
    summary_fields = [
        "checkpoint_iter",
        "unseen_item_id",
        "completed_iters",
        "metric_name",
        "final_metric_value",
        "threshold_reached",
    ]
    for checkpoint_index, (iteration, path) in enumerate(checkpoint_specs):
        checkpoint = (
            first_checkpoint
            if checkpoint_index == 0
            else torch.load(
                path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        )
        if checkpoint["config"]["data"] != training_config["data"]:
            raise ValueError("checkpoint data configs differ within the training run")
        for item_id in range(experiment_config["fine_tuning"]["num_unseen_items"]):
            row = run_trajectory(
                checkpoint,
                iteration,
                checkpoint_index,
                item_id,
                items,
                experiment_config,
                output_dir,
                device,
            )
            append_row(summary_path, summary_fields, row)
        if checkpoint_index == 0:
            first_checkpoint = None
        del checkpoint
    print(f"fine-tuning output directory: {output_dir}")


def main():
    args = parse_args()
    run(args.run_dir, load_config(args.config), args.output_dir)


if __name__ == "__main__":
    main()
