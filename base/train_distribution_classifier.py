import argparse
import contextlib
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from base.bit_sequences import (
    ProbabilityVectorAutoencoderMLP,
    ProbabilityVectorClassifierMLP,
    SequenceClassifierMLP,
    SummarySequenceClassifierMLP,
    ZipfBitSequenceGenerator,
)
from base.data_generator import (
    DirichletZipfBinaryClassificationGenerator,
    DirichletZipfBinaryProbabilityVectorGenerator,
    DirichletZipfBinaryVectorProbabilityVectorGenerator,
)
from base.training_utils import (
    append_csv,
    append_csv_rows,
    format_duration,
    load_config,
    make_run_dir,
    make_torch_generator,
    resolve_device as resolve_device_base,
    seed_everything,
)


MODEL_TYPES = {
    "bit_sequence_mlp": SequenceClassifierMLP,
    "probability_autoencoder_mlp": ProbabilityVectorAutoencoderMLP,
    "probability_mlp": ProbabilityVectorClassifierMLP,
    "summary_mlp": SummarySequenceClassifierMLP,
}

DATA_TYPES = {
    "dirichlet_zipf_binary": DirichletZipfBinaryClassificationGenerator,
    "dirichlet_zipf_binary_probability_vector": (
        DirichletZipfBinaryProbabilityVectorGenerator
    ),
    "dirichlet_zipf_binary_vector_probability_vector": (
        DirichletZipfBinaryVectorProbabilityVectorGenerator
    ),
    "zipf_bit_binary": ZipfBitSequenceGenerator,
}

PROBABILITY_VECTOR_DATA_TYPE = "dirichlet_zipf_binary_probability_vector"
VECTOR_LABEL_DATA_TYPE = "dirichlet_zipf_binary_vector_probability_vector"
BIT_SEQUENCE_DATA_TYPE = "zipf_bit_binary"


def is_bit_sequence_config(config):
    return config["data"]["type"] == BIT_SEQUENCE_DATA_TYPE


def is_vector_label_config(config):
    return config["data"]["type"] == VECTOR_LABEL_DATA_TYPE


def is_probability_autoencoder_config(config):
    return config["model"]["type"] == "probability_autoencoder_mlp"


def num_tasks(config):
    if is_bit_sequence_config(config):
        return config["data"]["num_sequences"]
    return config["data"]["num_distributions"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Zipf-sampled classifier.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def resolve_device(name):
    return resolve_device_base(name, prefer_mps=True)


def validate_config(config):
    model = config["model"]
    data = config["data"]
    training = config["training"]
    evaluation = config["evaluation"]

    if model.get("type") not in MODEL_TYPES:
        raise ValueError(f"unknown model.type {model.get('type')!r}")
    if data.get("type") not in DATA_TYPES:
        raise ValueError(f"unknown data.type {data.get('type')!r}")
    is_probability_vector = data["type"] in (
        PROBABILITY_VECTOR_DATA_TYPE,
        VECTOR_LABEL_DATA_TYPE,
    )
    is_bit_sequence = data["type"] == BIT_SEQUENCE_DATA_TYPE
    allowed_model_types = {
        "dirichlet_zipf_binary": "summary_mlp",
        PROBABILITY_VECTOR_DATA_TYPE: (
            "probability_mlp",
            "probability_autoencoder_mlp",
        ),
        VECTOR_LABEL_DATA_TYPE: "probability_mlp",
        BIT_SEQUENCE_DATA_TYPE: "bit_sequence_mlp",
    }[data["type"]]
    if isinstance(allowed_model_types, str):
        allowed_model_types = (allowed_model_types,)
    if model["type"] not in allowed_model_types:
        raise ValueError(
            f"data.type {data['type']!r} requires model.type "
            f"in {allowed_model_types!r}"
        )
    if not is_bit_sequence and model["vocab_size"] != data["num_states"]:
        raise ValueError("model.vocab_size must equal data.num_states")
    if (
        not is_probability_vector
        and model["sequence_length"] != data["sequence_length"]
    ):
        raise ValueError("model.sequence_length must equal data.sequence_length")
    label_scheme = data.get("label_scheme", "binary")
    if label_scheme not in ("binary", "identity"):
        raise ValueError("data.label_scheme must be 'binary' or 'identity'")
    if is_probability_autoencoder_config(config):
        expected_classes = None
    elif is_vector_label_config(config):
        d_label = data.get("d_label")
        if not isinstance(d_label, int) or isinstance(d_label, bool) or d_label < 1:
            raise ValueError("data.d_label must be a positive integer")
        if label_scheme != "binary":
            raise ValueError("vector labels require data.label_scheme 'binary'")
        if model.get("loss") != "mse":
            raise ValueError("vector labels require model.loss 'mse'")
        expected_classes = d_label
    else:
        expected_classes = 2 if label_scheme == "binary" else num_tasks(config)
    if expected_classes is not None and (
        model.get("num_classes", expected_classes) != expected_classes
    ):
        if is_vector_label_config(config):
            raise ValueError(
                "model.num_classes must match the configured label dimension"
            )
        raise ValueError(
            "model.num_classes must be 2 for binary labels or "
            "the number of tasks for identity labels"
        )
    if training["max_iters"] < 1:
        raise ValueError("training.max_iters must be at least 1")
    checkpoint_spacing = training.get("checkpoint_spacing", "linear")
    if checkpoint_spacing not in ("linear", "logarithmic"):
        raise ValueError(
            "training.checkpoint_spacing must be 'linear' or 'logarithmic'"
        )
    if checkpoint_spacing == "linear" and training["checkpoint_interval"] < 1:
        raise ValueError("training.checkpoint_interval must be at least 1")
    checkpoint_points_per_decade = training.get(
        "checkpoint_points_per_decade", 10
    )
    if checkpoint_spacing == "logarithmic" and (
        not isinstance(checkpoint_points_per_decade, int)
        or isinstance(checkpoint_points_per_decade, bool)
        or checkpoint_points_per_decade < 1
    ):
        raise ValueError(
            "training.checkpoint_points_per_decade must be a positive integer"
        )
    evaluation_spacing = evaluation.get("spacing", "linear")
    if evaluation_spacing not in ("linear", "logarithmic"):
        raise ValueError(
            "evaluation.spacing must be 'linear' or 'logarithmic'"
        )
    if evaluation_spacing == "linear" and evaluation["interval"] < 1:
        raise ValueError("evaluation.interval must be at least 1")
    points_per_decade = evaluation.get("points_per_decade", 10)
    if evaluation_spacing == "logarithmic" and (
        not isinstance(points_per_decade, int)
        or isinstance(points_per_decade, bool)
        or points_per_decade < 1
    ):
        raise ValueError(
            "evaluation.points_per_decade must be a positive integer"
        )
    if is_probability_vector or is_bit_sequence:
        if evaluation.get("seqs_per_distribution", 1) != 1:
            raise ValueError(
                "evaluation.seqs_per_distribution must be 1 for deterministic "
                "vector data"
            )
    elif evaluation["seqs_per_distribution"] < 1:
        raise ValueError("evaluation.seqs_per_distribution must be at least 1")
    if training.get("log_interval", 1) < 1:
        raise ValueError("training.log_interval must be at least 1")
    if not isinstance(training.get("compile", False), bool):
        raise ValueError("training.compile must be a boolean")
    amp_dtype = training.get("amp_dtype")
    if amp_dtype not in (None, "bf16", "fp16"):
        raise ValueError("training.amp_dtype must be one of null, 'bf16', or 'fp16'")


def build_data_generator(config, device, generator, checkpoint=None):
    data = dict(config["data"])
    data.pop("type")
    data.pop("label_scheme", None)
    data.pop("batch_size")
    if is_bit_sequence_config(config):
        if checkpoint is not None:
            data["sequences"] = checkpoint["sequences"].to(device)
            data["labels"] = checkpoint["labels"].to(device)
    else:
        data.pop("sequence_length", None)
        if checkpoint is not None:
            data["distributions"] = checkpoint["distributions"].to(device)
            data["distribution_labels"] = checkpoint["distribution_labels"].to(
                device
            )
    return DATA_TYPES[config["data"]["type"]](
        **data,
        device=device,
        generator=generator,
    )


def build_model(config):
    model = dict(config["model"])
    model_type = model.pop("type")
    return MODEL_TYPES[model_type](**model)


def logarithmic_iterations(max_iters, points_per_decade):
    """Return integer iterations spaced uniformly on a base-10 log scale."""
    max_exponent = math.ceil(points_per_decade * math.log10(max_iters))
    iterations = {
        round(10 ** (exponent / points_per_decade))
        for exponent in range(max_exponent + 1)
    }
    iterations = {iteration for iteration in iterations if iteration <= max_iters}
    iterations.add(max_iters)
    return iterations


def logarithmic_evaluation_iterations(max_iters, points_per_decade):
    return logarithmic_iterations(max_iters, points_per_decade)


def build_optimizer(model, config, device):
    opt = config["optimizer"]
    if opt.get("type", "adam") != "adam":
        raise ValueError("optimizer.type must be 'adam'")
    optimizer_kwargs = {
        "lr": opt["lr"],
        "betas": tuple(opt.get("betas", [0.9, 0.999])),
        "weight_decay": opt.get("weight_decay", 0.0),
    }
    if opt.get("fused", False):
        if device.type != "cuda":
            raise ValueError("optimizer.fused requires a CUDA device")
        optimizer_kwargs["fused"] = True
    try:
        return torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    except TypeError as error:
        if optimizer_kwargs.get("fused"):
            raise ValueError(
                "optimizer.fused is not supported by this PyTorch build"
            ) from error
        raise


def resolve_amp_dtype(config, device):
    amp_dtype = config["training"].get("amp_dtype")
    if amp_dtype is None:
        return None
    if device.type != "cuda":
        raise ValueError("training.amp_dtype requires a CUDA device")
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[amp_dtype]


def make_grad_scaler(config, device):
    enabled = device.type == "cuda" and config["training"].get("amp_dtype") == "fp16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def autocast_context(device, amp_dtype):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=amp_dtype)


def make_compiled_training_step(
    model,
    optimizer,
    grad_scaler,
    device,
    amp_dtype,
    grad_clip_norm,
):
    """Compile one complete parameter update, excluding data generation."""

    def training_step(tokens, labels):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_dtype):
            loss = model(tokens, targets=labels)["loss"]
        if grad_scaler.is_enabled():
            grad_scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        return loss

    compiled_step = torch.compile(training_step)
    if amp_dtype is None:
        return compiled_step

    def compiled_step_with_amp_semantics(tokens, labels):
        # AMP wraps only forward/loss; backward executes outside autocast.
        with torch._functorch.config.patch(backward_pass_autocast="off"):
            return compiled_step(tokens, labels)

    return compiled_step_with_amp_semantics


def rng_state(train_generator):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "train_generator": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if train_generator is not None:
        state["train_generator"] = train_generator.get_state()
    return state


def set_rng_state(state, train_generator):
    if not state:
        return
    random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if train_generator is not None and state.get("train_generator") is not None:
        train_generator.set_state(state["train_generator"])


def save_checkpoint(
    path,
    model,
    optimizer,
    iteration,
    config,
    run_dir,
    data_generator,
    presentation_counts,
    train_generator,
    eval_batch,
):
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
        "config": config,
        "run_dir": str(run_dir),
        "rng_state": rng_state(train_generator),
        "presentation_counts": presentation_counts.cpu(),
        "eval_batch": {key: value.cpu() for key, value in eval_batch.items()},
    }
    if is_bit_sequence_config(config):
        state["sequences"] = data_generator.sequences.cpu()
        state["labels"] = data_generator.labels.cpu()
    else:
        state["distributions"] = data_generator.distributions.cpu()
        state["distribution_labels"] = data_generator.distribution_labels.cpu()
    torch.save(state, path)


def make_eval_batch(data_generator, config, eval_generator):
    if is_bit_sequence_config(config):
        num_sequences = min(
            config["evaluation"].get("num_sequences", num_tasks(config)),
            num_tasks(config),
        )
        ids = torch.arange(num_sequences, device=data_generator.device)
        inputs = data_generator.sample_from_sequence_ids(ids)
        labels = (
            ids
            if config["data"].get("label_scheme", "binary") == "identity"
            else data_generator.labels[ids]
        )
        return {"tokens": inputs, "labels": labels, "distribution_ids": ids}

    num_distributions = min(
        config["evaluation"].get(
            "num_distributions",
            config["data"]["num_distributions"],
        ),
        config["data"]["num_distributions"],
    )
    seqs_per_distribution = config["evaluation"].get("seqs_per_distribution", 1)
    ids = torch.arange(num_distributions, device=data_generator.device)
    ids = ids.repeat_interleave(seqs_per_distribution)

    if config["data"]["type"] in (
        PROBABILITY_VECTOR_DATA_TYPE,
        VECTOR_LABEL_DATA_TYPE,
    ):
        inputs, binary_labels = data_generator.sample_from_distribution_ids(
            ids,
            return_labels=True,
        )
    else:
        train_generator = data_generator.generator
        data_generator.generator = eval_generator
        try:
            inputs, binary_labels = data_generator.sample_from_distribution_ids(
                ids,
                sequence_length=config["data"]["sequence_length"],
                return_labels=True,
            )
        finally:
            data_generator.generator = train_generator
    labels = (
        ids
        if config["data"].get("label_scheme", "binary") == "identity"
        else binary_labels
    )
    # Keep the existing checkpoint key for backward-compatible resume behavior.
    return {"tokens": inputs, "labels": labels, "distribution_ids": ids}


@torch.no_grad()
def evaluate(
    model,
    data_generator,
    config,
    eval_batch,
    presentation_counts,
    log_path,
    iteration,
    amp_dtype,
):
    model.eval()
    inputs = eval_batch["tokens"]
    labels = eval_batch["labels"]
    ids = eval_batch["distribution_ids"]
    if is_vector_label_config(config):
        accuracies = []
        microbatch = config["evaluation"].get("microbatch_size", ids.numel())
        for start in range(0, ids.numel(), microbatch):
            stop = min(start + microbatch, ids.numel())
            with autocast_context(inputs.device, amp_dtype):
                predictions = model(inputs[start:stop])["logits"]
            accuracies.append(
                (predictions.sign() * labels[start:stop]).mean(dim=1)
            )
        accuracies = torch.cat(accuracies).cpu()
        counts_cpu = presentation_counts[: ids.numel()].cpu()
        rows = [
            {
                "iter": iteration,
                "distribution_id": distribution_id,
                "accuracy": f"{accuracies[distribution_id].item():.8f}",
                "training_seen_count": int(counts_cpu[distribution_id].item()),
            }
            for distribution_id in range(ids.numel())
        ]
        append_csv_rows(
            log_path,
            ["iter", "distribution_id", "accuracy", "training_seen_count"],
            rows,
        )
        model.train()
        return
    if is_probability_autoencoder_config(config):
        losses = []
        microbatch = config["evaluation"].get("microbatch_size", ids.numel())
        for start in range(0, ids.numel(), microbatch):
            stop = min(start + microbatch, ids.numel())
            with autocast_context(inputs.device, amp_dtype):
                losses.append(
                    model(
                        inputs[start:stop],
                        targets=inputs[start:stop],
                        loss_reduction="none",
                    )["loss"]
                )
        losses = torch.cat(losses).cpu()
        counts_cpu = presentation_counts[: ids.numel()].cpu()
        rows = [
            {
                "iter": iteration,
                "distribution_id": distribution_id,
                "loss": f"{losses[distribution_id].item():.8f}",
                "training_seen_count": int(counts_cpu[distribution_id].item()),
            }
            for distribution_id in range(ids.numel())
        ]
        append_csv_rows(
            log_path,
            ["iter", "distribution_id", "loss", "training_seen_count"],
            rows,
        )
        model.train()
        return
    losses = []
    microbatch = config["evaluation"].get("microbatch_size", ids.numel())
    for start in range(0, ids.numel(), microbatch):
        stop = min(start + microbatch, ids.numel())
        with autocast_context(inputs.device, amp_dtype):
            logits = model(inputs[start:stop])["logits"]
            losses.append(
                F.cross_entropy(logits, labels[start:stop], reduction="none")
            )
    losses = torch.cat(losses)
    num_distributions = int(ids.max().item()) + 1
    loss_sums = torch.zeros(
        num_distributions, device=losses.device, dtype=losses.dtype
    )
    loss_sums.scatter_add_(0, ids, losses)
    loss_counts = torch.bincount(ids, minlength=num_distributions).clamp_min(1)
    mean_losses = (loss_sums / loss_counts.to(losses.dtype)).cpu()
    if config["data"].get("label_scheme", "binary") == "identity":
        labels_cpu = torch.arange(num_distributions)
    elif is_bit_sequence_config(config):
        labels_cpu = data_generator.labels[:num_distributions].cpu()
    else:
        labels_cpu = data_generator.distribution_labels[:num_distributions].cpu()
    counts_cpu = presentation_counts[:num_distributions].cpu()

    id_field = "sequence_id" if is_bit_sequence_config(config) else "distribution_id"
    fieldnames = [
        "iter",
        id_field,
        "label",
        "loss",
        "training_seen_count",
    ]
    rows = [
        {
            "iter": iteration,
            id_field: distribution_id,
            "label": int(labels_cpu[distribution_id].item()),
            "loss": f"{mean_losses[distribution_id].item():.8f}",
            "training_seen_count": int(counts_cpu[distribution_id].item()),
        }
        for distribution_id in range(num_distributions)
    ]
    append_csv_rows(log_path, fieldnames, rows)
    model.train()


def move_eval_batch(eval_batch, device):
    return {key: value.to(device) for key, value in eval_batch.items()}


def resolve_run_dir(config, checkpoint):
    if checkpoint is not None:
        return Path(checkpoint["run_dir"])
    if config["run"].get("run_dir"):
        run_dir = Path(config["run"]["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir
    return make_run_dir(config)


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    validate_config(config)

    resume_from = args.resume_from or config["run"].get("resume_from")
    checkpoint = None
    if resume_from:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)

    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(config.get("device", "auto"))
    start_iter = 0 if checkpoint is None else int(checkpoint["iteration"])
    train_generator = make_torch_generator(device, seed)
    eval_generator = make_torch_generator(
        device,
        config["evaluation"].get("seed", seed + 1),
    )

    run_dir = resolve_run_dir(config, checkpoint)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint is None:
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    data_generator = build_data_generator(config, device, train_generator, checkpoint)
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config, device)
    amp_dtype = resolve_amp_dtype(config, device)
    grad_scaler = make_grad_scaler(config, device)
    presentation_counts = torch.zeros(
        num_tasks(config),
        dtype=torch.long,
        device=device,
    )
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        presentation_counts = checkpoint["presentation_counts"].to(
            device=device,
            dtype=torch.long,
        )
        set_rng_state(checkpoint.get("rng_state"), train_generator)
        eval_batch = move_eval_batch(checkpoint["eval_batch"], device)
    else:
        eval_batch = make_eval_batch(data_generator, config, eval_generator)

    compile_training = config["training"].get("compile", False)
    compiled_training_step = None
    if compile_training:
        grad_clip_norm = config["training"].get("grad_clip_norm")
        if grad_clip_norm is not None:
            grad_clip_norm = float(grad_clip_norm)
        compiled_training_step = make_compiled_training_step(
            model,
            optimizer,
            grad_scaler,
            device,
            amp_dtype,
            grad_clip_norm,
        )

    train_log = run_dir / "train_log.csv"
    eval_log = run_dir / (
        "eval_by_sequence.csv"
        if is_bit_sequence_config(config)
        else "eval_by_distribution.csv"
    )
    start_time = time.perf_counter()
    last_report_time = start_time
    last_report_iter = start_iter
    max_iters = int(config["training"]["max_iters"])
    report_interval = config["training"].get(
        "report_interval",
        config["evaluation"].get("interval", max(1, max_iters // 10)),
    )
    log_interval = config["training"].get("log_interval", report_interval)
    logarithmic_eval_iterations = None
    if config["evaluation"].get("spacing", "linear") == "logarithmic":
        logarithmic_eval_iterations = logarithmic_evaluation_iterations(
            max_iters,
            config["evaluation"].get("points_per_decade", 10),
        )
    logarithmic_checkpoint_iterations = None
    if config["training"].get("checkpoint_spacing", "linear") == "logarithmic":
        logarithmic_checkpoint_iterations = logarithmic_iterations(
            max_iters,
            config["training"].get("checkpoint_points_per_decade", 10),
        )
    model.train()
    for iteration in range(start_iter + 1, max_iters + 1):
        batch_size = config["data"]["batch_size"]
        if is_bit_sequence_config(config):
            inputs, distribution_ids = data_generator.sample(
                batch_size=batch_size,
                return_sequence_ids=True,
            )
            binary_labels = data_generator.labels[distribution_ids]
        elif config["data"]["type"] in (
            PROBABILITY_VECTOR_DATA_TYPE,
            VECTOR_LABEL_DATA_TYPE,
        ):
            inputs, distribution_ids, binary_labels = data_generator.sample(
                batch_size=batch_size,
                return_distribution_ids=True,
                return_labels=True,
            )
        else:
            inputs, distribution_ids, binary_labels = data_generator.sample(
                batch_size=batch_size,
                sequence_length=config["data"]["sequence_length"],
                return_distribution_ids=True,
                return_labels=True,
            )
        labels = (
            distribution_ids
            if config["data"].get("label_scheme", "binary") == "identity"
            else binary_labels
        )
        targets = inputs if is_probability_autoencoder_config(config) else labels
        presentation_counts += torch.bincount(
            distribution_ids,
            minlength=num_tasks(config),
        )
        if compile_training:
            loss = compiled_training_step(inputs, targets)
        else:
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_dtype):
                loss = model(inputs, targets=targets)["loss"]
            grad_scaler.scale(loss).backward()
            if config["training"].get("grad_clip_norm") is not None:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config["training"]["grad_clip_norm"]),
                )
            grad_scaler.step(optimizer)
            grad_scaler.update()

        should_report = (
            iteration == start_iter + 1
            or iteration % report_interval == 0
            or iteration == max_iters
        )
        should_log_train = (
            iteration == start_iter + 1
            or iteration % log_interval == 0
            or iteration == max_iters
        )
        if logarithmic_checkpoint_iterations is None:
            should_checkpoint = (
                iteration % config["training"]["checkpoint_interval"] == 0
                or iteration == max_iters
            )
        else:
            should_checkpoint = iteration in logarithmic_checkpoint_iterations
        loss_value = None
        if should_log_train or should_report or should_checkpoint:
            loss_value = loss.item()

        if should_log_train:
            append_csv(
                train_log,
                ["iter", "loss", "lr", "time_sec"],
                {
                    "iter": iteration,
                    "loss": f"{loss_value:.8f}",
                    "lr": optimizer.param_groups[0]["lr"],
                    "time_sec": f"{time.perf_counter() - start_time:.4f}",
                },
            )
        if should_report:
            now = time.perf_counter()
            recent_iters = iteration - last_report_iter
            recent_sec_per_iter = (now - last_report_time) / recent_iters
            total_sec_per_iter = (now - start_time) / (iteration - start_iter)
            eta = format_duration((max_iters - iteration) * total_sec_per_iter)
            print(
                f"iter {iteration}/{max_iters} "
                f"loss {loss_value:.6f} "
                f"sec_per_iter {recent_sec_per_iter:.4f} "
                f"avg_sec_per_iter {total_sec_per_iter:.4f} "
                f"eta {eta}",
                flush=True,
            )
            last_report_time = now
            last_report_iter = iteration
        should_evaluate = iteration == start_iter + 1
        if logarithmic_eval_iterations is None:
            should_evaluate = should_evaluate or (
                iteration % config["evaluation"]["interval"] == 0
            )
        else:
            should_evaluate = (
                should_evaluate or iteration in logarithmic_eval_iterations
            )
        if should_evaluate:
            eval_start_time = time.perf_counter()
            evaluate(
                model,
                data_generator,
                config,
                eval_batch,
                presentation_counts,
                eval_log,
                iteration,
                amp_dtype,
            )
            print(
                f"evaluation iter {iteration} "
                f"time_sec {time.perf_counter() - eval_start_time:.4f}",
                flush=True,
            )
        if should_checkpoint:
            for path in (
                checkpoint_dir / f"checkpoint_{iteration:06d}.pt",
                checkpoint_dir / "latest.pt",
            ):
                save_checkpoint(
                    path,
                    model,
                    optimizer,
                    iteration,
                    config,
                    run_dir,
                    data_generator,
                    presentation_counts,
                    train_generator,
                    eval_batch,
                )
            print(f"checkpoint iter {iteration} loss {loss_value:.6f}", flush=True)

    print(f"run directory: {run_dir}")


if __name__ == "__main__":
    main()
