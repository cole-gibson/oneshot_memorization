import argparse
import contextlib
import csv
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from base.bit_sequences import SummarySequenceClassifierMLP
from base.data_generator import DirichletZipfBinaryClassificationGenerator
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
    "summary_mlp": SummarySequenceClassifierMLP,
}

DATA_TYPES = {
    "dirichlet_zipf_binary": DirichletZipfBinaryClassificationGenerator,
}

TRAINING_PHASES = (
    "data_sample",
    "label_and_count_bookkeeping",
    "zero_grad",
    "forward_and_loss",
    "backward",
    "grad_unscale_and_clip",
    "optimizer_step_and_scaler_update",
    "training_step_total",
)

TIMING_PHASES = TRAINING_PHASES + (
    "train_logging_and_reporting",
    "evaluation_forward",
    "evaluation_aggregation",
    "evaluation_csv_write",
    "checkpoint_numbered",
    "checkpoint_latest",
    "data_generator_init",
    "model_and_optimizer_init",
    "resume_restore",
    "eval_batch_init",
    "compile_first_step",
)

TIMING_FIELDS = (
    "start_iter",
    "end_iter",
    "phase",
    "backend",
    "num_calls",
    "num_examples",
    "total_ms",
    "mean_ms",
    "fraction_of_step",
)


class PhaseTimingCollector:
    """Aggregate CPU wall times or asynchronous CUDA event timings."""

    def __init__(self, enabled, device, path, measure_start, measure_end):
        self.enabled = enabled
        self.device = device
        self.path = path
        self.measure_start = measure_start
        self.measure_end = measure_end
        self.records = defaultdict(
            lambda: {
                "total_ms": 0.0,
                "num_calls": 0,
                "num_examples": 0,
                "start_iter": None,
                "end_iter": None,
                "backends": set(),
            }
        )
        self.pending_cuda = []

    def is_training_iteration(self, iteration):
        return (
            self.enabled
            and self.measure_start <= iteration <= self.measure_end
        )

    def _add(self, phase, elapsed_ms, iteration, num_examples, backend):
        record = self.records[phase]
        record["total_ms"] += elapsed_ms
        record["num_calls"] += 1
        record["num_examples"] += num_examples
        record["backends"].add(backend)
        if iteration is not None:
            if record["start_iter"] is None:
                record["start_iter"] = iteration
            record["start_iter"] = min(record["start_iter"], iteration)
            record["end_iter"] = max(record["end_iter"] or iteration, iteration)

    def add_wall_time(self, phase, elapsed_ms, iteration, num_examples=0):
        if self.enabled:
            self._add(phase, elapsed_ms, iteration, num_examples, "wall")

    @contextmanager
    def phase(
        self,
        phase,
        iteration=None,
        num_examples=0,
        use_cuda_events=False,
        synchronize_cuda=False,
        active=True,
    ):
        if not self.enabled or not active:
            yield
            return
        if use_cuda_events and self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self.pending_cuda.append(
                    (phase, start, end, iteration, num_examples)
                )
            return

        if synchronize_cuda and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            if synchronize_cuda and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            self._add(phase, elapsed_ms, iteration, num_examples, "wall")

    def resolve_cuda(self):
        if not self.pending_cuda:
            return
        torch.cuda.synchronize(self.device)
        for phase, start, end, iteration, num_examples in self.pending_cuda:
            self._add(
                phase,
                start.elapsed_time(end),
                iteration,
                num_examples,
                "cuda_event",
            )
        self.pending_cuda.clear()

    def write(self):
        if not self.enabled:
            return
        self.resolve_cuda()
        step_total = self.records["training_step_total"]["total_ms"]
        rows = []
        for phase in TIMING_PHASES:
            record = self.records[phase]
            calls = record["num_calls"]
            start_iter = record["start_iter"]
            end_iter = record["end_iter"]
            if phase in TRAINING_PHASES and start_iter is None:
                start_iter = self.measure_start
                end_iter = self.measure_end
            fraction = ""
            if phase in TRAINING_PHASES and step_total > 0:
                fraction = f'{record["total_ms"] / step_total:.8f}'
            rows.append(
                {
                    "start_iter": "" if start_iter is None else start_iter,
                    "end_iter": "" if end_iter is None else end_iter,
                    "phase": phase,
                    "backend": "+".join(sorted(record["backends"])),
                    "num_calls": calls,
                    "num_examples": record["num_examples"],
                    "total_ms": f'{record["total_ms"]:.6f}',
                    "mean_ms": (
                        f'{record["total_ms"] / calls:.6f}' if calls else ""
                    ),
                    "fraction_of_step": fraction,
                }
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=TIMING_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a distribution-label classifier.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def resolve_device(name):
    return resolve_device_base(name, prefer_mps=True)


def validate_benchmark_run(config, device, start_iter):
    benchmark = config.get("benchmark", {})
    if not benchmark.get("enabled", False):
        return
    if device.type == "mps":
        raise ValueError("benchmark mode supports CPU and CUDA, not MPS")
    warmup_iters = benchmark.get("warmup_iters", 50)
    measure_iters = benchmark.get("measure_iters", 200)
    remaining_iters = int(config["training"]["max_iters"]) - start_iter
    if warmup_iters + measure_iters > remaining_iters:
        raise ValueError(
            "benchmark warm-up and measurement window exceed remaining iterations"
        )


def validate_config(config):
    model = config["model"]
    data = config["data"]
    training = config["training"]
    evaluation = config["evaluation"]

    if model.get("type") not in MODEL_TYPES:
        raise ValueError(f"unknown model.type {model.get('type')!r}")
    if data.get("type") not in DATA_TYPES:
        raise ValueError(f"unknown data.type {data.get('type')!r}")
    if model["vocab_size"] != data["num_states"]:
        raise ValueError("model.vocab_size must equal data.num_states")
    if model["sequence_length"] != data["sequence_length"]:
        raise ValueError("model.sequence_length must equal data.sequence_length")
    label_scheme = data.get("label_scheme", "binary")
    if label_scheme not in ("binary", "identity"):
        raise ValueError("data.label_scheme must be 'binary' or 'identity'")
    expected_classes = 2 if label_scheme == "binary" else data["num_distributions"]
    if model.get("num_classes", expected_classes) != expected_classes:
        raise ValueError(
            "model.num_classes must be 2 for binary labels or "
            "data.num_distributions for identity labels"
        )
    if training["max_iters"] < 1:
        raise ValueError("training.max_iters must be at least 1")
    if training["checkpoint_interval"] < 1:
        raise ValueError("training.checkpoint_interval must be at least 1")
    if evaluation["interval"] < 1:
        raise ValueError("evaluation.interval must be at least 1")
    if evaluation["seqs_per_distribution"] < 1:
        raise ValueError("evaluation.seqs_per_distribution must be at least 1")
    if training.get("log_interval", 1) < 1:
        raise ValueError("training.log_interval must be at least 1")
    amp_dtype = training.get("amp_dtype")
    if amp_dtype not in (None, "bf16", "fp16"):
        raise ValueError("training.amp_dtype must be one of null, 'bf16', or 'fp16'")
    benchmark = config.get("benchmark", {})
    if not isinstance(benchmark, dict):
        raise ValueError("benchmark must be a mapping")
    if not isinstance(benchmark.get("enabled", False), bool):
        raise ValueError("benchmark.enabled must be a boolean")
    if benchmark.get("enabled", False):
        warmup_iters = benchmark.get("warmup_iters", 50)
        measure_iters = benchmark.get("measure_iters", 200)
        if type(warmup_iters) is not int or warmup_iters < 0:
            raise ValueError("benchmark.warmup_iters must be a nonnegative integer")
        if type(measure_iters) is not int or measure_iters < 1:
            raise ValueError("benchmark.measure_iters must be a positive integer")
        if training.get("compile", False) and warmup_iters < 1:
            raise ValueError(
                "compiled benchmark runs require at least one warm-up iteration"
            )


def build_data_generator(config, device, generator, checkpoint=None):
    data = dict(config["data"])
    data.pop("type")
    data.pop("label_scheme", None)
    data.pop("sequence_length")
    data.pop("batch_size")
    if checkpoint is not None:
        data["distributions"] = checkpoint["distributions"].to(device)
        data["distribution_labels"] = checkpoint["distribution_labels"].to(device)
    return DATA_TYPES[config["data"]["type"]](
        **data,
        device=device,
        generator=generator,
    )


def build_model(config):
    model = dict(config["model"])
    model_type = model.pop("type")
    return MODEL_TYPES[model_type](**model)


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
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "iteration": iteration,
            "config": config,
            "run_dir": str(run_dir),
            "rng_state": rng_state(train_generator),
            "presentation_counts": presentation_counts.cpu(),
            "distributions": data_generator.distributions.cpu(),
            "distribution_labels": data_generator.distribution_labels.cpu(),
            "eval_batch": {key: value.cpu() for key, value in eval_batch.items()},
        },
        path,
    )


def make_eval_batch(data_generator, config, eval_generator):
    num_distributions = min(
        config["evaluation"].get(
            "num_distributions",
            config["data"]["num_distributions"],
        ),
        config["data"]["num_distributions"],
    )
    seqs_per_distribution = config["evaluation"]["seqs_per_distribution"]
    ids = torch.arange(num_distributions, device=data_generator.device)
    ids = ids.repeat_interleave(seqs_per_distribution)

    train_generator = data_generator.generator
    data_generator.generator = eval_generator
    try:
        tokens, binary_labels = data_generator.sample_from_distribution_ids(
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
    return {"tokens": tokens, "labels": labels, "distribution_ids": ids}


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
    timing=None,
):
    model.eval()
    tokens = eval_batch["tokens"]
    labels = eval_batch["labels"]
    ids = eval_batch["distribution_ids"]
    timing = timing or PhaseTimingCollector(False, tokens.device, None, 0, 0)
    num_examples = ids.numel()
    with timing.phase(
        "evaluation_forward",
        iteration,
        num_examples,
        use_cuda_events=True,
    ):
        losses = []
        microbatch = config["evaluation"].get("microbatch_size", ids.numel())
        for start in range(0, ids.numel(), microbatch):
            stop = min(start + microbatch, ids.numel())
            with autocast_context(tokens.device, amp_dtype):
                logits = model(tokens[start:stop])["logits"]
                losses.append(
                    F.cross_entropy(logits, labels[start:stop], reduction="none")
                )
        losses = torch.cat(losses)
    with timing.phase(
        "evaluation_aggregation",
        iteration,
        num_examples,
        use_cuda_events=True,
    ):
        num_distributions = int(ids.max().item()) + 1
        loss_sums = torch.zeros(
            num_distributions, device=losses.device, dtype=losses.dtype
        )
        loss_sums.scatter_add_(0, ids, losses)
        loss_counts = torch.bincount(ids, minlength=num_distributions).clamp_min(1)
        mean_losses = (loss_sums / loss_counts.to(losses.dtype)).cpu()
        labels_cpu = data_generator.distribution_labels[:num_distributions].cpu()
        counts_cpu = presentation_counts[:num_distributions].cpu()

    with timing.phase("evaluation_csv_write", iteration, num_examples):
        fieldnames = [
            "iter",
            "distribution_id",
            "label",
            "loss",
            "training_seen_count",
        ]
        rows = [
            {
                "iter": iteration,
                "distribution_id": distribution_id,
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
    benchmark = config.get("benchmark", {})
    benchmark_enabled = benchmark.get("enabled", False)
    start_iter = 0 if checkpoint is None else int(checkpoint["iteration"])
    validate_benchmark_run(config, device, start_iter)
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

    warmup_iters = benchmark.get("warmup_iters", 50)
    measure_iters = benchmark.get("measure_iters", 200)
    measure_start = start_iter + warmup_iters + 1
    measure_end = start_iter + warmup_iters + measure_iters
    timing = PhaseTimingCollector(
        benchmark_enabled,
        device,
        run_dir / "timing.csv",
        measure_start,
        measure_end,
    )

    with timing.phase("data_generator_init", start_iter, synchronize_cuda=True):
        data_generator = build_data_generator(
            config, device, train_generator, checkpoint
        )
    with timing.phase(
        "model_and_optimizer_init", start_iter, synchronize_cuda=True
    ):
        model = build_model(config).to(device)
        optimizer = build_optimizer(model, config, device)
    amp_dtype = resolve_amp_dtype(config, device)
    grad_scaler = make_grad_scaler(config, device)
    presentation_counts = torch.zeros(
        config["data"]["num_distributions"],
        dtype=torch.long,
        device=device,
    )
    if checkpoint is not None:
        with timing.phase("resume_restore", start_iter, synchronize_cuda=True):
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            presentation_counts = checkpoint["presentation_counts"].to(
                device=device,
                dtype=torch.long,
            )
            set_rng_state(checkpoint.get("rng_state"), train_generator)
            eval_batch = move_eval_batch(checkpoint["eval_batch"], device)
    else:
        with timing.phase("eval_batch_init", start_iter, synchronize_cuda=True):
            eval_batch = make_eval_batch(data_generator, config, eval_generator)

    forward_model = model
    if config["training"].get("compile", False):
        forward_model = torch.compile(model)

    train_log = run_dir / "train_log.csv"
    eval_log = run_dir / "eval_by_distribution.csv"
    start_time = time.perf_counter()
    last_report_time = start_time
    last_report_iter = start_iter
    max_iters = int(config["training"]["max_iters"])
    report_interval = config["training"].get(
        "report_interval",
        config["evaluation"]["interval"],
    )
    log_interval = config["training"].get("log_interval", report_interval)
    model.train()
    for iteration in range(start_iter + 1, max_iters + 1):
        batch_size = config["data"]["batch_size"]
        measure_iteration = False
        if not benchmark_enabled:
            tokens, distribution_ids, binary_labels = data_generator.sample(
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
            presentation_counts += torch.bincount(
                distribution_ids,
                minlength=config["data"]["num_distributions"],
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_dtype):
                loss = forward_model(tokens, targets=labels)["loss"]
            grad_scaler.scale(loss).backward()
            if config["training"].get("grad_clip_norm") is not None:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config["training"]["grad_clip_norm"]),
                )
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            measure_iteration = timing.is_training_iteration(iteration)
            compile_first = (
                config["training"].get("compile", False)
                and iteration == start_iter + 1
            )
            if compile_first and device.type == "cuda":
                torch.cuda.synchronize(device)
            compile_start_ns = time.perf_counter_ns() if compile_first else None
            with timing.phase(
                "training_step_total",
                iteration,
                batch_size,
                use_cuda_events=True,
                active=measure_iteration,
            ):
                with timing.phase(
                    "data_sample",
                    iteration,
                    batch_size,
                    use_cuda_events=True,
                    active=measure_iteration,
                ):
                    tokens, distribution_ids, binary_labels = data_generator.sample(
                        batch_size=batch_size,
                        sequence_length=config["data"]["sequence_length"],
                        return_distribution_ids=True,
                        return_labels=True,
                    )
                with timing.phase(
                    "label_and_count_bookkeeping",
                    iteration,
                    batch_size,
                    use_cuda_events=True,
                    active=measure_iteration,
                ):
                    labels = (
                        distribution_ids
                        if config["data"].get("label_scheme", "binary") == "identity"
                        else binary_labels
                    )
                    presentation_counts += torch.bincount(
                        distribution_ids,
                        minlength=config["data"]["num_distributions"],
                    )

                with timing.phase(
                    "zero_grad",
                    iteration,
                    batch_size,
                    use_cuda_events=True,
                    active=measure_iteration,
                ):
                    optimizer.zero_grad(set_to_none=True)
                with timing.phase(
                    "forward_and_loss",
                    iteration,
                    batch_size,
                    use_cuda_events=True,
                    active=measure_iteration,
                ):
                    with autocast_context(device, amp_dtype):
                        loss = forward_model(tokens, targets=labels)["loss"]
                with timing.phase(
                    "backward",
                    iteration,
                    batch_size,
                    use_cuda_events=True,
                    active=measure_iteration,
                ):
                    grad_scaler.scale(loss).backward()
                if config["training"].get("grad_clip_norm") is not None:
                    with timing.phase(
                        "grad_unscale_and_clip",
                        iteration,
                        batch_size,
                        use_cuda_events=True,
                        active=measure_iteration,
                    ):
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            float(config["training"]["grad_clip_norm"]),
                        )
                with timing.phase(
                    "optimizer_step_and_scaler_update",
                    iteration,
                    batch_size,
                    use_cuda_events=True,
                    active=measure_iteration,
                ):
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
            if compile_first:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timing.add_wall_time(
                    "compile_first_step",
                    (time.perf_counter_ns() - compile_start_ns) / 1_000_000,
                    iteration,
                    batch_size,
                )
            if measure_iteration and iteration == measure_end:
                timing.resolve_cuda()

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
        should_checkpoint = (
            iteration % config["training"]["checkpoint_interval"] == 0
            or iteration == max_iters
        )
        with timing.phase(
            "train_logging_and_reporting",
            iteration,
            active=measure_iteration and (should_log_train or should_report),
        ):
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
        if (
            iteration % config["evaluation"]["interval"] == 0
            or iteration == start_iter + 1
        ):
            evaluate(
                forward_model,
                data_generator,
                config,
                eval_batch,
                presentation_counts,
                eval_log,
                iteration,
                amp_dtype,
                timing,
            )
        if should_checkpoint:
            for phase, path in (
                (
                    "checkpoint_numbered",
                    checkpoint_dir / f"checkpoint_{iteration:06d}.pt",
                ),
                ("checkpoint_latest", checkpoint_dir / "latest.pt"),
            ):
                with timing.phase(phase, iteration):
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

    timing.write()
    print(f"run directory: {run_dir}")


if __name__ == "__main__":
    main()
