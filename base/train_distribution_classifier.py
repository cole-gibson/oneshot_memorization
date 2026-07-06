import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from base.bit_sequences import SummarySequenceClassifierMLP
from base.data_generator import DirichletZipfBinaryClassificationGenerator
from base.train import make_run_dir


MODEL_TYPES = {
    "summary_mlp": SummarySequenceClassifierMLP,
}

DATA_TYPES = {
    "dirichlet_zipf_binary": DirichletZipfBinaryClassificationGenerator,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a distribution-label classifier.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_config(path):
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    return config


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_torch_generator(device, seed):
    if device.type not in ("cpu", "cuda"):
        return None
    return torch.Generator(device=device).manual_seed(seed)


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
    if model.get("num_classes", 2) != 2:
        raise ValueError("this trainer currently supports binary labels only")
    if training["max_iters"] < 1:
        raise ValueError("training.max_iters must be at least 1")
    if training["checkpoint_interval"] < 1:
        raise ValueError("training.checkpoint_interval must be at least 1")
    if evaluation["interval"] < 1:
        raise ValueError("evaluation.interval must be at least 1")
    if evaluation["seqs_per_distribution"] < 1:
        raise ValueError("evaluation.seqs_per_distribution must be at least 1")


def build_data_generator(config, device, generator, checkpoint=None):
    data = dict(config["data"])
    data.pop("type")
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


def build_optimizer(model, config):
    opt = config["optimizer"]
    if opt.get("type", "adam") != "adam":
        raise ValueError("optimizer.type must be 'adam'")
    return torch.optim.Adam(
        model.parameters(),
        lr=opt["lr"],
        betas=tuple(opt.get("betas", [0.9, 0.999])),
        weight_decay=opt.get("weight_decay", 0.0),
    )


def rng_state(train_generator, eval_generator):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "train_generator": None,
        "eval_generator": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if train_generator is not None:
        state["train_generator"] = train_generator.get_state()
    if eval_generator is not None:
        state["eval_generator"] = eval_generator.get_state()
    return state


def set_rng_state(state, train_generator, eval_generator):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if train_generator is not None and state.get("train_generator") is not None:
        train_generator.set_state(state["train_generator"])
    if eval_generator is not None and state.get("eval_generator") is not None:
        eval_generator.set_state(state["eval_generator"])


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
            "rng_state": rng_state(train_generator, None),
            "presentation_counts": presentation_counts.cpu(),
            "distributions": data_generator.distributions.cpu(),
            "distribution_labels": data_generator.distribution_labels.cpu(),
            "eval_batch": {key: value.cpu() for key, value in eval_batch.items()},
        },
        path,
    )


def append_csv(path, fieldnames, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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
        tokens, labels = data_generator.sample_from_distribution_ids(
            ids,
            sequence_length=config["data"]["sequence_length"],
            return_labels=True,
        )
    finally:
        data_generator.generator = train_generator
    return {"tokens": tokens, "labels": labels, "distribution_ids": ids}


@torch.no_grad()
def evaluate(model, data_generator, config, eval_batch, log_path, iteration):
    model.eval()
    tokens = eval_batch["tokens"]
    labels = eval_batch["labels"]
    ids = eval_batch["distribution_ids"]
    losses = []
    microbatch = config["evaluation"].get("microbatch_size", ids.numel())
    for start in range(0, ids.numel(), microbatch):
        stop = min(start + microbatch, ids.numel())
        logits = model(tokens[start:stop])["logits"]
        losses.append(F.cross_entropy(logits, labels[start:stop], reduction="none").cpu())
    losses = torch.cat(losses)
    ids_cpu = ids.cpu()
    num_distributions = int(ids_cpu.max().item()) + 1

    for distribution_id in range(num_distributions):
        mask = ids_cpu == distribution_id
        append_csv(
            log_path,
            ["iter", "distribution_id", "label", "loss"],
            {
                "iter": iteration,
                "distribution_id": distribution_id,
                "label": int(data_generator.distribution_labels[distribution_id].item()),
                "loss": f"{losses[mask].mean().item():.8f}",
            },
        )
    model.train()


def move_eval_batch(eval_batch, device):
    return {key: value.to(device) for key, value in eval_batch.items()}


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


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
    optimizer = build_optimizer(model, config)
    presentation_counts = torch.zeros(
        config["data"]["num_distributions"],
        dtype=torch.long,
    )
    start_iter = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_iter = int(checkpoint["iteration"])
        presentation_counts = checkpoint["presentation_counts"].to(dtype=torch.long)
        set_rng_state(checkpoint.get("rng_state"), train_generator, None)
        eval_batch = move_eval_batch(checkpoint["eval_batch"], device)
    else:
        eval_batch = make_eval_batch(data_generator, config, eval_generator)

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
    model.train()
    for iteration in range(start_iter + 1, max_iters + 1):
        tokens, distribution_ids, labels = data_generator.sample(
            batch_size=config["data"]["batch_size"],
            sequence_length=config["data"]["sequence_length"],
            return_distribution_ids=True,
            return_labels=True,
        )
        presentation_counts += torch.bincount(
            distribution_ids.cpu(),
            minlength=config["data"]["num_distributions"],
        )

        optimizer.zero_grad(set_to_none=True)
        loss = model(tokens, targets=labels)["loss"]
        loss.backward()
        if config["training"].get("grad_clip_norm") is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["training"]["grad_clip_norm"]),
            )
        optimizer.step()

        append_csv(
            train_log,
            ["iter", "loss", "lr", "time_sec"],
            {
                "iter": iteration,
                "loss": f"{loss.item():.8f}",
                "lr": optimizer.param_groups[0]["lr"],
                "time_sec": f"{time.perf_counter() - start_time:.4f}",
            },
        )
        if (
            iteration == start_iter + 1
            or iteration % report_interval == 0
            or iteration == max_iters
        ):
            now = time.perf_counter()
            recent_iters = iteration - last_report_iter
            recent_sec_per_iter = (now - last_report_time) / recent_iters
            total_sec_per_iter = (now - start_time) / (iteration - start_iter)
            eta = format_duration((max_iters - iteration) * total_sec_per_iter)
            print(
                f"iter {iteration}/{max_iters} "
                f"loss {loss.item():.6f} "
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
            evaluate(model, data_generator, config, eval_batch, eval_log, iteration)
        if (
            iteration % config["training"]["checkpoint_interval"] == 0
            or iteration == max_iters
        ):
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
            print(f"checkpoint iter {iteration} loss {loss.item():.6f}", flush=True)

    print(f"run directory: {run_dir}")


if __name__ == "__main__":
    main()
