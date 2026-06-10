import argparse
import csv
import random
import time
from pathlib import Path

import coolname
import numpy as np
import torch
import yaml

from base.data_generator import DirichletZipfSequenceGenerator
from base.model import Transformer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer on Zipf data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base/config.yaml"),
        help="Path to a YAML training config.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint to resume from. Overrides run.resume_from in the config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to use for this run. Overrides seed in the config.",
    )
    parser.add_argument(
        "--run-dir-file",
        type=Path,
        default=None,
        help="Optional path where the resolved run directory will be written.",
    )
    return parser.parse_args()


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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(config):
    model_config = config["model"]
    data_config = config["data"]

    if data_config["num_states"] != model_config["vocab_size"]:
        raise ValueError(
            "data.num_states must match model.vocab_size "
            f"({data_config['num_states']} != {model_config['vocab_size']})"
        )

    input_seq_len = data_config["sequence_length"] - 1
    if input_seq_len < 1:
        raise ValueError("data.sequence_length must be at least 2")

    if input_seq_len > model_config["max_seq_len"]:
        raise ValueError(
            "data.sequence_length - 1 must be <= model.max_seq_len "
            f"({input_seq_len} > {model_config['max_seq_len']})"
        )

    checkpoint_interval = config["training"]["checkpoint_interval"]
    if checkpoint_interval < 1:
        raise ValueError("training.checkpoint_interval must be at least 1")


def slugify(text):
    slug = str(text).strip().lower().replace("_", "-").replace(" ", "-")
    return "".join(char for char in slug if char.isalnum() or char == "-").strip("-")


def make_run_dir(config):
    output_dir = Path(config["run"]["output_dir"])
    experiment_name = slugify(config["experiment_name"])
    for _ in range(100):
        suffix = coolname.generate_slug(2)
        run_dir = output_dir / f"{experiment_name}-{suffix}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True)
            return run_dir
    raise RuntimeError("failed to create a unique run directory")


def make_torch_generator(device, seed):
    if device.type not in ("cpu", "cuda"):
        return None

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def get_rng_state(data_generator):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "data_generator": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if data_generator.generator is not None:
        state["data_generator"] = data_generator.generator.get_state()
    return state


def set_rng_state(state, data_generator):
    if not state:
        return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if data_generator.generator is not None and state.get("data_generator") is not None:
        data_generator.generator.set_state(state["data_generator"])


def save_checkpoint(
    path,
    model,
    optimizer,
    iteration,
    config,
    run_dir,
    data_generator,
    distribution_use_counts,
):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
        "config": config,
        "run_dir": str(run_dir),
        "rng_state": get_rng_state(data_generator),
        "distribution_use_counts": distribution_use_counts.cpu(),
    }
    torch.save(checkpoint, path)


def append_log_row(log_path, row):
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["iter", "loss", "lr", "time_sec"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_optimizer(model, optimizer_config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config.get("weight_decay", 0.0),
        betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    validate_config(config)

    resume_from = args.resume_from or config["run"].get("resume_from")
    resume_from = Path(resume_from) if resume_from else None

    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(config.get("device", "auto"))

    model = Transformer(**config["model"]).to(device)
    optimizer = build_optimizer(model, config["optimizer"])

    data_generator = DirichletZipfSequenceGenerator(
        num_distributions=config["data"]["num_distributions"],
        num_states=config["data"]["num_states"],
        alpha=config["data"]["alpha"],
        zipf_exponent=config["data"]["zipf_exponent"],
        device=device,
        generator=make_torch_generator(device, seed),
    )

    start_iter = 0
    distribution_use_counts = torch.zeros(
        config["data"]["num_distributions"],
        dtype=torch.long,
    )
    if resume_from is None:
        run_dir = make_run_dir(config)
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(config, config_file, sort_keys=False)
    else:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_iter = int(checkpoint["iteration"])
        run_dir = Path(checkpoint.get("run_dir", resume_from.parent))
        set_rng_state(checkpoint.get("rng_state"), data_generator)
        if "distribution_use_counts" in checkpoint:
            distribution_use_counts = checkpoint["distribution_use_counts"].to(
                dtype=torch.long,
                device="cpu",
            )
            if distribution_use_counts.numel() != config["data"]["num_distributions"]:
                raise ValueError(
                    "checkpoint distribution_use_counts length does not match "
                    "data.num_distributions"
                )

    if args.run_dir_file is not None:
        args.run_dir_file.parent.mkdir(parents=True, exist_ok=True)
        args.run_dir_file.write_text(f"{run_dir}\n", encoding="utf-8")

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    model.train()
    train_start = time.perf_counter()
    last_checkpoint_time = train_start
    last_checkpoint_iter = start_iter

    max_iters = int(config["training"]["max_iters"])
    checkpoint_interval = int(config["training"]["checkpoint_interval"])
    grad_clip_norm = config["training"].get("grad_clip_norm")

    for iteration in range(start_iter + 1, max_iters + 1):
        tokens, distribution_ids = data_generator.sample(
            batch_size=config["data"]["batch_size"],
            sequence_length=config["data"]["sequence_length"],
            return_distribution_ids=True,
        )
        distribution_use_counts += torch.bincount(
            distribution_ids.cpu(),
            minlength=config["data"]["num_distributions"],
        )
        input_ids = tokens[:, :-1]
        targets = tokens[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, targets=targets)
        loss = output["loss"]
        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))

        optimizer.step()

        append_log_row(
            log_path,
            {
                "iter": iteration,
                "loss": f"{loss.item():.8f}",
                "lr": optimizer.param_groups[0]["lr"],
                "time_sec": f"{time.perf_counter() - train_start:.4f}",
            },
        )

        if iteration % checkpoint_interval == 0:
            numbered_path = checkpoint_dir / f"checkpoint_{iteration:06d}.pt"
            latest_path = checkpoint_dir / "latest.pt"
            save_checkpoint(
                numbered_path,
                model,
                optimizer,
                iteration,
                config,
                run_dir,
                data_generator,
                distribution_use_counts,
            )
            save_checkpoint(
                latest_path,
                model,
                optimizer,
                iteration,
                config,
                run_dir,
                data_generator,
                distribution_use_counts,
            )
            now = time.perf_counter()
            checkpoint_iters = iteration - last_checkpoint_iter
            seconds_per_iter = (now - last_checkpoint_time) / checkpoint_iters
            last_checkpoint_time = now
            last_checkpoint_iter = iteration
            print(
                f"iter {iteration} loss {loss.item():.6f} "
                f"time_per_iter {seconds_per_iter:.4f}s"
            )

    print(f"run directory: {run_dir}")


if __name__ == "__main__":
    main()
