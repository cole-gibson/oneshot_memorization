import csv
import random
from pathlib import Path

import coolname
import numpy as np
import torch
import yaml


def load_config(path):
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def resolve_device(device_name, prefer_mps=False):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if prefer_mps and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


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


def append_csv_rows(path, fieldnames, rows):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def append_csv(path, fieldnames, row):
    append_csv_rows(path, fieldnames, [row])


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


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"
