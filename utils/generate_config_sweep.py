import copy
import itertools
from pathlib import Path

import yaml


# Sweep configuration. Paths in the same tuple are assigned the same value.
REFERENCE_CONFIG = Path("sample_configs/distribution_classifier.yaml")
OUTPUT_DIR = Path("configs/sweeps")
SWEEP = [
    (("data.sequence_length", "model.sequence_length"), [32, 64]),
]
EXPERIMENT_NAME = None
DESCRIPTION = None
OVERWRITE = False


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def set_dotted(config, path, value):
    parts = path.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot set {path!r}; {part!r} is not a mapping")
    node[parts[-1]] = value


def slug(value):
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    return "".join(ch for ch in text if ch.isalnum() or ch in ".-").strip("-")


def main():
    reference = load_yaml(REFERENCE_CONFIG)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path_groups = [paths for paths, _ in SWEEP]
    value_lists = [values for _, values in SWEEP]
    for index, values in enumerate(itertools.product(*value_lists)):
        config = copy.deepcopy(reference)
        for paths, value in zip(path_groups, values):
            for path in paths:
                set_dotted(config, path, value)
        if EXPERIMENT_NAME is not None:
            config["experiment_name"] = EXPERIMENT_NAME
        if DESCRIPTION is not None:
            config["experiment_description"] = DESCRIPTION

        suffix = "__".join(
            f"{'-'.join(path.replace('.', '-') for path in paths)}-{slug(value)}"
            for paths, value in zip(path_groups, values)
        )
        output_path = OUTPUT_DIR / f"{index:04d}_{suffix}.yaml"
        if output_path.exists() and not OVERWRITE:
            raise FileExistsError(f"{output_path} exists; set OVERWRITE = True to replace it")
        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(output_path)


if __name__ == "__main__":
    main()
