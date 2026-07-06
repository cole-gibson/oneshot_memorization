import argparse
import copy
import itertools
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Generate YAML configs from a sweep.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        required=True,
        help=(
            "Dotted path and comma-separated values. Use + to set paths together, "
            "e.g. data.sequence_length+model.sequence_length=32,64"
        ),
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def parse_sweep_item(item):
    if "=" not in item:
        raise ValueError(f"sweep item must be path=value1,value2: {item!r}")
    raw_paths, raw_values = item.split("=", 1)
    paths = raw_paths.split("+")
    values = [yaml.safe_load(value) for value in raw_values.split(",")]
    return paths, values


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
    args = parse_args()
    reference = load_yaml(args.reference)
    sweeps = [parse_sweep_item(item) for item in args.sets]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    path_groups = [paths for paths, _ in sweeps]
    value_lists = [values for _, values in sweeps]
    for index, values in enumerate(itertools.product(*value_lists)):
        config = copy.deepcopy(reference)
        for paths, value in zip(path_groups, values):
            for path in paths:
                set_dotted(config, path, value)
        if args.experiment_name is not None:
            config["experiment_name"] = args.experiment_name
        if args.description is not None:
            config["experiment_description"] = args.description

        suffix = "__".join(
            f"{'-'.join(path.replace('.', '-') for path in paths)}-{slug(value)}"
            for paths, value in zip(path_groups, values)
        )
        output_path = args.output_dir / f"{index:04d}_{suffix}.yaml"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(output_path)


if __name__ == "__main__":
    main()
