import copy
import itertools
from pathlib import Path

import yaml


# Sweep configuration. Use a string for one path or a tuple for coupled paths.

###########

# REFERENCE_CONFIG = Path("sample_configs/bit_sequence_classifier.yaml")
# OUTPUT_DIR = Path("configs/bit_sequence")
# SWEEP = [
#     (("training.max_iters"), [1000]),
#     (("training.report_interval"), [100]),
#     (("training.checkpoint_interval"), [1000]),

#     (("data.num_sequences"), [10000]),
#     (("data.batch_size"), [256]),

#     (("model.hidden_dim"), [1024, 2048, 4096, 8192]),

#     (("evaluation.spacing"), ['linear']),
#     (("evaluation.interval"), [16]),
#     (("evaluation.num_sequences"), [10000]),
#     (("evaluation.microbatch_size"), [10000])
# ]
# EXPERIMENT_NAME = 'bit_sequence_benchmark'
# DESCRIPTION = None
# OVERWRITE = True

############

REFERENCE_CONFIG = Path("sample_configs/distribution_vector_classifier.yaml")
OUTPUT_DIR = Path("configs/distribution_vector")
SWEEP = [
    (("training.max_iters"), [1000]),
    (("training.report_interval"), [100]),
    (("training.checkpoint_interval"), [1000]),

    (("data.num_distributions"), [10000]),
    (("data.batch_size"), [256]),

    (("model.embed_dim"), [256, 512, 1024, 2048]),

    (("evaluation.spacing"), ['linear']),
    (("evaluation.interval"), [16]),
    (("evaluation.num_distributions"), [10000]),
    (("evaluation.microbatch_size"), [10000])
]
EXPERIMENT_NAME = 'distribution_vector_benchmark'
DESCRIPTION = None
OVERWRITE = True


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

    path_groups = [
        (paths,) if isinstance(paths, str) else paths
        for paths, _ in SWEEP
    ]
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
            for paths, value, sweep_values in zip(path_groups, values, value_lists)
            if len(sweep_values) > 1
        )
        filename = f"{index:04d}_{suffix}.yaml" if suffix else f"{index:04d}.yaml"
        output_path = OUTPUT_DIR / filename
        if output_path.exists() and not OVERWRITE:
            raise FileExistsError(f"{output_path} exists; set OVERWRITE = True to replace it")
        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(output_path)


if __name__ == "__main__":
    main()
