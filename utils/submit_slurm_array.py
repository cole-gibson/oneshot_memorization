import argparse
import copy
import csv
import subprocess
from pathlib import Path

import coolname
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Submit a Slurm array over YAML configs.")
    parser.add_argument("config_dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path('/home/cg5763/data/output/oneshot_memorization'))
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--trainer-module", default="base.train_distribution_classifier")
    parser.add_argument("--venv", type=Path, default=Path(".venv"))
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--mem", default='4G')
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--gres", default='gpu:1')
    parser.add_argument("--constraint", default='gpu80')
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def slug(value):
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    return "".join(ch for ch in text if ch.isalnum() or ch == "-").strip("-")


def metadata(config, args):
    name = args.experiment_name or config.get("experiment_name")
    description = args.description or config.get("experiment_description")
    if not name:
        raise ValueError("set --experiment-name or experiment_name in the configs")
    return name, description


def write_config(config, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def sbatch_line(key, value):
    return [] if value is None else [f"#SBATCH --{key}={value}"]


def write_sbatch(path, args, manifest_path, log_dir, num_jobs):
    repo_root = Path.cwd().resolve()
    venv = args.venv if args.venv.is_absolute() else repo_root / args.venv
    python_path = venv / "bin" / "python"
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=oneshot-array",
        f"#SBATCH --array=0-{num_jobs - 1}",
        f"#SBATCH --output={log_dir}/%A_%a.out",
        f"#SBATCH --error={log_dir}/%A_%a.err",
        *sbatch_line("partition", args.partition),
        *sbatch_line("account", args.account),
        *sbatch_line("qos", args.qos),
        *sbatch_line("time", args.time),
        *sbatch_line("mem", args.mem),
        *sbatch_line("cpus-per-task", args.cpus_per_task),
        *sbatch_line("gpus", args.gpus),
        *sbatch_line("gres", args.gres),
        *sbatch_line("constraint", args.constraint),
        "",
        "set -euo pipefail",
        f"cd {repo_root}",
        f"ROW=$(sed -n \"$((SLURM_ARRAY_TASK_ID + 1))p\" {manifest_path})",
        "CONFIG_PATH=$(printf '%s' \"$ROW\" | cut -f1)",
        "RUN_DIR=$(printf '%s' \"$ROW\" | cut -f2)",
        "",
        f"{python_path} -m {args.trainer_module} --config \"$CONFIG_PATH\"",
        "",
        "mkdir -p \"$RUN_DIR/slurm_logs\"",
        "job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}",
        "task_id=${SLURM_ARRAY_TASK_ID:-0}",
        "for ext in out err; do",
        f"    source_path=\"{log_dir}/${{job_id}}_${{task_id}}.${{ext}}\"",
        "    if [ -f \"$source_path\" ]; then",
        "        mv \"$source_path\" \"$RUN_DIR/slurm_logs/\"",
        "    fi",
        "done",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    repo_root = Path.cwd().resolve()
    config_paths = sorted(args.config_dir.glob("*.yaml")) + sorted(
        args.config_dir.glob("*.yml")
    )
    if not config_paths:
        raise ValueError(f"no YAML configs found in {args.config_dir}")

    first_config = load_yaml(config_paths[0])
    experiment_name, description = metadata(first_config, args)
    experiment_dir = args.output_root / f"{slug(experiment_name)}-{coolname.generate_slug(2)}"
    configs_dir = experiment_dir / "configs"
    runs_dir = experiment_dir / "runs"
    log_dir = repo_root / "logs" / "slurm" / experiment_dir.name
    configs_dir.mkdir(parents=True, exist_ok=False)
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = experiment_dir / "manifest.tsv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file, delimiter="\t")
        for index, source_path in enumerate(config_paths):
            config = copy.deepcopy(load_yaml(source_path))
            config["experiment_name"] = experiment_name
            if description is not None:
                config["experiment_description"] = description
            run_dir = runs_dir / f"{index:04d}_{slug(source_path.stem)}"
            config.setdefault("run", {})
            config["run"]["output_dir"] = str(runs_dir)
            config["run"]["run_dir"] = str(run_dir)
            config["run"]["resume_from"] = None
            config_path = configs_dir / f"{index:04d}_{source_path.name}"
            write_config(config, config_path)
            writer.writerow([config_path.resolve(), run_dir.resolve()])

    sbatch_path = experiment_dir / "submit_array.sbatch"
    write_sbatch(sbatch_path, args, manifest_path.resolve(), log_dir.resolve(), len(config_paths))
    print(f"experiment directory: {experiment_dir}")
    print(f"manifest: {manifest_path}")
    print(f"sbatch script: {sbatch_path}")
    if args.dry_run:
        return
    subprocess.run(["sbatch", str(sbatch_path)], check=True)


if __name__ == "__main__":
    main()
