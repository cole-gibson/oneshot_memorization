import argparse
import copy
import csv
import shlex
import subprocess
from pathlib import Path

import coolname
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit individual Slurm jobs over YAML configs with per-config runtimes."
    )
    parser.add_argument(
        "runtime_manifest",
        type=Path,
        help="CSV with rows of: config_path,time",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root for job outputs. Defaults to run.output_dir from the first config.",
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--trainer-module", default="base.train_distribution_classifier")
    parser.add_argument("--venv", type=Path, default=Path(".venv"))
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--mem", default="4G")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--constraint", default="gpu80")
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


def read_runtime_manifest(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.reader(manifest_file)
        for line_number, row in enumerate(reader, start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 2:
                raise ValueError(
                    f"{path}:{line_number} must contain config_path,time"
                )
            raw_config_path, runtime = [value.strip() for value in row]
            if not raw_config_path:
                raise ValueError(f"{path}:{line_number} has an empty config path")
            if not runtime:
                raise ValueError(f"{path}:{line_number} has an empty runtime")
            config_path = Path(raw_config_path).expanduser()
            if config_path.suffix not in {".yaml", ".yml"}:
                raise ValueError(f"{config_path} must be a YAML config")
            if not config_path.is_file():
                raise FileNotFoundError(f"{config_path} does not exist")
            rows.append((config_path, runtime))
    if not rows:
        raise ValueError(f"no config runtimes found in {path}")
    return rows


def write_sbatch(path, args, config_path, run_dir, log_dir, runtime, job_stem):
    repo_root = Path.cwd().resolve()
    venv = args.venv if args.venv.is_absolute() else repo_root / args.venv
    python_path = venv / "bin" / "python"
    quoted_repo_root = shlex.quote(str(repo_root))
    quoted_config_path = shlex.quote(str(config_path))
    quoted_run_dir = shlex.quote(str(run_dir))
    quoted_python_path = shlex.quote(str(python_path))
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=oneshot-job",
        f"#SBATCH --output={log_dir}/{job_stem}_%j.out",
        f"#SBATCH --error={log_dir}/{job_stem}_%j.err",
        *sbatch_line("partition", args.partition),
        *sbatch_line("account", args.account),
        *sbatch_line("qos", args.qos),
        f"#SBATCH --time={runtime}",
        *sbatch_line("mem", args.mem),
        *sbatch_line("cpus-per-task", args.cpus_per_task),
        *sbatch_line("gpus", args.gpus),
        *sbatch_line("gres", args.gres),
        *sbatch_line("constraint", args.constraint),
        "",
        "set -euo pipefail",
        f"cd {quoted_repo_root}",
        f"CONFIG_PATH={quoted_config_path}",
        f"RUN_DIR={quoted_run_dir}",
        "",
        f"{quoted_python_path} -m {args.trainer_module} --config \"$CONFIG_PATH\"",
        "",
        "mkdir -p \"$RUN_DIR/slurm_logs\"",
        "job_id=${SLURM_JOB_ID:-}",
        "for ext in out err; do",
        f"    source_path=\"{log_dir}/{job_stem}_${{job_id}}.${{ext}}\"",
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
    manifest_source = args.runtime_manifest.resolve()
    config_runtimes = read_runtime_manifest(manifest_source)

    first_config = load_yaml(config_runtimes[0][0])
    experiment_name, description = metadata(first_config, args)
    output_root = args.output_root or Path(first_config["run"]["output_dir"])
    experiment_dir = output_root / f"{slug(experiment_name)}-{coolname.generate_slug(2)}"
    configs_dir = experiment_dir / "configs"
    runs_dir = experiment_dir / "runs"
    sbatch_dir = experiment_dir / "sbatch"
    log_dir = repo_root / "logs" / "slurm" / experiment_dir.name
    configs_dir.mkdir(parents=True, exist_ok=False)
    runs_dir.mkdir(parents=True, exist_ok=True)
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    generated_sbatch_paths = []
    manifest_path = experiment_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file, lineterminator="\n")
        for index, (source_path, runtime) in enumerate(config_runtimes):
            config = copy.deepcopy(load_yaml(source_path))
            config["experiment_name"] = experiment_name
            if description is not None:
                config["experiment_description"] = description
            job_stem = f"{index:04d}_{slug(source_path.stem)}"
            run_dir = runs_dir / job_stem
            config.setdefault("run", {})
            config["run"]["output_dir"] = str(runs_dir)
            config["run"]["run_dir"] = str(run_dir)
            config["run"]["resume_from"] = None
            config_path = configs_dir / f"{job_stem}{source_path.suffix}"
            sbatch_path = sbatch_dir / f"{job_stem}.sbatch"
            write_config(config, config_path)
            write_sbatch(
                sbatch_path,
                args,
                config_path.resolve(),
                run_dir.resolve(),
                log_dir.resolve(),
                runtime,
                job_stem,
            )
            generated_sbatch_paths.append(sbatch_path)
            writer.writerow([config_path.resolve(), run_dir.resolve(), runtime])

    print(f"experiment directory: {experiment_dir}")
    print(f"manifest: {manifest_path}")
    print(f"sbatch directory: {sbatch_dir}")
    if args.dry_run:
        return
    for sbatch_path in generated_sbatch_paths:
        subprocess.run(["sbatch", str(sbatch_path)], check=True)


if __name__ == "__main__":
    main()
