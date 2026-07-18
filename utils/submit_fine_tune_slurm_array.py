import argparse
import csv
import shlex
import subprocess
from pathlib import Path

import coolname
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit one unseen-item fine-tuning array task per training run."
    )
    parser.add_argument("training_output_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--venv", type=Path, default=Path(".venv"))
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--mem", default="4G")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--constraint", default="gpu80")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path):
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def slug(value):
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    return "".join(char for char in text if char.isalnum() or char == "-").strip("-")


def find_training_runs(output_dir):
    run_dirs = {
        checkpoint_dir.parent.resolve()
        for checkpoint_dir in output_dir.rglob("checkpoints")
        if (checkpoint_dir.parent / "config.yaml").is_file()
        and any(checkpoint_dir.glob("checkpoint_*.pt"))
    }
    if not run_dirs:
        raise ValueError(f"no training runs with numbered checkpoints found in {output_dir}")
    return sorted(run_dirs)


def sbatch_line(key, value):
    return [] if value is None else [f"#SBATCH --{key}={value}"]


def write_sbatch(path, args, manifest_path, config_path, log_dir, num_jobs):
    repo_root = Path.cwd().resolve()
    venv = args.venv if args.venv.is_absolute() else repo_root / args.venv
    python_path = venv / "bin" / "python"
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=oneshot-finetune",
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
        f"cd {shlex.quote(str(repo_root))}",
        f"ROW=$(sed -n \"$((SLURM_ARRAY_TASK_ID + 1))p\" {shlex.quote(str(manifest_path))} | tr -d '\\r')",
        "SOURCE_RUN=$(printf '%s' \"$ROW\" | cut -f1)",
        "OUTPUT_DIR=$(printf '%s' \"$ROW\" | cut -f2)",
        "",
        f"{shlex.quote(str(python_path))} -m base.fine_tune_unseen \\",
        f"    --config {shlex.quote(str(config_path))} \\",
        "    --run-dir \"$SOURCE_RUN\" \\",
        "    --output-dir \"$OUTPUT_DIR\"",
        "",
        "mkdir -p \"$OUTPUT_DIR/slurm_logs\"",
        "job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}",
        "task_id=${SLURM_ARRAY_TASK_ID:-0}",
        "for ext in out err; do",
        f"    source_path=\"{log_dir}/${{job_id}}_${{task_id}}.${{ext}}\"",
        "    if [ -f \"$source_path\" ]; then",
        "        mv \"$source_path\" \"$OUTPUT_DIR/slurm_logs/\"",
        "    fi",
        "done",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    training_output_dir = args.training_output_dir.resolve()
    source_config = load_yaml(args.config)
    experiment_name = source_config.get("experiment_name", "unseen_fine_tuning")
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else training_output_dir / "fine_tuning"
    )
    experiment_dir = output_root / f"{slug(experiment_name)}-{coolname.generate_slug(2)}"
    runs_dir = experiment_dir / "runs"
    log_dir = Path.cwd().resolve() / "logs" / "slurm" / experiment_dir.name
    runs_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_training_runs(training_output_dir)
    config_path = experiment_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(source_config, config_file, sort_keys=False)
    manifest_path = experiment_dir / "manifest.tsv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file, delimiter="\t", lineterminator="\n")
        for index, run_dir in enumerate(run_dirs):
            output_dir = runs_dir / f"{index:04d}_{slug(run_dir.name)}"
            writer.writerow([run_dir, output_dir.resolve()])

    sbatch_path = experiment_dir / "submit_array.sbatch"
    write_sbatch(
        sbatch_path,
        args,
        manifest_path.resolve(),
        config_path.resolve(),
        log_dir.resolve(),
        len(run_dirs),
    )
    print(f"experiment directory: {experiment_dir}")
    print(f"manifest: {manifest_path}")
    print(f"sbatch script: {sbatch_path}")
    if not args.dry_run:
        subprocess.run(["sbatch", str(sbatch_path)], check=True)


if __name__ == "__main__":
    main()
