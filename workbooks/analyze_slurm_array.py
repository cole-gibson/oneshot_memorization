import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import torch
    import yaml

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from base.bit_sequences import SummarySequenceClassifierMLP
    from base.minimal_model import Transformer as MinimalTransformer
    from base.model import Transformer as FullTransformer

    MODEL_TYPES = {
        "full": FullTransformer,
        "minimal": MinimalTransformer,
        "summary_mlp": SummarySequenceClassifierMLP,
    }
    return MODEL_TYPES, Path, pd, plt, sns, torch, yaml


@app.cell
def _(Path):
    array_output_dir = Path(
        "/home/cg5763/data/output_oneshot_memorization/test-green-armadillo"
    )
    loss_threshold = 0.1
    return array_output_dir, loss_threshold


@app.cell
def _(MODEL_TYPES, Path, torch, yaml):
    def load_yaml(path):
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        return data

    def flatten_config(value, prefix=""):
        rows = {}
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                rows.update(flatten_config(child, child_prefix))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            rows[prefix] = value
        return rows

    def parameter_setting(config):
        model = config.get("model", {})
        preferred = [
            "type",
            "embed_dim",
            "num_heads",
            "num_layers",
            "mlp_ratio",
            "mlp_num_layers",
        ]
        parts = [
            f"{key}={model[key]}"
            for key in preferred
            if key in model
        ]
        if parts:
            return ", ".join(parts)
        return config.get("experiment_name", "run")

    def count_parameters_from_config(config):
        model_config = dict(config["model"])
        model_type = model_config.pop("type", "full")
        if model_type not in MODEL_TYPES:
            raise ValueError(f"unknown model.type {model_type!r}")
        model = MODEL_TYPES[model_type](**model_config)
        return sum(parameter.numel() for parameter in model.parameters())

    def count_parameters_from_checkpoint(run_dir):
        latest = Path(run_dir) / "checkpoints" / "latest.pt"
        if not latest.exists():
            return None
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        model_state = checkpoint.get("model_state")
        if model_state is None:
            return None
        return sum(value.numel() for value in model_state.values())

    def count_parameters(run_dir, config):
        try:
            return count_parameters_from_config(config)
        except Exception:
            return count_parameters_from_checkpoint(run_dir)

    return count_parameters, flatten_config, load_yaml, parameter_setting


@app.cell
def _(Path, pd):
    def find_run_dirs(array_dir):
        array_dir = Path(array_dir).expanduser()
        run_dirs = []
        for config_path in array_dir.rglob("config.yaml"):
            run_dir = config_path.parent
            if (run_dir / "eval_by_distribution.csv").exists():
                run_dirs.append(run_dir)
        return sorted(set(run_dirs))

    def load_eval_csv(path):
        df = pd.read_csv(path)
        df = df.rename(columns={"iter": "iteration", "loss": "eval_loss"})
        required = {
            "iteration",
            "distribution_id",
            "eval_loss",
            "training_seen_count",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        df["distribution_rank"] = df["distribution_id"] + 1
        return df

    return find_run_dirs, load_eval_csv


@app.cell
def _(
    count_parameters,
    find_run_dirs,
    flatten_config,
    load_eval_csv,
    load_yaml,
    parameter_setting,
    pd,
):
    def load_array_runs(array_dir):
        eval_frames = []
        run_rows = []
        for run_dir in find_run_dirs(array_dir):
            config = load_yaml(run_dir / "config.yaml")
            eval_path = run_dir / "eval_by_distribution.csv"
            eval_df = load_eval_csv(eval_path)
            run_id = run_dir.name
            eval_df["run_id"] = run_id
            eval_df["run_dir"] = str(run_dir)
            eval_df["eval_path"] = str(eval_path)
            eval_frames.append(eval_df)

            config_flat = flatten_config(config)
            run_rows.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "parameter_setting": parameter_setting(config),
                    "num_parameters": count_parameters(run_dir, config),
                    **{f"config.{key}": value for key, value in config_flat.items()},
                }
            )

        if not eval_frames:
            empty_eval = pd.DataFrame(
                columns=[
                    "run_id",
                    "run_dir",
                    "eval_path",
                    "iteration",
                    "distribution_id",
                    "distribution_rank",
                    "eval_loss",
                    "training_seen_count",
                ]
            )
            empty_runs = pd.DataFrame(
                columns=[
                    "run_id",
                    "run_dir",
                    "parameter_setting",
                    "num_parameters",
                ]
            )
            return empty_eval, empty_runs

        return pd.concat(eval_frames, ignore_index=True), pd.DataFrame(run_rows)

    return (load_array_runs,)


@app.cell
def _(pd):
    def first_below_threshold(eval_df, threshold):
        if eval_df.empty:
            return pd.DataFrame(
                columns=[
                    "run_id",
                    "distribution_id",
                    "distribution_rank",
                    "min_training_seen_count",
                ]
            )
        if "training_seen_count" not in eval_df.columns:
            raise ValueError("eval data must include training_seen_count")
        hits = eval_df[eval_df["eval_loss"] < threshold]
        first_hits = (
            hits.groupby(["run_id", "distribution_id"], as_index=False)
            ["training_seen_count"]
            .min()
            .rename(columns={"training_seen_count": "min_training_seen_count"})
        )
        distributions = eval_df[
            ["run_id", "distribution_id", "distribution_rank"]
        ].drop_duplicates()
        return distributions.merge(
            first_hits,
            on=["run_id", "distribution_id"],
            how="left",
        )

    def final_run_metrics(eval_df, run_df, threshold):
        if eval_df.empty:
            return run_df.assign(
                final_loss=pd.Series(dtype="float64"),
                max_memorized_distribution_rank=pd.Series(dtype="float64"),
            )
        final_iteration = (
            eval_df.groupby("run_id")["iteration"]
            .max()
            .rename("final_iteration")
            .reset_index()
        )
        final_eval = eval_df.merge(final_iteration, on="run_id")
        final_eval = final_eval[
            final_eval["iteration"] == final_eval["final_iteration"]
        ]
        final_loss = (
            final_eval.groupby("run_id", as_index=False)["eval_loss"]
            .mean()
            .rename(columns={"eval_loss": "final_loss"})
        )
        memorized = final_eval[final_eval["eval_loss"] < threshold]
        max_rank = (
            memorized.groupby("run_id", as_index=False)["distribution_rank"]
            .max()
            .rename(columns={"distribution_rank": "max_memorized_distribution_rank"})
        )
        summary = run_df.merge(final_loss, on="run_id", how="left")
        summary = summary.merge(max_rank, on="run_id", how="left")
        summary["max_memorized_distribution_rank"] = summary[
            "max_memorized_distribution_rank"
        ].fillna(0)
        return summary

    return final_run_metrics, first_below_threshold


@app.cell
def _(array_output_dir, load_array_runs):
    eval_df, run_df = load_array_runs(array_output_dir)
    print(f"loaded {len(run_df)} runs and {len(eval_df)} eval rows")
    return eval_df, run_df


@app.cell
def _(eval_df, loss_threshold, sns):
    ax = sns.lineplot(data=eval_df[eval_df['distribution_id'] < 50], x='iteration', y='eval_loss', hue='distribution_id', legend=False, alpha=0.5)
    ax.axhline(loss_threshold, color='red', linestyle='--', label=f'Threshold = {loss_threshold}')
    return


@app.cell
def _(eval_df, first_below_threshold, loss_threshold, run_df):
    threshold_df = first_below_threshold(eval_df, loss_threshold)
    threshold_df = threshold_df.merge(
        run_df[
            [
                "run_id",
                "parameter_setting",
                "num_parameters",
            ]
        ],
        on="run_id",
        how="left",
    )
    threshold_df.head()
    return (threshold_df,)


@app.cell
def _(threshold_df):
    threshold_df.dropna(subset=["min_training_seen_count"])['distribution_rank'].max()
    return


@app.cell
def _(plt, sns, threshold_df):
    _fig, _ax = plt.subplots(figsize=(9, 5))
    _plot_df = threshold_df.dropna(subset=["min_training_seen_count"])
    sns.scatterplot(
        data=_plot_df,
        x="distribution_rank",
        y="min_training_seen_count",
        hue="parameter_setting",
        marker="o",
        ax=_ax,
    )
    _ax.set_xscale("log")
    _ax.set_yscale("log")
    _ax.set_xlabel("Distribution rank")
    _ax.set_ylabel("Minimum training appearances")
    _ax.set_title("Appearances needed to cross loss threshold")
    _fig
    return


@app.cell
def _(eval_df, final_run_metrics, loss_threshold, run_df):
    summary_df = final_run_metrics(eval_df, run_df, loss_threshold)
    summary_df.sort_values("num_parameters").head()
    return (summary_df,)


@app.cell
def _(plt, sns, summary_df):
    _fig, _ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=summary_df,
        x="num_parameters",
        y="final_loss",
        hue="parameter_setting",
        ax=_ax,
    )
    _ax.set_xscale("log")
    _ax.set_xlabel("Number of parameters")
    _ax.set_yscale("log")
    _ax.set_ylabel("Final loss")
    _ax.set_title("Final loss by model size")
    _fig
    return


@app.cell
def _(plt, sns, summary_df):
    _fig, _ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=summary_df,
        x="num_parameters",
        y="max_memorized_distribution_rank",
        hue="parameter_setting",
        ax=_ax,
    )
    _ax.set_xscale("log")
    _ax.set_xlabel("Number of parameters")
    _ax.set_ylabel("Maximum memorized distribution rank")
    _ax.set_title("Memorized rank by model size")
    _fig
    return


if __name__ == "__main__":
    app.run()
