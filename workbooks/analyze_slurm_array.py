import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    import yaml

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from base.bit_sequences import SummarySequenceClassifierMLP

    MODEL_TYPES = {
        "summary_mlp": SummarySequenceClassifierMLP,
    }
    return MODEL_TYPES, Path, np, pd, plt, sns, torch, yaml


@app.cell
def _(Path):
    array_output_dir = Path(
        "/home/cg5763/data/output_oneshot_memorization/scaling-garnet-jackalope"
    )
    loss_threshold = 0.1
    loss_average_window = 10
    initialization_exclusion_iterations = 100
    rank_bin_count = 20
    return (
        array_output_dir,
        initialization_exclusion_iterations,
        loss_average_window,
        loss_threshold,
        rank_bin_count,
    )


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
                    "previous_training_seen_count",
                ]
            )
        if "training_seen_count" not in eval_df.columns:
            raise ValueError("eval data must include training_seen_count")
        first_hit_rows = []
        for (run_id, distribution_id), group in eval_df.groupby(
            ["run_id", "distribution_id"]
        ):
            group = group.sort_values("iteration")
            hit_positions = (
                (group["rolling_eval_loss"] < threshold).to_numpy().nonzero()[0]
            )
            if len(hit_positions) == 0:
                continue

            hit_position = hit_positions[0]
            hit_row = group.iloc[hit_position]
            previous_training_seen_count = pd.NA
            if hit_position > 0:
                previous_training_seen_count = group.iloc[hit_position - 1][
                    "training_seen_count"
                ]

            first_hit_rows.append(
                {
                    "run_id": run_id,
                    "distribution_id": distribution_id,
                    "min_training_seen_count": hit_row["training_seen_count"],
                    "previous_training_seen_count": previous_training_seen_count,
                }
            )

        first_hits = pd.DataFrame(
            first_hit_rows,
            columns=[
                "run_id",
                "distribution_id",
                "min_training_seen_count",
                "previous_training_seen_count",
            ],
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
        memorized = final_eval[final_eval["rolling_eval_loss"] < threshold]
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
def _(
    array_output_dir,
    initialization_exclusion_iterations,
    load_array_runs,
    loss_average_window,
):
    eval_df, run_df = load_array_runs(array_output_dir)
    loaded_eval_rows = len(eval_df)
    eval_df = eval_df[
        eval_df["iteration"] > initialization_exclusion_iterations
    ].copy()
    eval_df = eval_df.sort_values(["run_id", "distribution_id", "iteration"])
    eval_df["rolling_eval_loss"] = eval_df.groupby(
        ["run_id", "distribution_id"]
    )["eval_loss"].transform(
        lambda loss: loss.rolling(window=loss_average_window, min_periods=1).mean()
    )
    print(
        f"loaded {len(run_df)} runs and {loaded_eval_rows} eval rows; "
        f"kept {len(eval_df)} eval rows after excluding iterations <= "
        f"{initialization_exclusion_iterations}; "
        f"using {loss_average_window}-eval running average for thresholding"
    )
    return eval_df, run_df


@app.cell
def _(eval_df, sns):
    sns.lineplot(data=eval_df[eval_df['run_id'].eq('0000_0000-model-embed-dim-256') & eval_df['distribution_id'].isin([8000, 8001])], x='iteration', y='rolling_eval_loss', hue='distribution_id')
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
def _(eval_df, loss_threshold, pd, plt, run_df, sns):
    _plot_df = eval_df.copy()
    _plot_df["is_memorized"] = _plot_df["rolling_eval_loss"] < loss_threshold
    _plot_df = (
        _plot_df.groupby(["run_id", "iteration"], as_index=False)["is_memorized"]
        .mean()
        .rename(columns={"is_memorized": "memorized_fraction"})
    )
    _plot_df = _plot_df.merge(
        run_df[["run_id", "num_parameters"]],
        on="run_id",
        how="left",
    )
    _plot_df = _plot_df.dropna(subset=["num_parameters"])
    _plot_df["num_parameters"] = pd.to_numeric(_plot_df["num_parameters"])

    _fig, _ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=_plot_df,
        x="iteration",
        y="memorized_fraction",
        hue="num_parameters",
        palette="viridis",
        ax=_ax,
    )
    _ax.set_xscale("log")
    _ax.set_ylim(-0.02, 1.02)
    _ax.set_xlabel("Iteration")
    _ax.set_ylabel("Memorized fraction")
    _ax.set_title("Memorized fraction over time by parameter count")
    _fig
    return


@app.cell
def _(plt, sns, threshold_df):
    _plot_df = threshold_df.dropna(subset=["min_training_seen_count"]).copy()
    if _plot_df.empty:
        _fig, _ax = plt.subplots(figsize=(7, 4))
        _ax.text(
            0.5,
            0.5,
            "No memorized distributions",
            ha="center",
            va="center",
            transform=_ax.transAxes,
        )
        _ax.set_axis_off()
    else:
        _grid = sns.displot(
            data=_plot_df,
            x="distribution_rank",
            col="parameter_setting",
            col_wrap=2,
            bins=30,
            height=3.2,
            aspect=1.3,
            facet_kws={"sharey": False},
        )
        _grid.set(
            # xscale="log",
            xlabel="Task rank",
            ylabel="Memorized tasks",
        )
        _grid.set_titles("{col_name}")
        _grid.fig.suptitle(
            "Distribution of memorized tasks by task rank",
            y=1.02,
        )
        _fig = _grid.fig
    _fig
    return


@app.cell
def _(pd, plt, sns, threshold_df):
    _plot_df = threshold_df.dropna(subset=["min_training_seen_count"]).copy()
    _plot_df["min_training_seen_count"] = pd.to_numeric(
        _plot_df["min_training_seen_count"]
    )
    _plot_df = _plot_df[_plot_df["min_training_seen_count"] > 0]
    if _plot_df.empty:
        _fig, _ax = plt.subplots(figsize=(7, 4))
        _ax.text(
            0.5,
            0.5,
            "No memorized distributions",
            ha="center",
            va="center",
            transform=_ax.transAxes,
        )
        _ax.set_axis_off()
    else:
        _grid = sns.displot(
            data=_plot_df,
            x="min_training_seen_count",
            col="parameter_setting",
            col_wrap=2,
            bins=30,
            height=3.2,
            aspect=1.3,
            facet_kws={"sharey": False},
        )
        _grid.set(
            # xscale="log",
            xlabel="Training appearances at memorization",
            ylabel="Distributions",
        )
        _grid.set_titles("{col_name}")
        _grid.fig.suptitle(
            "Memorization time distribution by parameter setting",
            y=1.02,
        )
        _fig = _grid.fig
    _fig
    return


@app.cell
def _(pd, plt, sns, threshold_df):
    _plot_df = threshold_df.dropna(subset=["min_training_seen_count"]).copy()
    _previous_counts = _plot_df["previous_training_seen_count"].fillna(1)
    _previous_counts = pd.to_numeric(_previous_counts)
    _plot_df["min_training_seen_count"] = pd.to_numeric(
        _plot_df["min_training_seen_count"]
    )
    _plot_df["lower_error"] = (
        _plot_df["min_training_seen_count"] - _previous_counts
    ).clip(lower=0)

    if _plot_df.empty:
        _fig, _ax = plt.subplots(figsize=(7, 4))
        _ax.text(
            0.5,
            0.5,
            "No memorized distributions",
            ha="center",
            va="center",
            transform=_ax.transAxes,
        )
        _ax.set_axis_off()
    else:
        _grid = sns.relplot(
            data=_plot_df,
            x="distribution_rank",
            y="min_training_seen_count",
            col="parameter_setting",
            col_wrap=2,
            kind="scatter",
            marker=".",
            linewidth=0,
            edgecolor="none",
            height=3.2,
            aspect=1.3,
        )
        for _setting, _ax in _grid.axes_dict.items():
            _facet_df = _plot_df[_plot_df["parameter_setting"].eq(_setting)]
            _ax.errorbar(
                _facet_df["distribution_rank"],
                _facet_df["min_training_seen_count"],
                yerr=[_facet_df["lower_error"], _facet_df["lower_error"] * 0],
                fmt="none",
                ecolor="0.35",
                # elinewidth=0.8,
                # capsize=2,
                elinewidth=0.,
                capsize=0,
                alpha=0.6,
                zorder=1.5,
            )
            _ax.set_xscale("log")
            _ax.set_yscale("log")
        _grid.set(
            xlabel="Distribution rank",
            ylabel="Minimum training appearances",
        )
        _grid.set_titles("{col_name}")
        _grid.fig.suptitle(
            "Appearances needed to cross loss threshold",
            y=1.02,
        )
        _fig = _grid.fig
    _fig
    return


@app.cell
def _(np, pd, plt, rank_bin_count, sns, threshold_df):
    _plot_df = threshold_df.dropna(subset=["min_training_seen_count"]).copy()
    _plot_df["distribution_rank"] = pd.to_numeric(_plot_df["distribution_rank"])
    _plot_df["min_training_seen_count"] = pd.to_numeric(
        _plot_df["min_training_seen_count"]
    )
    _plot_df = _plot_df[
        (_plot_df["distribution_rank"] > 0)
        & (_plot_df["min_training_seen_count"] > 0)
    ]

    if _plot_df.empty:
        _fig, _ax = plt.subplots(figsize=(7, 4))
        _ax.text(
            0.5,
            0.5,
            "No memorized distributions",
            ha="center",
            va="center",
            transform=_ax.transAxes,
        )
        _ax.set_axis_off()
    else:
        _log_rank = np.log10(_plot_df["distribution_rank"])
        _bin_count = min(
            max(1, int(rank_bin_count)),
            _plot_df["distribution_rank"].nunique(),
        )
        if _log_rank.min() == _log_rank.max():
            _bin_edges = np.array([_log_rank.min() - 0.5, _log_rank.max() + 0.5])
            _bin_count = 1
        else:
            _bin_edges = np.linspace(_log_rank.min(), _log_rank.max(), _bin_count + 1)
        _plot_df["rank_bin"] = pd.cut(
            _log_rank,
            bins=_bin_edges,
            labels=False,
            include_lowest=True,
        )
        _plot_df = _plot_df.dropna(subset=["rank_bin"]).copy()
        _plot_df["rank_bin"] = _plot_df["rank_bin"].astype(int)
        _bin_centers = 10 ** ((_bin_edges[:-1] + _bin_edges[1:]) / 2)

        _binned_df = (
            _plot_df.groupby(["parameter_setting", "rank_bin"], as_index=False)
            .agg(
                mean_training_seen_count=("min_training_seen_count", "mean"),
                std_training_seen_count=("min_training_seen_count", "std"),
                memorized_tasks=("min_training_seen_count", "size"),
            )
        )
        _binned_df["std_training_seen_count"] = _binned_df[
            "std_training_seen_count"
        ].fillna(0)
        _binned_df["bin_center_rank"] = _binned_df["rank_bin"].map(
            lambda rank_bin: _bin_centers[rank_bin]
        )

        _fig, _ax = plt.subplots(figsize=(8, 5))
        _palette = sns.color_palette(
            n_colors=_binned_df["parameter_setting"].nunique()
        )
        for (_setting, _setting_df), _color in zip(
            _binned_df.groupby("parameter_setting"),
            _palette,
        ):
            _ax.errorbar(
                _setting_df["bin_center_rank"],
                _setting_df["mean_training_seen_count"],
                yerr=_setting_df["std_training_seen_count"],
                marker="o",
                linewidth=1.2,
                capsize=2,
                label=_setting,
                color=_color,
            )
        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.set_xlabel("Distribution rank bin center")
        _ax.set_ylabel("Mean minimum training appearances")
        _ax.set_title("Binned appearances needed to cross loss threshold")
        _ax.legend(title="Parameter setting")
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
