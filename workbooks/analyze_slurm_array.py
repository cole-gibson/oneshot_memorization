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

    from base.bit_sequences import (
        ProbabilityVectorClassifierMLP,
        SequenceClassifierMLP,
        SummarySequenceClassifierMLP,
    )

    MODEL_TYPES = {
        "bit_sequence_mlp": SequenceClassifierMLP,
        "probability_mlp": ProbabilityVectorClassifierMLP,
        "summary_mlp": SummarySequenceClassifierMLP,
    }
    return MODEL_TYPES, Path, np, pd, plt, sns, torch, yaml


@app.cell
def _(Path):
    array_output_dirs = (
        Path(
            "/home/cg5763/data/output_oneshot_memorization/distribution-vector-label-regression-tidy-hoatzin"
        ),
        Path(
            "/home/cg5763/data/output_oneshot_memorization/distribution-vector-label-regression-neon-quokka"
        ),
    )
    metric_threshold = 0.9
    metric_average_window = 1
    initialization_exclusion_iterations = 0
    exclude_first_evaluation_memorizations = False
    rank_bin_count = 20
    return (
        array_output_dirs,
        exclude_first_evaluation_memorizations,
        initialization_exclusion_iterations,
        metric_average_window,
        metric_threshold,
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
            "hidden_dim",
            "num_heads",
            "num_layers",
            "mlp_ratio",
            "mlp_num_layers",
            "num_hidden_layers",
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
    def find_run_dirs(array_dirs):
        if isinstance(array_dirs, (str, Path)):
            array_dirs = (array_dirs,)
        run_dirs = []
        for array_dir in array_dirs:
            array_dir = Path(array_dir).expanduser()
            for config_path in array_dir.rglob("config.yaml"):
                run_dir = config_path.parent
                if any(
                    (run_dir / filename).exists()
                    for filename in (
                        "eval_by_distribution.csv",
                        "eval_by_sequence.csv",
                    )
                ):
                    run_dirs.append(run_dir)
        return sorted(set(run_dirs))

    def load_eval_csv(path):
        analysis_columns = {
            "iter",
            "loss",
            "accuracy",
            "distribution_id",
            "sequence_id",
            "training_seen_count",
        }
        df = pd.read_csv(
            path,
            usecols=lambda column: column in analysis_columns,
            dtype={
                "iter": "int32",
                "distribution_id": "int32",
                "sequence_id": "int32",
                "training_seen_count": "int32",
                "loss": "float64",
                "accuracy": "float64",
            },
        )
        df = df.rename(
            columns={
                "iter": "iteration",
                "loss": "eval_loss",
                "distribution_id": "task_id",
                "sequence_id": "task_id",
            }
        )
        metric_columns = [
            column for column in ("eval_loss", "accuracy") if column in df.columns
        ]
        if len(metric_columns) != 1:
            raise ValueError(
                f"{path} must contain exactly one of 'loss' or 'accuracy'"
            )
        metric_column = metric_columns[0]
        df["metric_name"] = metric_column
        df["metric_value"] = df[metric_column]
        required = {
            "iteration",
            "task_id",
            "metric_value",
            "training_seen_count",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        df["task_rank"] = df["task_id"] + 1
        # Keep the existing analysis columns while treating sequences and
        # distributions uniformly as tasks.
        df["distribution_id"] = df["task_id"]
        df["distribution_rank"] = df["task_rank"]
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
            eval_path = next(
                path
                for path in (
                    run_dir / "eval_by_distribution.csv",
                    run_dir / "eval_by_sequence.csv",
                )
                if path.exists()
            )
            eval_df = load_eval_csv(eval_path)
            run_id = str(run_dir.resolve())
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
                    "task_id",
                    "task_rank",
                    "distribution_id",
                    "distribution_rank",
                    "metric_name",
                    "metric_value",
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
    def is_memorized(eval_df, threshold):
        return (
            (eval_df["metric_name"] == "accuracy")
            & (eval_df["rolling_metric"] >= threshold)
        ) | (
            (eval_df["metric_name"] == "eval_loss")
            & (eval_df["rolling_metric"] < threshold)
        )

    def first_threshold_crossing(eval_df, threshold):
        if eval_df.empty:
            return pd.DataFrame(
                columns=[
                    "run_id",
                    "distribution_id",
                    "distribution_rank",
                    "min_training_seen_count",
                    "previous_training_seen_count",
                    "previous_metric_value",
                    "memorized_at_first_evaluation",
                ]
            )
        if "training_seen_count" not in eval_df.columns:
            raise ValueError("eval data must include training_seen_count")
        group_columns = ["run_id", "distribution_id"]
        ordered = eval_df.sort_values(group_columns + ["iteration"])
        grouped = ordered.groupby(group_columns, sort=False)
        ordered = ordered.assign(
            previous_training_seen_count=grouped["training_seen_count"].shift(),
            previous_metric_value=grouped["metric_value"].shift(),
            memorized_at_first_evaluation=grouped.cumcount().eq(0),
        )
        first_hits = (
            ordered.loc[
                is_memorized(ordered, threshold),
                group_columns
                + [
                    "training_seen_count",
                    "previous_training_seen_count",
                    "previous_metric_value",
                    "memorized_at_first_evaluation",
                ],
            ]
            .drop_duplicates(group_columns, keep="first")
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

    def final_evaluation_metrics(eval_df, run_df, threshold):
        if eval_df.empty:
            final_eval = eval_df.assign(
                is_memorized=pd.Series(dtype="bool"),
            )
            final_eval = final_eval.merge(
                run_df[["run_id", "num_parameters"]],
                on="run_id",
                how="left",
            )
            summary = run_df.assign(
                final_memorized_fraction=pd.Series(dtype="float64"),
            )
            return final_eval, summary
        final_iteration = (
            eval_df.groupby("run_id")["iteration"]
            .max()
            .rename("final_iteration")
            .reset_index()
        )
        final_eval = eval_df.merge(final_iteration, on="run_id")
        final_eval = final_eval[
            final_eval["iteration"] == final_eval["final_iteration"]
        ].copy()
        final_eval["is_memorized"] = is_memorized(final_eval, threshold)
        final_eval = final_eval.merge(
            run_df[["run_id", "num_parameters"]],
            on="run_id",
            how="left",
        )
        final_fraction = (
            final_eval.groupby("run_id", as_index=False)["is_memorized"]
            .mean()
            .rename(columns={"is_memorized": "final_memorized_fraction"})
        )
        summary = run_df.merge(final_fraction, on="run_id", how="left")
        return final_eval, summary

    return final_evaluation_metrics, first_threshold_crossing, is_memorized


@app.cell
def _(
    array_output_dirs,
    initialization_exclusion_iterations,
    load_array_runs,
    metric_average_window,
):
    eval_df, run_df = load_array_runs(array_output_dirs)
    loaded_eval_rows = len(eval_df)
    eval_df = eval_df[
        eval_df["iteration"] > initialization_exclusion_iterations
    ].copy()
    eval_df = eval_df.sort_values(["run_id", "distribution_id", "iteration"])
    eval_df["rolling_metric"] = eval_df.groupby(
        ["run_id", "distribution_id"]
    )["metric_value"].transform(
        lambda metric: metric.rolling(
            window=metric_average_window, min_periods=1
        ).mean()
    )
    metric_names = sorted(eval_df["metric_name"].unique())
    metric_label = (
        "Accuracy" if metric_names == ["accuracy"]
        else "Loss" if metric_names == ["eval_loss"]
        else "Metric"
    )
    print(
        f"loaded {len(run_df)} runs and {loaded_eval_rows} eval rows; "
        f"kept {len(eval_df)} eval rows after excluding iterations <= "
        f"{initialization_exclusion_iterations}; "
        f"using {metric_average_window}-eval running average for thresholding "
        f"({', '.join(metric_names) or 'no metric'})"
    )
    return eval_df, metric_label, run_df


@app.cell
def _(eval_df, final_evaluation_metrics, metric_threshold, run_df):
    final_eval_df, summary_df = final_evaluation_metrics(
        eval_df,
        run_df,
        metric_threshold,
    )
    return final_eval_df, summary_df


@app.cell
def _(
    eval_df,
    first_threshold_crossing,
    is_memorized,
    metric_threshold,
    run_df,
):
    _zero_appearance_memorizations = eval_df[
        is_memorized(eval_df, metric_threshold)
        & (eval_df["training_seen_count"] == 0)
    ][
        [
            "run_id",
            "task_id",
            "task_rank",
            "iteration",
            "metric_name",
            "metric_value",
            "eval_path",
        ]
    ]
    if not _zero_appearance_memorizations.empty:
        print(
            "WARNING: tasks were memorized with 0 training appearances:\n"
            + _zero_appearance_memorizations.to_string(index=False)
        )
    threshold_df = first_threshold_crossing(eval_df, metric_threshold)
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
def _(eval_df, is_memorized, metric_threshold, pd, plt, run_df, sns):
    _plot_df = eval_df.copy()
    _plot_df["is_memorized"] = is_memorized(_plot_df, metric_threshold)
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
def _(np, pd, plt, rank_bin_count, sns, threshold_df):
    _plot_df = threshold_df.copy()
    _plot_df["distribution_rank"] = pd.to_numeric(_plot_df["distribution_rank"])
    _plot_df = _plot_df[_plot_df["distribution_rank"] > 0]
    if _plot_df.empty:
        _fig, _ax = plt.subplots(figsize=(7, 4))
        _ax.text(
            0.5,
            0.5,
            "No tasks",
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
        _bin_edges = (
            np.array([_log_rank.min() - 0.5, _log_rank.max() + 0.5])
            if _log_rank.min() == _log_rank.max()
            else np.linspace(_log_rank.min(), _log_rank.max(), _bin_count + 1)
        )
        _plot_df["rank_bin"] = pd.cut(
            _log_rank, bins=_bin_edges, labels=False, include_lowest=True
        )
        _plot_df["is_unmemorized"] = _plot_df["min_training_seen_count"].isna()
        _binned_df = (
            _plot_df.dropna(subset=["rank_bin"])
            .groupby(["num_parameters", "rank_bin"], as_index=False)
            .agg(proportion_unmemorized=("is_unmemorized", "mean"))
        )
        _centers = 10 ** ((_bin_edges[:-1] + _bin_edges[1:]) / 2)
        _binned_df["task_rank"] = _binned_df["rank_bin"].map(
            lambda rank_bin: _centers[int(rank_bin)]
        )
        _fig, _ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=_binned_df,
            x="task_rank",
            y="proportion_unmemorized",
            hue="num_parameters",
            marker="o",
            ax=_ax,
        )
        _ax.set_xscale("log")
        _ax.set_ylim(-0.02, 1.02)
        _ax.set_xlabel("Task rank bin center")
        _ax.set_ylabel("Proportion unmemorized")
        _ax.set_title("Proportion of unmemorized tasks by task rank")
    _fig
    return


@app.cell
def _(metric_label, plt, sns, threshold_df):
    _plot_df = threshold_df.dropna(subset=["previous_metric_value"]).copy()
    if _plot_df.empty:
        _fig, _ax = plt.subplots(figsize=(7, 4))
        _ax.text(
            0.5,
            0.5,
            "No memorized distributions with a preceding evaluation",
            ha="center",
            va="center",
            transform=_ax.transAxes,
        )
        _ax.set_axis_off()
    else:
        _grid = sns.displot(
            data=_plot_df,
            x="previous_metric_value",
            col="num_parameters",
            col_wrap=2,
            bins=30,
            height=3.2,
            aspect=1.3,
            facet_kws={"sharey": False},
        )
        _grid.set(
            xlabel=f"{metric_label} immediately before memorization",
            ylabel="Distributions",
        )
        _grid.set_titles("{col_name}")
        _grid.fig.suptitle(
            f"Distribution of {metric_label.lower()} immediately before "
            "memorization",
            y=1.02,
        )
        _fig = _grid.fig
    _fig
    return


@app.cell
def _(metric_label, metric_threshold, pd, plt, sns, threshold_df):
    _plot_df = threshold_df.dropna(subset=["previous_metric_value"]).copy()
    _plot_df["min_training_seen_count"] = pd.to_numeric(
        _plot_df["min_training_seen_count"]
    )
    _plot_df = _plot_df[_plot_df["min_training_seen_count"] == 1]
    if _plot_df.empty:
        _fig, _ax = plt.subplots(figsize=(7, 4))
        _ax.text(
            0.5,
            0.5,
            "No distributions memorized after one appearance\n"
            "with a preceding evaluation",
            ha="center",
            va="center",
            transform=_ax.transAxes,
        )
        _ax.set_axis_off()
    else:
        _grid = sns.displot(
            data=_plot_df,
            x="previous_metric_value",
            col="num_parameters",
            col_wrap=2,
            bins=30,
            height=3.2,
            aspect=1.3,
            facet_kws={"sharey": False},
        )
        _grid.set(
            xlabel=f"{metric_label} immediately before memorization",
            ylabel="Distributions",
        )
        _grid.set_titles("{col_name}")
        _grid.fig.suptitle(
            f"Distribution of {metric_label.lower()} immediately before "
            "memorization after one appearance",
            y=1.02,
        )
        for ax in _grid.axes.flat:
            ax.set_xlim(-1, metric_threshold)
        _fig = _grid.fig
    _fig
    return


@app.cell
def _(exclude_first_evaluation_memorizations, pd, plt, sns, threshold_df):
    _plot_df = threshold_df.dropna(subset=["min_training_seen_count"]).copy()
    if exclude_first_evaluation_memorizations:
        _plot_df = _plot_df[~_plot_df["memorized_at_first_evaluation"]]
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
            col="num_parameters",
            col_wrap=2,
            kind="scatter",
            marker=".",
            linewidth=0,
            edgecolor="none",
            height=3.2,
            aspect=1.3,
        )
        for _setting, _ax in _grid.axes_dict.items():
            _facet_df = _plot_df[_plot_df["num_parameters"].eq(_setting)]
            _ax.errorbar(
                _facet_df["distribution_rank"],
                _facet_df["min_training_seen_count"],
                yerr=[_facet_df["lower_error"], _facet_df["lower_error"] * 0],
                fmt="none",
                ecolor="0.35",
                elinewidth=0.8,
                capsize=2,
                # elinewidth=0.,
                # capsize=0,
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
            "Appearances needed to cross metric threshold",
            y=1.02,
        )
        _fig = _grid.fig
    _fig
    return


@app.cell
def _(
    exclude_first_evaluation_memorizations,
    np,
    pd,
    plt,
    rank_bin_count,
    sns,
    threshold_df,
):
    _plot_df = threshold_df.dropna(subset=["min_training_seen_count"]).copy()
    if exclude_first_evaluation_memorizations:
        _plot_df = _plot_df[~_plot_df["memorized_at_first_evaluation"]]
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
            _plot_df.groupby(["num_parameters", "rank_bin"], as_index=False)
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
            n_colors=_binned_df["num_parameters"].nunique()
        )
        for (_setting, _setting_df), _color in zip(
            _binned_df.groupby("num_parameters"),
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
        _ax.set_title("Binned appearances needed to cross metric threshold")
        _ax.legend(title="P")
        _ax.set_aspect("equal")
    _fig
    return


@app.cell
def _(plt, sns, summary_df):
    _fig, _ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(
        data=summary_df,
        x="num_parameters",
        y="final_memorized_fraction",
        marker="o",
        ax=_ax,
    )
    _ax.set_xscale("log")
    _ax.set_ylim(-0.02, 1.02)
    _ax.set_xlabel("Number of parameters")
    _ax.set_ylabel("Final memorized fraction")
    _ax.set_title("Final memorized fraction by model size")
    _fig
    return


@app.cell
def _(final_eval_df, metric_label, np, plt, sns, threshold_df):
    _memorized_distributions = threshold_df.dropna(
        subset=["min_training_seen_count"]
    )[["run_id", "distribution_id"]]
    _plot_df = final_eval_df.merge(
        _memorized_distributions,
        on=["run_id", "distribution_id"],
        how="inner",
    ).dropna(
        subset=["num_parameters"]
    )
    if _plot_df.empty:
        _fig, _ax = plt.subplots(figsize=(7, 4))
        _ax.text(
            0.5,
            0.5,
            "No distributions reached the memorization threshold",
            ha="center",
            va="center",
            transform=_ax.transAxes,
        )
        _ax.set_axis_off()
    else:
        _grid = sns.displot(
            data=_plot_df,
            x="metric_value",
            col="num_parameters",
            col_wrap=2,
            bins=np.linspace(-1, 1, 21),
            height=3.2,
            aspect=1.3,
            facet_kws={"sharey": False},
        )
        _grid.set(
            xlabel=f"Final {metric_label.lower()}",
            ylabel="Memorized distributions",
            xlim=(-1, 1)
        )
        _grid.set_titles("{col_name} parameters")
        _grid.fig.suptitle(
            f"Final {metric_label.lower()} for distributions memorized "
            "during training",
            y=1.02,
        )
        _fig = _grid.fig
    _fig
    return


@app.cell
def _(final_eval_df, metric_label, np, plt, sns, threshold_df):
    _memorized_distributions = threshold_df[threshold_df["min_training_seen_count"].isin([1])].dropna(
        subset=["min_training_seen_count"]
    )[["run_id", "distribution_id"]]
    _plot_df = final_eval_df.merge(
        _memorized_distributions,
        on=["run_id", "distribution_id"],
        how="inner",
    ).dropna(
        subset=["num_parameters"]
    )
    if _plot_df.empty:
        _fig, _ax = plt.subplots(figsize=(7, 4))
        _ax.text(
            0.5,
            0.5,
            "No distributions reached the memorization threshold",
            ha="center",
            va="center",
            transform=_ax.transAxes,
        )
        _ax.set_axis_off()
    else:
        _grid = sns.displot(
            data=_plot_df,
            x="metric_value",
            col="num_parameters",
            col_wrap=2,
            bins=np.linspace(-1, 1, 21),
            height=3.2,
            aspect=1.3,
            facet_kws={"sharey": False},
        )
        _grid.set(
            xlabel=f"Final {metric_label.lower()}",
            ylabel="Memorized distributions",
            xlim=(-1, 1)
        )
        _grid.set_titles("{col_name} parameters")
        _grid.fig.suptitle(
            f"Final {metric_label.lower()} for distributions memorized after 1 appearance",
            y=1.02,
        )
        _fig = _grid.fig
    _fig
    return


if __name__ == "__main__":
    app.run()
