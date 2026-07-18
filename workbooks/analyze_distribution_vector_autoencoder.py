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

    from base.bit_sequences import ProbabilityVectorAutoencoderMLP

    MODEL_TYPES = {
        "probability_autoencoder_mlp": ProbabilityVectorAutoencoderMLP,
    }
    return MODEL_TYPES, Path, np, pd, plt, sns, torch, yaml


@app.cell
def _():
    array_output_dirs = [
        "/home/cg5763/data/output_oneshot_memorization/distribution-vector-autoencoder-meticulous-jerboa",
        "/home/cg5763/data/output_oneshot_memorization/distribution-vector-autoencoder-psychedelic-dragonfly",
    ]
    metric_threshold = 0.05
    metric_average_window = 1
    initialization_exclusion_iterations = 0
    exclude_first_evaluation_memorizations = False
    rank_bin_count = 20
    bin_memorized_proportion_threshold = 0.9
    low_memorized_bin_alpha = 0.0
    return (
        array_output_dirs,
        bin_memorized_proportion_threshold,
        exclude_first_evaluation_memorizations,
        initialization_exclusion_iterations,
        low_memorized_bin_alpha,
        metric_average_window,
        metric_threshold,
        rank_bin_count,
    )


@app.cell
def _(np, pd, sns):
    def log_parameter_palette(values):
        parameters = np.sort(pd.to_numeric(pd.Series(values).dropna()).unique())
        if len(parameters) == 0:
            return {}
        if len(parameters) == 1:
            positions = np.array([0.5])
        else:
            log_parameters = np.log10(parameters.astype(float))
            positions = (log_parameters - log_parameters.min()) / (
                log_parameters.max() - log_parameters.min()
            )
        color_map = sns.color_palette("crest", as_cmap=True)
        return {
            parameter: color_map(position)
            for parameter, position in zip(parameters, positions)
        }

    return (log_parameter_palette,)


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
            array_dir = Path(array_dir).expanduser().resolve()
            for config_path in array_dir.rglob("config.yaml"):
                run_dir = config_path.parent
                if (run_dir / "eval_by_distribution.csv").exists():
                    run_dirs.append(run_dir)
        return sorted(set(run_dirs))

    def load_eval_csv(path):
        analysis_columns = {
            "iter",
            "loss",
            "distribution_id",
            "training_seen_count",
        }
        df = pd.read_csv(
            path,
            usecols=lambda column: column in analysis_columns,
            dtype={
                "iter": "int32",
                "distribution_id": "int32",
                "training_seen_count": "int32",
                "loss": "float64",
            },
        )
        df = df.rename(
            columns={
                "iter": "iteration",
                "loss": "eval_loss",
                "distribution_id": "task_id",
            }
        )
        df["metric_name"] = "eval_loss"
        df["metric_value"] = df["eval_loss"]
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
def _(pd):
    def is_memorized(eval_df, threshold):
        return eval_df["rolling_metric"] < threshold

    def first_threshold_crossing(eval_df, threshold):
        if eval_df.empty:
            return pd.DataFrame(
                columns=[
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
        group_columns = ["distribution_id"]
        ordered = eval_df
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
            ["distribution_id", "distribution_rank"]
        ].drop_duplicates()
        return distributions.merge(
            first_hits,
            on="distribution_id",
            how="left",
        )

    def summarize_eval_run(eval_df, threshold, run_id, eval_path):
        if eval_df.empty:
            threshold_df = first_threshold_crossing(eval_df, threshold)
            threshold_df.insert(0, "run_id", run_id)
            final_eval_df = eval_df.assign(
                is_memorized=pd.Series(dtype="bool")
            )
            final_eval_df.insert(0, "run_id", run_id)
            memorized_fraction_df = pd.DataFrame(
                columns=["run_id", "iteration", "memorized_fraction"]
            )
            zero_appearance_df = pd.DataFrame(
                columns=[
                    "run_id",
                    "task_id",
                    "task_rank",
                    "iteration",
                    "metric_name",
                    "metric_value",
                    "eval_path",
                ]
            )
            return (
                threshold_df,
                final_eval_df,
                memorized_fraction_df,
                zero_appearance_df,
            )

        memorized = is_memorized(eval_df, threshold)
        zero_appearance_df = eval_df.loc[
            memorized & (eval_df["training_seen_count"] == 0),
            [
                "task_id",
                "task_rank",
                "iteration",
                "metric_name",
                "metric_value",
            ],
        ].copy()
        zero_appearance_df.insert(0, "run_id", run_id)
        zero_appearance_df["eval_path"] = str(eval_path)
        threshold_df = first_threshold_crossing(eval_df, threshold)
        threshold_df.insert(0, "run_id", run_id)

        final_iteration = eval_df["iteration"].max()
        final_eval_df = eval_df[
            eval_df["iteration"] == final_iteration
        ].copy()
        final_eval_df.insert(0, "run_id", run_id)
        final_eval_df["final_iteration"] = final_iteration
        final_eval_df["is_memorized"] = is_memorized(
            final_eval_df, threshold
        )
        memorized_fraction_df = (
            memorized.groupby(eval_df["iteration"], sort=False)
            .mean()
            .rename("memorized_fraction")
            .reset_index()
        )
        memorized_fraction_df.insert(0, "run_id", run_id)
        return (
            threshold_df,
            final_eval_df,
            memorized_fraction_df,
            zero_appearance_df,
        )

    return (summarize_eval_run,)


@app.cell
def _(
    array_output_dirs,
    count_parameters,
    find_run_dirs,
    flatten_config,
    initialization_exclusion_iterations,
    load_eval_csv,
    load_yaml,
    metric_average_window,
    metric_threshold,
    parameter_setting,
    pd,
    summarize_eval_run,
):
    _threshold_frames = []
    _final_eval_frames = []
    _memorized_fraction_frames = []
    _zero_appearance_frames = []
    _run_rows = []
    _metric_names = set()
    _loaded_eval_rows = 0
    _retained_eval_rows = 0

    for _run_dir in find_run_dirs(array_output_dirs):
        _config = load_yaml(_run_dir / "config.yaml")
        _eval_path = _run_dir / "eval_by_distribution.csv"
        _run_id = str(_run_dir.resolve())
        _eval_df = load_eval_csv(_eval_path)
        _loaded_eval_rows += len(_eval_df)
        _eval_df = _eval_df[
            _eval_df["iteration"] > initialization_exclusion_iterations
        ].copy()
        _retained_eval_rows += len(_eval_df)
        _eval_df = _eval_df.sort_values(["distribution_id", "iteration"])
        _rolling_metric = (
            _eval_df.groupby("distribution_id", sort=False)["metric_value"]
            .rolling(window=metric_average_window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        _eval_df["rolling_metric"] = _rolling_metric
        _metric_names.update(_eval_df["metric_name"].unique())

        (
            _threshold_df,
            _final_eval_df,
            _memorized_fraction_df,
            _zero_appearance_df,
        ) = summarize_eval_run(
            _eval_df, metric_threshold, _run_id, _eval_path
        )
        _threshold_frames.append(_threshold_df)
        _final_eval_frames.append(_final_eval_df)
        _memorized_fraction_frames.append(_memorized_fraction_df)
        _zero_appearance_frames.append(_zero_appearance_df)

        _config_flat = flatten_config(_config)
        _run_rows.append(
            {
                "run_id": _run_id,
                "run_dir": str(_run_dir),
                "parameter_setting": parameter_setting(_config),
                "num_parameters": count_parameters(_run_dir, _config),
                **{
                    f"config.{key}": value
                    for key, value in _config_flat.items()
                },
            }
        )
        print(
            f"processed {_eval_path}: {_loaded_eval_rows:,} total rows read",
            flush=True,
        )

    run_df = (
        pd.DataFrame(_run_rows)
        if _run_rows
        else pd.DataFrame(
            columns=[
                "run_id",
                "run_dir",
                "parameter_setting",
                "num_parameters",
            ]
        )
    )
    threshold_df = (
        pd.concat(_threshold_frames, ignore_index=True)
        if _threshold_frames
        else pd.DataFrame(
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
    )
    final_eval_df = (
        pd.concat(_final_eval_frames, ignore_index=True)
        if _final_eval_frames
        else pd.DataFrame(
            columns=[
                "run_id",
                "distribution_id",
                "metric_value",
                "is_memorized",
            ]
        )
    )
    memorized_fraction_df = (
        pd.concat(_memorized_fraction_frames, ignore_index=True)
        if _memorized_fraction_frames
        else pd.DataFrame(
            columns=["run_id", "iteration", "memorized_fraction"]
        )
    )
    zero_appearance_df = (
        pd.concat(_zero_appearance_frames, ignore_index=True)
        if _zero_appearance_frames
        else pd.DataFrame(
            columns=[
                "run_id",
                "task_id",
                "task_rank",
                "iteration",
                "metric_name",
                "metric_value",
                "eval_path",
            ]
        )
    )

    _run_plot_columns = ["run_id", "parameter_setting", "num_parameters"]
    threshold_df = threshold_df.merge(
        run_df[_run_plot_columns], on="run_id", how="left"
    )
    final_eval_df = final_eval_df.merge(
        run_df[["run_id", "num_parameters"]], on="run_id", how="left"
    )
    memorized_fraction_df = memorized_fraction_df.merge(
        run_df[["run_id", "num_parameters"]], on="run_id", how="left"
    )
    _final_fraction = (
        final_eval_df.groupby("run_id", as_index=False)["is_memorized"]
        .mean()
        .rename(columns={"is_memorized": "final_memorized_fraction"})
    )
    summary_df = run_df.merge(_final_fraction, on="run_id", how="left")

    metric_names = sorted(_metric_names)
    metric_label = (
        "Accuracy" if metric_names == ["accuracy"]
        else "Loss" if metric_names == ["eval_loss"]
        else "Metric"
    )
    print(
        f"processed {len(run_df)} runs and {_loaded_eval_rows:,} eval rows; "
        f"analyzed {_retained_eval_rows:,} rows after excluding iterations <= "
        f"{initialization_exclusion_iterations}; "
        f"using {metric_average_window}-eval running average for thresholding "
        f"({', '.join(metric_names) or 'no metric'})"
    )
    return (
        final_eval_df,
        memorized_fraction_df,
        metric_label,
        summary_df,
        threshold_df,
        zero_appearance_df,
    )


@app.cell
def _(threshold_df, zero_appearance_df):
    if not zero_appearance_df.empty:
        print(
            "WARNING: tasks were memorized with 0 training appearances:\n"
            + zero_appearance_df.to_string(index=False)
        )
    threshold_df.head()
    return


@app.cell
def _(log_parameter_palette, memorized_fraction_df, pd, plt, sns):
    _plot_df = memorized_fraction_df.copy()
    _plot_df = _plot_df.dropna(subset=["num_parameters"])
    _plot_df["num_parameters"] = pd.to_numeric(_plot_df["num_parameters"])
    _parameter_colors = log_parameter_palette(_plot_df["num_parameters"])

    _fig, _ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=_plot_df,
        x="iteration",
        y="memorized_fraction",
        hue="num_parameters",
        hue_order=list(_parameter_colors),
        palette=_parameter_colors,
        ax=_ax,
    )
    _ax.set_xscale("log")
    _ax.set_ylim(-0.02, 1.02)
    _ax.set_xlabel("Iteration")
    _ax.set_ylabel("Memorized fraction")
    _ax.set_title("Memorized fraction over time by parameter count")
    _ax.legend(title="P")
    _fig
    return


@app.cell
def _(log_parameter_palette, np, pd, plt, rank_bin_count, sns, threshold_df):
    _plot_df = threshold_df.copy()
    _plot_df["distribution_rank"] = pd.to_numeric(_plot_df["distribution_rank"])
    _plot_df = _plot_df[_plot_df["distribution_rank"] > 0]
    rank_bin_edges = np.array([], dtype=float)
    rank_bin_summary_df = pd.DataFrame(
        columns=[
            "num_parameters",
            "rank_bin",
            "proportion_unmemorized",
            "task_rank",
        ]
    )
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
        rank_bin_edges = _bin_edges
        rank_bin_summary_df = _binned_df
        _parameter_colors = log_parameter_palette(
            rank_bin_summary_df["num_parameters"]
        )
        _fig, _ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=rank_bin_summary_df,
            x="task_rank",
            y="proportion_unmemorized",
            hue="num_parameters",
            hue_order=list(_parameter_colors),
            palette=_parameter_colors,
            marker="o",
            ax=_ax,
        )
        _ax.set_xscale("log")
        _ax.set_ylim(-0.02, 1.02)
        _ax.set_xlabel("Task rank bin center")
        _ax.set_ylabel("Proportion unmemorized")
        _ax.set_title("Proportion of unmemorized tasks by task rank")
        _ax.legend(title="P")
    _fig
    return rank_bin_edges, rank_bin_summary_df


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
    _plot_df = _plot_df[_plot_df["min_training_seen_count"].lt(50)]
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
            # ax.set_xlim(-1, metric_threshold)
            ax.set_xlim(metric_threshold, None)
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
            _errorbar_df = _facet_df[
                _facet_df["min_training_seen_count"] > 1
            ]
            _ax.errorbar(
                _errorbar_df["distribution_rank"],
                _errorbar_df["min_training_seen_count"],
                yerr=[
                    _errorbar_df["lower_error"],
                    _errorbar_df["lower_error"] * 0,
                ],
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
    bin_memorized_proportion_threshold,
    exclude_first_evaluation_memorizations,
    log_parameter_palette,
    low_memorized_bin_alpha,
    np,
    pd,
    plt,
    rank_bin_edges,
    rank_bin_summary_df,
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

    if _plot_df.empty or len(rank_bin_edges) < 2:
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
        _plot_df["rank_bin"] = pd.cut(
            _log_rank,
            bins=rank_bin_edges,
            labels=False,
            include_lowest=True,
        )
        _plot_df = _plot_df.dropna(subset=["rank_bin"]).copy()
        _plot_df["rank_bin"] = _plot_df["rank_bin"].astype(int)

        _binned_df = (
            _plot_df.groupby(["num_parameters", "rank_bin"], as_index=False)
            .agg(
                mean_training_seen_count=("min_training_seen_count", "mean"),
                std_training_seen_count=("min_training_seen_count", "std"),
            )
        )
        _binned_df = _binned_df.merge(
            rank_bin_summary_df[
                [
                    "num_parameters",
                    "rank_bin",
                    "proportion_unmemorized",
                    "task_rank",
                ]
            ],
            on=["num_parameters", "rank_bin"],
            how="left",
        )
        _binned_df["memorized_proportion"] = (
            1 - _binned_df["proportion_unmemorized"]
        )
        _binned_df["std_training_seen_count"] = _binned_df[
            "std_training_seen_count"
        ].fillna(0)
        _binned_df["bin_center_rank"] = _binned_df["task_rank"]

        _fig, _ax = plt.subplots(figsize=(8, 5))
        _parameter_colors = log_parameter_palette(
            _binned_df["num_parameters"]
        )
        for _setting, _setting_df in _binned_df.groupby("num_parameters"):
            _color = _parameter_colors[_setting]
            _setting_df = _setting_df.sort_values("rank_bin")
            _x = _setting_df["bin_center_rank"].to_numpy()
            _y = _setting_df["mean_training_seen_count"].to_numpy()
            _low_proportion_mask = _setting_df[
                "memorized_proportion"
            ].lt(bin_memorized_proportion_threshold).to_numpy()
            _low_proportion_mask = np.maximum.accumulate(
                _low_proportion_mask
            )
            _ax.plot([], [], linewidth=1.2, label=_setting, color=_color)
            _low_indices = np.flatnonzero(_low_proportion_mask)
            if len(_low_indices) == 0:
                _ax.plot(_x, _y, linewidth=1.2, color=_color)
            else:
                _first_low_index = _low_indices[0]
                _ax.plot(
                    _x[:_first_low_index],
                    _y[:_first_low_index],
                    linewidth=1.2,
                    color=_color,
                )
                _ax.plot(
                    _x[max(0, _first_low_index - 1) :],
                    _y[max(0, _first_low_index - 1) :],
                    linewidth=1.2,
                    color=_color,
                    alpha=low_memorized_bin_alpha,
                )
            for _low_proportion, _alpha in (
                (False, 1.0),
                (True, low_memorized_bin_alpha),
            ):
                _point_df = _setting_df[
                    _low_proportion_mask == _low_proportion
                ]
                _ax.errorbar(
                    _point_df["bin_center_rank"],
                    _point_df["mean_training_seen_count"],
                    yerr=_point_df["std_training_seen_count"],
                    fmt="o",
                    linestyle="none",
                    capsize=2,
                    color=_color,
                    alpha=_alpha,
                )
        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.set_xlabel("Distribution rank bin center")
        _ax.set_ylabel("Mean minimum training appearances")
        _ax.set_title(
            "Binned appearances needed to cross metric threshold\n"
            f"Points below {bin_memorized_proportion_threshold:.0%} memorized "
            "and adjacent line segments are faded"
        )
        _ax.legend(title="P")
        # _ax.set_aspect("equal")
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
    _ax.set_yscale("log")
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
            # bins=np.linspace(-1, 1, 21),
            bins=np.linspace(0, 1, 11),
            height=3.2,
            aspect=1.3,
            facet_kws={"sharey": False},
        )
        _grid.set(
            xlabel=f"Final {metric_label.lower()}",
            ylabel="Memorized distributions",
            # xlim=(-1, 1)
            xlim=(0, 1)
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
            bins=np.linspace(0, 1, 11),
            height=3.2,
            aspect=1.3,
            facet_kws={"sharey": False},
        )
        _grid.set(
            xlabel=f"Final {metric_label.lower()}",
            ylabel="Memorized distributions",
            xlim=(0, 1),
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
