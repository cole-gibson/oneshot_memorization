import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    return Path, mpl, pd, plt, sns


@app.cell
def _():
    # Set this to either the full fine-tuning experiment directory or a single
    # task output directory containing eval_log.csv.
    fine_tuning_output_dir = "/home/cg5763/data/output_oneshot_memorization/distribution-vector-autoencoder-uptight-caracara/runs/0000_0000-model-mlp-ratio-16/fine_tuning/unseen-fine-tuning-chestnut-condor/runs/0000_0000-0000-model-mlp-ratio-16"
    # fine_tuning_output_dir = Path("/path/to/unseen-fine-tuning-experiment")
    return (fine_tuning_output_dir,)


@app.cell
def _(Path, fine_tuning_output_dir, pd):
    def find_eval_logs(output_dir):
        if not output_dir:
            return []
        root = Path(output_dir).expanduser()
        if not root.is_dir():
            raise ValueError(f"fine-tuning output directory does not exist: {root}")
        return sorted(root.rglob("eval_log.csv"))

    def load_eval_logs(paths):
        frames = []
        required_columns = {
            "checkpoint_iter",
            "unseen_item_id",
            "fine_tune_iter",
            "metric_name",
            "metric_value",
        }
        for path in paths:
            frame = pd.read_csv(
                path,
                dtype={
                    "checkpoint_iter": "int64",
                    "unseen_item_id": "int64",
                    "fine_tune_iter": "int64",
                    "metric_name": "string",
                    "metric_value": "float64",
                },
            )
            missing = required_columns - set(frame.columns)
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            frame = frame[list(required_columns)].copy()
            frame["fine_tune_run"] = str(path.parent)
            frames.append(frame)
        if not frames:
            return pd.DataFrame(
                columns=[*sorted(required_columns), "fine_tune_run"]
            )
        combined = pd.concat(frames, ignore_index=True)
        metric_names = combined["metric_name"].dropna().unique()
        if len(metric_names) != 1:
            raise ValueError(
                "the selected fine-tuning outputs must contain exactly one "
                f"evaluation metric; found {sorted(metric_names.tolist())}"
            )
        return combined

    eval_log_paths = find_eval_logs(fine_tuning_output_dir)
    fine_tune_eval_df = load_eval_logs(eval_log_paths)
    return eval_log_paths, fine_tune_eval_df


@app.cell
def _(fine_tune_eval_df, pd):
    if fine_tune_eval_df.empty:
        learning_curve_df = pd.DataFrame(
            columns=[
                "checkpoint_iter",
                "fine_tune_iter",
                "metric_name",
                "metric_value",
                "metric_std",
                "num_observations",
            ]
        )
    else:
        learning_curve_df = (
            fine_tune_eval_df.groupby(
                ["checkpoint_iter", "fine_tune_iter", "metric_name"],
                as_index=False,
                observed=True,
            )
            .agg(
                metric_value=("metric_value", "mean"),
                metric_std=("metric_value", "std"),
                num_observations=("metric_value", "size"),
            )
            .sort_values(["checkpoint_iter", "fine_tune_iter"])
            .reset_index(drop=True)
        )
    return (learning_curve_df,)


@app.cell
def _(eval_log_paths, fine_tune_eval_df, learning_curve_df):
    if fine_tune_eval_df.empty:
        print(
            "Set fine_tuning_output_dir to either a fine-tuning experiment "
            "directory or a single task output directory."
        )
    else:
        print(
            f"Loaded {len(eval_log_paths)} evaluation logs with "
            f"{len(fine_tune_eval_df):,} observations."
        )
    learning_curve_df
    return


@app.cell
def _(learning_curve_df, mpl, plt, sns):
    if learning_curve_df.empty:
        learning_curve_plot = "The learning curve will appear here."
    else:
        sns.set_theme(style="whitegrid")
        _checkpoint_iters = learning_curve_df["checkpoint_iter"].unique()
        _checkpoint_iters.sort()
        _checkpoint_iters = _checkpoint_iters[8:]
        _minimum = float(_checkpoint_iters.min())
        _maximum = float(_checkpoint_iters.max())
        if _minimum <= 0:
            raise ValueError(
                "checkpoint iterations must be positive for logarithmic coloring"
            )
        if _minimum == _maximum:
            _norm = mpl.colors.LogNorm(
                vmin=_minimum / 10**0.5,
                vmax=_maximum * 10**0.5,
            )
        else:
            _norm = mpl.colors.LogNorm(vmin=_minimum, vmax=_maximum)
        _cmap = mpl.colormaps["viridis"]
        _figure, _axis = plt.subplots(figsize=(8, 5))
        for _checkpoint_iter in _checkpoint_iters:
            _curve = learning_curve_df[
                learning_curve_df["checkpoint_iter"] == _checkpoint_iter
            ]
            _axis.plot(
                _curve["fine_tune_iter"],
                _curve["metric_value"],
                color=_cmap(_norm(_checkpoint_iter)),
                linewidth=2,
            )
        _metric_name = str(learning_curve_df["metric_name"].iloc[0])
        _axis.set(
            xlabel="Fine-tuning iteration",
            ylabel=f"Mean {_metric_name}",
            title="Learning curve on unseen fine-tuning items",
            # xscale='log',
        )
        _colorbar = _figure.colorbar(
            mpl.cm.ScalarMappable(norm=_norm, cmap=_cmap),
            ax=_axis,
            pad=0.02,
        )
        _colorbar.set_label("Checkpoint training iteration (log scale)")
        _figure.tight_layout()
        learning_curve_plot = _figure
    learning_curve_plot
    return


if __name__ == "__main__":
    app.run()
