import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    return mo, pd, plt, sns


@app.cell
def _(mo, plt, sns):
    def memorization_thresholds(
        df,
        *,
        gap_column='mean_dirichlet_gap',
        threshold=0.0,
        group_column='distribution_id',
        count_column='training_seen_count',
    ):
        below_threshold = df[df[gap_column] < threshold]
        thresholds = (
            below_threshold.groupby(group_column)[count_column]
            .min()
            .rename('result')
        )
        return (
            df[[group_column]]
            .drop_duplicates()
            .merge(thresholds.reset_index(), on=group_column, how='left')
            .sort_values(group_column)
        )

    def plot_run_outputs(
        eval_df,
        *,
        title='',
        train_log_df=None,
        gap_column='mean_dirichlet_gap',
        smooth_gap=False,
        rolling_window=10,
        threshold=0.0,
        final_iteration=None,
        show_distribution_legend=False,
        train_loss_reference=2.93,
    ):
        df = eval_df.copy()
        plot_y = gap_column

        if smooth_gap:
            plot_y = f'{gap_column}_rolling'
            df[plot_y] = (
                df.groupby('distribution_id')[gap_column]
                .rolling(window=rolling_window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

        figures = []

        # Gap trajectories.
        fig, ax = plt.subplots(figsize=(9, 4.8))
        sns.lineplot(
            data=df,
            x='iteration',
            y=plot_y,
            hue='distribution_id',
            legend=show_distribution_legend,
            ax=ax,
        )
        ax.set_xscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(plot_y)
        ax.set_title('Dirichlet gap over training')
        figures.append(fig)

        thresholds = memorization_thresholds(
            df,
            gap_column=gap_column,
            threshold=threshold,
        )

        # First training count where each distribution crosses the gap threshold.
        fig, ax = plt.subplots(figsize=(9, 4.8))
        threshold_df = thresholds.dropna(subset=['result'])
        ax.plot(threshold_df['distribution_id'], threshold_df['result'])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Appearances')
        ax.set_title('Memorization threshold')
        figures.append(fig)

        if final_iteration is None:
            final_iteration = df['iteration'].max()
        final_df = df[df['iteration'] == final_iteration]

        # Final gap by distribution rank.
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.scatter(final_df['distribution_id'], final_df[gap_column])
        ax.set_xlabel('Rank')
        ax.set_ylabel('Loss (rel. to generalizing)')
        ax.set_title(f'{gap_column} at iteration {final_iteration:g}')
        figures.append(fig)

        if train_log_df is not None:
            # Optional training loss from the same run.
            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.plot(train_log_df['iter'], train_log_df['loss'])
            if train_loss_reference is not None:
                ax.axhline(train_loss_reference)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.set_title('Training loss')
            figures.append(fig)

        output = figures
        if title:
            output = [mo.md(f'### {title}'), *figures]
        return mo.vstack(output)

    return (plot_run_outputs,)


@app.cell
def _(pd, plot_run_outputs):
    _df = pd.read_csv(
        '/home/cg5763/data/output_oneshot_memorization/minimal-qualified-oarfish/balanced_eval.csv'
    )

    plot_run_outputs(
        _df,
        title='Large relative generalization cost',
        # smooth_gap=True,
        # final_iteration=341900,
    )
    return


@app.cell
def _(pd, plot_run_outputs):
    _df = pd.read_csv(
        '/home/cg5763/data/output_oneshot_memorization/minimal-inescapable-wombat/balanced_eval.csv'
    )

    plot_run_outputs(
        _df,
        title='Large absolute generalization cost',
        threshold=-0.1,
    )
    return


if __name__ == "__main__":
    app.run()
