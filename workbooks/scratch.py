import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    return pd, plt, sns


@app.cell
def _(pd, sns):
    df = pd.read_csv('/home/cg5763/data/output_oneshot_memorization/less-tasks-private-eel/balanced_eval.csv')
    # df = df[df['iteration'] == 341900]
    # df = df[df['iteration'] == 5e4]

    sns.lineplot(data=df, x='iteration', y='mean_dirichlet_gap', hue='distribution_id')
    return (df,)


@app.cell
def _(df):
    df.keys()
    return


@app.cell
def _(df):
    def min_seqs_negative_gap(df):
        def per_group(g):
            neg = g[g['relative_gap'] < 1.0]
            if not neg.empty:
                return neg['training_seen_count'].min()
            return None

        return df.groupby('distribution_id').apply(per_group).rename('result')

    df['relative_gap'] = (df['mean_model_loss'] - df['mean_bayes_loss'])/(df['mean_dirichlet_loss'] - df['mean_bayes_loss'])

    result = min_seqs_negative_gap(df).reset_index()
    return (result,)


@app.cell
def _(plt, result):
    plt.plot(result['distribution_id'], result['result'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rank')
    plt.ylabel('Appearances')
    plt.title('Memorization Threshold')
    plt.show()
    return


@app.cell
def _(df):
    df['relative_gap'] = (df['mean_model_loss'] - df['mean_bayes_loss'])/(df['mean_dirichlet_loss'] - df['mean_bayes_loss'])
    return


@app.cell
def _(df, plt):
    plt.plot(df['distribution_id'], df['relative_gap'])
    plt.xscale('log')
    plt.axhline(1, color='red', linestyle='--')
    # plt.ylim(top=1)
    plt.show()
    return


@app.cell
def _(pd, plt):
    df_log = pd.read_csv('/home/cg5763/data/output_oneshot_memorization/less-tasks-private-eel/train_log.csv')

    plt.plot(df_log['iter'], df_log['loss'])
    plt.axhline(2.93)
    return


if __name__ == "__main__":
    app.run()
