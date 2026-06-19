import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt

    return pd, plt


@app.cell
def _(pd):
    df = pd.read_csv('/home/cg5763/data/output_oneshot_memorization/less-tasks-urban-chimera/balanced_eval.csv')
    df = df[df['iteration'] == 5e4]
    return (df,)


@app.cell
def _(df):
    df.keys()
    return


@app.cell
def _(df):
    df['relative_gap'] = (df['mean_model_loss'] - df['mean_bayes_loss'])/(df['mean_dirichlet_loss'] - df['mean_bayes_loss'])
    return


@app.cell
def _(df, plt):
    plt.plot(df['distribution_id'], df['relative_gap'])
    plt.xscale('log')
    # plt.ylim(top=1)
    plt.show()
    return


@app.cell
def _(pd, plt):
    df_log = pd.read_csv('/home/cg5763/data/output_oneshot_memorization/less-tasks-urban-chimera/train_log.csv')

    plt.plot(df_log['iter'], df_log['loss'])
    plt.axhline(2.93)

    return


if __name__ == "__main__":
    app.run()
