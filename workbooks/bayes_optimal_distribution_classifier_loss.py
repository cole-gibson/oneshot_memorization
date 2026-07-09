import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import pandas as pd
    import torch
    import matplotlib.pyplot as plt

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from base.train_distribution_classifier import (
        build_data_generator,
        load_config,
        make_eval_batch,
        make_torch_generator,
        resolve_device,
        seed_everything,
        validate_config,
    )
    from workbooks.classification_helpers import (
        make_distribution_label_classifier,
        mean_loss_by_task,
        predictor_losses,
    )

    return (
        Path,
        build_data_generator,
        load_config,
        make_distribution_label_classifier,
        make_eval_batch,
        make_torch_generator,
        mean_loss_by_task,
        pd,
        plt,
        predictor_losses,
        resolve_device,
        seed_everything,
        torch,
        validate_config,
    )


@app.cell
def _(Path):
    config_path = Path("../configs/test/0004_model-embed_dim-512.yaml")
    checkpoint_path = None
    device_name = "auto"
    max_eval_distributions = None
    eval_microbatch_size = 1024
    return (
        checkpoint_path,
        config_path,
        device_name,
        eval_microbatch_size,
        max_eval_distributions,
    )


@app.cell
def _(Path, checkpoint_path, config_path, load_config, torch, validate_config):
    if checkpoint_path is None:
        checkpoint = None
        config = load_config(config_path)
        source = f"config: {config_path}"
    else:
        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
        config = checkpoint["config"]
        source = f"checkpoint: {checkpoint_path}"

    validate_config(config)
    print(source)
    return checkpoint, config


@app.cell
def _(
    build_data_generator,
    checkpoint,
    config,
    device_name,
    make_eval_batch,
    make_torch_generator,
    resolve_device,
    seed_everything,
):
    seed = int(config["seed"])
    seed_everything(seed)

    resolved_device_name = (
        config.get("device", "auto") if device_name == "auto" else device_name
    )
    device = resolve_device(resolved_device_name)
    train_generator = make_torch_generator(device, seed)
    eval_generator = make_torch_generator(
        device,
        config["evaluation"].get("seed", seed + 1),
    )

    data_generator = build_data_generator(
        config=config,
        device=device,
        generator=train_generator,
        checkpoint=checkpoint,
    )
    eval_batch = (
        {key: value.to(device) for key, value in checkpoint["eval_batch"].items()}
        if checkpoint is not None
        else make_eval_batch(data_generator, config, eval_generator)
    )

    print(f"Using device: {device}")
    print(f"eval sequences: {eval_batch['tokens'].shape[0]:,}")
    return data_generator, eval_batch


@app.cell
def _(
    config,
    data_generator,
    eval_batch,
    make_distribution_label_classifier,
    max_eval_distributions,
    torch,
):
    distribution_ids = eval_batch["distribution_ids"]
    if max_eval_distributions is not None:
        keep = distribution_ids < int(max_eval_distributions)
        eval_tokens = eval_batch["tokens"][keep]
        eval_labels = eval_batch["labels"][keep]
        eval_distribution_ids = distribution_ids[keep]
    else:
        eval_tokens = eval_batch["tokens"]
        eval_labels = eval_batch["labels"]
        eval_distribution_ids = distribution_ids

    num_eval_distributions = int(eval_distribution_ids.max().item()) + 1
    distribution_counts = torch.bincount(
        eval_distribution_ids.cpu(),
        minlength=num_eval_distributions,
    )

    bayes = make_distribution_label_classifier(
        data_generator,
        config["data"].get("label_scheme", "binary"),
    )
    random_loss = torch.log(torch.tensor(float(bayes.num_classes))).item()

    print(f"evaluated distributions: {num_eval_distributions:,}")
    print(f"random classifier loss: {random_loss:.6f}")
    return (
        bayes,
        distribution_counts,
        eval_distribution_ids,
        eval_labels,
        eval_tokens,
        num_eval_distributions,
        random_loss,
    )


@app.cell
def _(data_generator, plt):
    plt.bar(x=range(len(data_generator.distributions[1])), height=data_generator.distributions[0].cpu().numpy())
    plt.show()
    return


@app.cell
def _(bayes, eval_labels, eval_microbatch_size, eval_tokens, predictor_losses):
    bayes_losses = predictor_losses(
        bayes,
        eval_tokens,
        eval_labels,
        microbatch_size=eval_microbatch_size,
    )
    return (bayes_losses,)


@app.cell
def _(
    bayes_losses,
    distribution_counts,
    eval_distribution_ids,
    mean_loss_by_task,
    num_eval_distributions,
    pd,
    random_loss,
):
    mean_losses = mean_loss_by_task(
        losses=bayes_losses,
        task_ids=eval_distribution_ids.cpu(),
        task_counts=distribution_counts.clamp_min(1),
        num_tasks=num_eval_distributions,
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "mean_bayes_loss",
                "value": bayes_losses.mean().item(),
            },
            {
                "metric": "mean_bayes_improvement_over_random",
                "value": random_loss - bayes_losses.mean().item(),
            },
            {
                "metric": "median_distribution_bayes_loss",
                "value": mean_losses.median().item(),
            },
            {
                "metric": "max_distribution_bayes_loss",
                "value": mean_losses.max().item(),
            },
        ]
    )
    summary
    return (mean_losses,)


@app.cell
def _(
    bayes,
    distribution_counts,
    mean_losses,
    num_eval_distributions,
    pd,
    random_loss,
):
    per_distribution = pd.DataFrame(
        {
            "distribution_id": range(num_eval_distributions),
            "label": bayes.distribution_labels[:num_eval_distributions].cpu().numpy(),
            "num_eval_sequences": distribution_counts.numpy(),
            "bayes_loss": mean_losses.numpy(),
        }
    )
    per_distribution["bayes_improvement_over_random"] = (
        random_loss - per_distribution["bayes_loss"]
    )
    per_distribution
    return (per_distribution,)


@app.cell
def _(per_distribution):
    per_distribution.sort_values("bayes_loss", ascending=False).head(20)
    return


if __name__ == "__main__":
    app.run()
