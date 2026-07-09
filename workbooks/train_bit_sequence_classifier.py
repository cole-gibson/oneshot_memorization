import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import torch

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from base.bit_sequences import SequenceClassifierMLP, ZipfBitSequenceGenerator
    from workbooks.classification_helpers import (
        classification_losses,
        make_memorization_rows,
        make_torch_generator,
        plot_memorization_by_task_rank,
        plot_memorization_fraction_over_time,
        plot_training_loss,
        plot_unmemorized_distribution_by_rank,
        resolve_device,
    )

    return (
        SequenceClassifierMLP,
        ZipfBitSequenceGenerator,
        classification_losses,
        make_memorization_rows,
        make_torch_generator,
        plot_memorization_by_task_rank,
        plot_memorization_fraction_over_time,
        plot_training_loss,
        plot_unmemorized_distribution_by_rank,
        resolve_device,
        torch,
    )


@app.cell
def _():
    num_sequences = 20_000
    sequence_length = 100
    zipf_exponent = 1.0
    batch_size = 256
    max_iters = 10_000
    learning_rate = 1e-3
    seed = 0
    device_name = "auto"
    eval_max_sequences = min(num_sequences, 20_000)
    eval_microbatch_size = 50_000
    eval_interval = 1_000
    embed_dim = 128
    memorization_margin = 0.001
    return (
        batch_size,
        device_name,
        embed_dim,
        eval_interval,
        eval_max_sequences,
        eval_microbatch_size,
        learning_rate,
        max_iters,
        memorization_margin,
        num_sequences,
        seed,
        sequence_length,
        zipf_exponent,
    )


@app.cell
def _(
    SequenceClassifierMLP,
    ZipfBitSequenceGenerator,
    batch_size,
    classification_losses,
    device_name,
    embed_dim,
    eval_interval,
    eval_max_sequences,
    eval_microbatch_size,
    learning_rate,
    make_torch_generator,
    max_iters,
    memorization_margin,
    num_sequences,
    resolve_device,
    seed,
    sequence_length,
    torch,
    zipf_exponent,
):
    torch.manual_seed(seed)
    device = resolve_device(device_name)
    print(f"Using device: {device}")
    rng = make_torch_generator(device, seed)

    data_generator = ZipfBitSequenceGenerator(
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        zipf_exponent=zipf_exponent,
        device=device,
        generator=rng,
    )
    num_classes = 2

    model = SequenceClassifierMLP(
        sequence_length=sequence_length,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_hidden_layers=2,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    tracked_sequence_ids = torch.arange(eval_max_sequences, device=device)
    eval_tokens = data_generator.sample_from_sequence_ids(tracked_sequence_ids)
    eval_targets = data_generator.labels[tracked_sequence_ids]
    bayes_accuracy = 1.0
    bayes_loss = 0.0

    print("Bayes optimal classification accuracy:", f"{bayes_accuracy:.4f}")
    print("Bayes optimal loss:", f"{bayes_loss:.4f}")
    print("memorization loss threshold:", f"{memorization_margin:.4f}")

    presentation_counts = torch.zeros(num_sequences, dtype=torch.long)
    memorization_presentations = torch.full(
        (eval_max_sequences,),
        -1,
        dtype=torch.long,
    )
    margin_history = []
    losses = []

    model.train()
    for _iteration in range(max_iters):
        tokens, sequence_ids = data_generator.sample(
            batch_size=batch_size,
            return_sequence_ids=True,
        )
        labels = data_generator.labels[sequence_ids]
        presentation_counts += torch.bincount(
            sequence_ids.cpu(),
            minlength=num_sequences,
        )

        optimizer.zero_grad(set_to_none=True)
        output = model(tokens, targets=labels)
        loss = output["loss"]
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        report_interval = max(1, max_iters // 10)
        should_report = (_iteration + 1) % report_interval == 0 or _iteration == 0
        should_eval = (
            (_iteration + 1) % eval_interval == 0
            or _iteration == 0
            or _iteration + 1 == max_iters
        )
        if should_eval:
            model.eval()
            eval_losses = classification_losses(
                model,
                eval_tokens,
                eval_targets,
                microbatch_size=eval_microbatch_size,
            )
            memorized = eval_losses <= memorization_margin
            tracked_presentations = presentation_counts[:eval_max_sequences].clone()
            newly_memorized = (memorization_presentations < 0) & memorized
            memorization_presentations[newly_memorized] = (
                tracked_presentations[newly_memorized]
            )
            margin_history.append(
                {
                    "iteration": _iteration + 1,
                    "mean_loss": eval_losses.mean().item(),
                    "mean_model_minus_bayes": eval_losses.mean().item() - bayes_loss,
                    "num_memorized_sequences": int(memorized.sum().item()),
                    "num_sequences_ever_memorized": int(
                        (memorization_presentations >= 0).sum().item()
                    ),
                    "mean_presentations": tracked_presentations.double().mean().item(),
                    "memorization_margin": memorization_margin,
                }
            )
            if memorized.all():
                print(f"All tracked sequences memorized at iteration {_iteration + 1}.")
                break
            model.train()
        if should_report:
            print(
                f"Iteration {_iteration + 1}/{max_iters}: "
                f"loss={loss.item():.4f}"
            )
            if margin_history:
                print(
                    "  Bayes gap="
                    f"{margin_history[-1]['mean_model_minus_bayes']:.4f}; "
                    "memorized sequences="
                    f"{margin_history[-1]['num_memorized_sequences']}"
                    f"/{eval_max_sequences} "
                    f"(loss threshold={memorization_margin:.4f})"
                )
    return (
        losses,
        margin_history,
        memorization_presentations,
        presentation_counts,
    )


@app.cell
def _(losses, plot_training_loss):
    _fig = plot_training_loss(losses)
    print(f"final loss: {losses[-1]:.4f}")
    _fig
    return


@app.cell
def _(
    eval_max_sequences,
    margin_history,
    plot_memorization_fraction_over_time,
):
    _fig = plot_memorization_fraction_over_time(
        history=margin_history,
        total_tasks=eval_max_sequences,
        memorized_key="num_sequences_ever_memorized",
    )
    _fig
    return


@app.cell
def _(
    eval_max_sequences,
    make_memorization_rows,
    memorization_presentations,
    presentation_counts,
):
    memorization_rows = make_memorization_rows(
        eval_max_tasks=eval_max_sequences,
        memorization_presentations=memorization_presentations,
        presentation_counts=presentation_counts,
    )
    return


@app.cell
def _(
    eval_max_sequences,
    memorization_presentations,
    plot_memorization_by_task_rank,
):
    _num_memorized = int((memorization_presentations >= 0).sum().item())
    print(f"memorized sequences: {_num_memorized}/{eval_max_sequences}")

    _fig = plot_memorization_by_task_rank(
        memorization_presentations=memorization_presentations,
        eval_max_tasks=eval_max_sequences,
    )
    _fig
    return


@app.cell
def _(
    eval_max_sequences,
    memorization_presentations,
    plot_unmemorized_distribution_by_rank,
):
    _num_unmemorized = int((memorization_presentations < 0).sum().item())
    print(f"unmemorized sequences: {_num_unmemorized}/{eval_max_sequences}")

    _fig = plot_unmemorized_distribution_by_rank(
        memorization_presentations=memorization_presentations,
        eval_max_tasks=eval_max_sequences,
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
