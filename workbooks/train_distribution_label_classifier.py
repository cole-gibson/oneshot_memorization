import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import torch
    import torch.nn.functional as F

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from base.bit_sequences import SummarySequenceClassifierMLP
    from base.data_generator import DirichletZipfBinaryClassificationGenerator
    from base.estimators import BayesOptimalDistributionLabelClassifier
    from workbooks.train_model_helpers import (
        make_memorization_rows,
        make_torch_generator,
        mean_loss_by_task,
        plot_memorization_by_task_rank,
        plot_memorization_fraction_over_time,
        plot_training_loss,
        plot_unmemorized_distribution_by_rank,
        resolve_device,
    )

    return (
        BayesOptimalDistributionLabelClassifier,
        DirichletZipfBinaryClassificationGenerator,
        F,
        SummarySequenceClassifierMLP,
        make_memorization_rows,
        make_torch_generator,
        mean_loss_by_task,
        plot_memorization_by_task_rank,
        plot_memorization_fraction_over_time,
        plot_training_loss,
        plot_unmemorized_distribution_by_rank,
        resolve_device,
        torch,
    )


@app.cell
def _():
    num_distributions = 1_000
    num_states = 100
    sequence_length = 32
    alpha = 0.1
    zipf_exponent = 1.0
    batch_size = 256
    max_iters = 10_000
    learning_rate = 1e-3
    seed = 0
    device_name = "auto"
    eval_seed = 12345
    eval_seqs_per_distribution = 32
    eval_max_distributions = min(num_distributions, 1_000)
    eval_microbatch_size = 4096
    eval_interval = 10
    label_scheme = "binary"
    memorization_fraction = 0.9
    return (
        alpha,
        batch_size,
        device_name,
        eval_interval,
        eval_max_distributions,
        eval_microbatch_size,
        eval_seed,
        eval_seqs_per_distribution,
        label_scheme,
        learning_rate,
        max_iters,
        memorization_fraction,
        num_distributions,
        num_states,
        seed,
        sequence_length,
        zipf_exponent,
    )


@app.cell
def _(F, torch):
    @torch.no_grad()
    def model_classification_losses(model, tokens, targets, microbatch_size):
        losses = []
        for start in range(0, tokens.shape[0], microbatch_size):
            stop = min(tokens.shape[0], start + microbatch_size)
            logits = model(tokens[start:stop])["logits"]
            batch_losses = F.cross_entropy(
                logits,
                targets[start:stop],
                reduction="none",
            )
            losses.append(batch_losses.cpu())
        return torch.cat(losses)

    return (model_classification_losses,)


@app.cell
def _(
    BayesOptimalDistributionLabelClassifier,
    DirichletZipfBinaryClassificationGenerator,
    SummarySequenceClassifierMLP,
    alpha,
    batch_size,
    device_name,
    eval_interval,
    eval_max_distributions,
    eval_microbatch_size,
    eval_seed,
    eval_seqs_per_distribution,
    label_scheme,
    learning_rate,
    make_torch_generator,
    max_iters,
    mean_loss_by_task,
    memorization_fraction,
    model_classification_losses,
    num_distributions,
    num_states,
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

    data_generator = DirichletZipfBinaryClassificationGenerator(
        num_distributions=num_distributions,
        num_states=num_states,
        alpha=alpha,
        zipf_exponent=zipf_exponent,
        device=device,
        generator=rng,
    )
    if label_scheme == "binary":
        num_classes = 2
    elif label_scheme == "identity":
        num_classes = num_distributions
    else:
        raise ValueError("label_scheme must be 'binary' or 'identity'")

    model = SummarySequenceClassifierMLP(
        vocab_size=num_states,
        sequence_length=sequence_length,
        num_classes=num_classes,
        embed_dim=256,
        mlp_ratio=4,
        mlp_num_layers=2,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    eval_generator = make_torch_generator(device, eval_seed)
    tracked_distribution_ids = torch.arange(
        eval_max_distributions,
        device=device,
    ).repeat_interleave(eval_seqs_per_distribution)
    train_generator = data_generator.generator
    data_generator.generator = eval_generator
    try:
        eval_tokens = data_generator.sample_from_distribution_ids(
            tracked_distribution_ids,
            sequence_length=sequence_length,
        )
    finally:
        data_generator.generator = train_generator
    eval_targets = (
        data_generator.distribution_labels[tracked_distribution_ids]
        if label_scheme == "binary"
        else tracked_distribution_ids
    )

    tracked_distribution_ids_cpu = tracked_distribution_ids.cpu()
    tracked_distribution_counts = torch.bincount(
        tracked_distribution_ids_cpu,
        minlength=eval_max_distributions,
    )
    random_loss = torch.log(torch.tensor(float(num_classes))).item()
    distribution_labels = (
        data_generator.distribution_labels
        if label_scheme == "binary"
        else torch.arange(num_distributions, device=device, dtype=torch.long)
    )
    bayes = BayesOptimalDistributionLabelClassifier(
        distributions=data_generator.distributions,
        distribution_labels=distribution_labels,
        distribution_weights=data_generator.distribution_weights,
        num_classes=num_classes,
    )
    bayes_eval_losses = bayes.losses(eval_tokens, eval_targets).cpu()
    tracked_distribution_bayes_loss = mean_loss_by_task(
        losses=bayes_eval_losses,
        task_ids=tracked_distribution_ids_cpu,
        task_counts=tracked_distribution_counts,
        num_tasks=eval_max_distributions,
    )
    print(
        "Bayes optimal mean classification loss:",
        f"{tracked_distribution_bayes_loss.mean().item():.4f}",
        "Bayes optimal max classification loss:",
        f"{tracked_distribution_bayes_loss.max().item():.4f}",
    )
    print("random loss:", f"{random_loss:.4f}")

    presentation_counts = torch.zeros(num_distributions, dtype=torch.long)
    memorization_presentations = torch.full(
        (eval_max_distributions,),
        -1,
        dtype=torch.long,
    )
    progress_history = []
    losses = []

    model.train()
    for _iteration in range(max_iters):
        tokens, distribution_ids = data_generator.sample(
            batch_size=batch_size,
            sequence_length=sequence_length,
            return_distribution_ids=True,
        )
        labels = (
            data_generator.distribution_labels[distribution_ids]
            if label_scheme == "binary"
            else distribution_ids
        )
        presentation_counts += torch.bincount(
            distribution_ids.cpu(),
            minlength=num_distributions,
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
            eval_losses = model_classification_losses(
                model,
                eval_tokens,
                eval_targets,
                microbatch_size=eval_microbatch_size,
            )
            tracked_distribution_model_loss = mean_loss_by_task(
                losses=eval_losses,
                task_ids=tracked_distribution_ids_cpu,
                task_counts=tracked_distribution_counts,
                num_tasks=eval_max_distributions,
            )
            bayes_improvement = random_loss - tracked_distribution_bayes_loss
            model_improvement = random_loss - tracked_distribution_model_loss
            progress_to_bayes = model_improvement / bayes_improvement.clamp_min(1e-12)
            bayes_gap = tracked_distribution_model_loss - tracked_distribution_bayes_loss
            memorized = (bayes_improvement > 0) & (
                progress_to_bayes >= memorization_fraction
            )
            tracked_presentations = presentation_counts[:eval_max_distributions].clone()
            newly_memorized = (memorization_presentations < 0) & memorized
            memorization_presentations[newly_memorized] = (
                tracked_presentations[newly_memorized]
            )
            progress_history.append(
                {
                    "iteration": _iteration + 1,
                    "mean_model_loss": tracked_distribution_model_loss.mean().item(),
                    "mean_model_minus_bayes": bayes_gap.mean().item(),
                    "mean_progress_to_bayes": progress_to_bayes.mean().item(),
                    "num_memorized_distributions": int(memorized.sum().item()),
                    "num_distributions_ever_memorized": int(
                        (memorization_presentations >= 0).sum().item()
                    ),
                    "mean_presentations": tracked_presentations.double().mean().item(),
                    "memorization_fraction": memorization_fraction,
                }
            )
            if memorized.all():
                print(
                    "All tracked distributions memorized "
                    f"at iteration {_iteration + 1}."
                )
                break
            model.train()
        if should_report:
            print(
                f"Iteration {_iteration + 1}/{max_iters}: "
                f"loss={loss.item():.4f}"
            )
            if progress_history:
                print(
                    "  Bayes gap="
                    f"{progress_history[-1]['mean_model_minus_bayes']:.4f}; "
                    "memorized distributions="
                    f"{progress_history[-1]['num_memorized_distributions']}"
                    f"/{eval_max_distributions} "
                    f"(fraction={memorization_fraction:.2f})"
                )
    return (
        losses,
        memorization_presentations,
        presentation_counts,
        progress_history,
    )


@app.cell
def _(losses, plot_training_loss):
    _fig = plot_training_loss(losses)
    print(f"final loss: {losses[-1]:.4f}")
    _fig
    return


@app.cell
def _(
    eval_max_distributions,
    plot_memorization_fraction_over_time,
    progress_history,
):
    _fig = plot_memorization_fraction_over_time(
        history=progress_history,
        total_tasks=eval_max_distributions,
        memorized_key="num_distributions_ever_memorized",
    )
    _fig
    return


@app.cell
def _(
    eval_max_distributions,
    make_memorization_rows,
    memorization_presentations,
    presentation_counts,
):
    memorization_rows = make_memorization_rows(
        eval_max_tasks=eval_max_distributions,
        memorization_presentations=memorization_presentations,
        presentation_counts=presentation_counts,
    )
    return


@app.cell
def _(
    eval_max_distributions,
    memorization_presentations,
    plot_memorization_by_task_rank,
):
    _num_memorized = int((memorization_presentations >= 0).sum().item())
    print(f"memorized distributions: {_num_memorized}/{eval_max_distributions}")

    _fig = plot_memorization_by_task_rank(
        memorization_presentations=memorization_presentations,
        eval_max_tasks=eval_max_distributions,
    )
    _fig
    return


@app.cell
def _(
    eval_max_distributions,
    memorization_presentations,
    plot_unmemorized_distribution_by_rank,
):
    _num_unmemorized = int((memorization_presentations < 0).sum().item())
    print(f"unmemorized distributions: {_num_unmemorized}/{eval_max_distributions}")

    _fig = plot_unmemorized_distribution_by_rank(
        memorization_presentations=memorization_presentations,
        eval_max_tasks=eval_max_distributions,
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
