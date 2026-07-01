import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import torch
    import matplotlib.pyplot as plt

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from base.data_generator import DirichletZipfSequenceGenerator
    from base.estimators import DirichletEmpiricalEstimator
    from base.minimal_model import Transformer
    from workbooks.train_model_helpers import (
        make_balanced_eval_tokens,
        make_memorization_rows,
        make_torch_generator,
        mean_loss_by_task,
        model_autoregressive_losses,
        plot_memorization_by_task_rank,
        plot_training_loss,
        plot_unmemorized_distribution_by_rank,
        resolve_device,
    )

    return (
        DirichletEmpiricalEstimator,
        DirichletZipfSequenceGenerator,
        Transformer,
        make_balanced_eval_tokens,
        make_memorization_rows,
        make_torch_generator,
        mean_loss_by_task,
        model_autoregressive_losses,
        plot_memorization_by_task_rank,
        plot_training_loss,
        plot_unmemorized_distribution_by_rank,
        resolve_device,
        torch,
    )


@app.cell
def _():
    num_distributions = 10_000
    num_states = 100
    sequence_length = 32
    batch_size = 256
    max_iters = 10_000
    learning_rate = 1e-3
    seed = 0
    device_name = "auto"
    eval_seed = 12345
    eval_seqs_per_task = 32
    eval_max_tasks = min(num_distributions, 2000)
    eval_microbatch_size = 4096
    eval_interval = 10
    memorization_threshold = 0.0
    return (
        batch_size,
        device_name,
        eval_interval,
        eval_max_tasks,
        eval_microbatch_size,
        eval_seed,
        eval_seqs_per_task,
        learning_rate,
        max_iters,
        memorization_threshold,
        num_distributions,
        num_states,
        seed,
        sequence_length,
    )


@app.cell
def _(
    DirichletEmpiricalEstimator,
    DirichletZipfSequenceGenerator,
    Transformer,
    batch_size,
    device_name,
    eval_interval,
    eval_max_tasks,
    eval_microbatch_size,
    eval_seed,
    eval_seqs_per_task,
    learning_rate,
    make_balanced_eval_tokens,
    make_torch_generator,
    max_iters,
    mean_loss_by_task,
    memorization_threshold,
    model_autoregressive_losses,
    num_distributions,
    num_states,
    resolve_device,
    seed,
    sequence_length,
    torch,
):
    torch.manual_seed(seed)
    device = resolve_device(device_name)
    print(f"Using device: {device}")
    rng = make_torch_generator(device, seed)

    data_generator = DirichletZipfSequenceGenerator(
        num_distributions=num_distributions,
        num_states=num_states,
        alpha=0.1,
        zipf_exponent=1.0,
        device=device,
        generator=rng,
    )
    model = Transformer(
        vocab_size=num_states,
        max_seq_len=sequence_length - 1,
        embed_dim=256,
        num_heads=1,
        num_layers=1,
        mlp_ratio=4,
        mlp_num_layers=2,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    dirichlet_eval_tokens, tracked_task_ids = make_balanced_eval_tokens(
        data_generator=data_generator,
        eval_max_tasks=eval_max_tasks,
        eval_seqs_per_task=eval_seqs_per_task,
        sequence_length=sequence_length,
        eval_seed=eval_seed,
        device=device,
    )
    tracked_task_ids_cpu = tracked_task_ids.cpu()
    tracked_task_counts = torch.bincount(
        tracked_task_ids_cpu,
        minlength=eval_max_tasks,
    )

    _dirichlet_estimator = DirichletEmpiricalEstimator(
        num_states=num_states,
        alpha=0.1,
        device=device,
    )
    dirichlet_eval_losses = _dirichlet_estimator.autoregressive_losses(
        dirichlet_eval_tokens,
    ).cpu()
    tracked_task_dirichlet_loss = mean_loss_by_task(
        losses=dirichlet_eval_losses,
        task_ids=tracked_task_ids_cpu,
        task_counts=tracked_task_counts,
        num_tasks=eval_max_tasks,
    )

    presentation_counts = torch.zeros(num_distributions, dtype=torch.long)
    dirichlet_gap_history = []
    memorization_presentations = torch.full(
        (eval_max_tasks,),
        -1,
        dtype=torch.long,
    )

    losses = []
    model.train()
    for _iteration in range(max_iters):
        tokens, distribution_ids = data_generator.sample(
            batch_size=batch_size,
            sequence_length=sequence_length,
            return_distribution_ids=True,
        )
        presentation_counts += torch.bincount(
            distribution_ids.cpu(),
            minlength=num_distributions,
        )
        input_ids = tokens[:, :-1]
        targets = tokens[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, targets=targets)
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
            eval_model_losses = model_autoregressive_losses(
                model,
                dirichlet_eval_tokens,
                microbatch_size=eval_microbatch_size,
            )
            tracked_task_model_loss = mean_loss_by_task(
                losses=eval_model_losses,
                task_ids=tracked_task_ids_cpu,
                task_counts=tracked_task_counts,
                num_tasks=eval_max_tasks,
            )
            tracked_task_gap = tracked_task_model_loss - tracked_task_dirichlet_loss
            memorized = tracked_task_gap < memorization_threshold
            tracked_presentations = presentation_counts[:eval_max_tasks].clone()
            newly_memorized = (
                (memorization_presentations < 0)
                & memorized
            )
            memorization_presentations[newly_memorized] = (
                tracked_presentations[newly_memorized]
            )
            dirichlet_gap_history.append(
                {
                    "iteration": _iteration + 1,
                    "mean_model_minus_dirichlet": tracked_task_gap.mean().item(),
                    "num_memorized_tasks": int(
                        memorized.sum().item()
                    ),
                    "num_tasks_ever_memorized": int(
                        (memorization_presentations >= 0).sum().item()
                    ),
                    "mean_presentations": tracked_presentations.double().mean().item(),
                    "memorization_threshold": memorization_threshold,
                }
            )
            model.train()
        if should_report:
            print(
                f"Iteration {_iteration + 1}/{max_iters}: "
                f"loss={loss.item():.4f}"
            )
            if dirichlet_gap_history:
                print(
                    "  Dirichlet gap="
                    f"{dirichlet_gap_history[-1]['mean_model_minus_dirichlet']:.4f}; "
                    "memorized tasks="
                    f"{dirichlet_gap_history[-1]['num_memorized_tasks']}"
                    f"/{eval_max_tasks} "
                    f"(threshold={memorization_threshold:.4f})"
                )
    return losses, memorization_presentations, presentation_counts


@app.cell
def _(losses, plot_training_loss):
    _fig = plot_training_loss(losses)
    print(f"final loss: {losses[-1]:.4f}")
    _fig
    return


@app.cell
def _(
    eval_max_tasks,
    make_memorization_rows,
    memorization_presentations,
    presentation_counts,
):
    memorization_rows = make_memorization_rows(
        eval_max_tasks=eval_max_tasks,
        memorization_presentations=memorization_presentations,
        presentation_counts=presentation_counts,
    )
    return


@app.cell
def _(
    eval_max_tasks,
    memorization_presentations,
    plot_memorization_by_task_rank,
):
    _num_memorized = int((memorization_presentations >= 0).sum().item())
    print(f"memorized tasks: {_num_memorized}/{eval_max_tasks}")

    _fig = plot_memorization_by_task_rank(
        memorization_presentations=memorization_presentations,
        eval_max_tasks=eval_max_tasks,
    )
    _fig
    return


@app.cell
def _(
    eval_max_tasks,
    memorization_presentations,
    plot_unmemorized_distribution_by_rank,
):
    _num_unmemorized = int((memorization_presentations < 0).sum().item())
    print(f"unmemorized tasks: {_num_unmemorized}/{eval_max_tasks}")

    _fig = plot_unmemorized_distribution_by_rank(
        memorization_presentations=memorization_presentations,
        eval_max_tasks=eval_max_tasks,
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
