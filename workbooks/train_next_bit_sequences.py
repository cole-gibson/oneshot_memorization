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

    from base.bit_sequences import (
        BayesOptimalNextBitPredictor,
        ZipfBitSequenceGenerator,
    )
    from base.model import Transformer
    from workbooks.train_model_helpers import (
        make_memorization_rows,
        make_torch_generator,
        plot_memorization_by_task_rank,
        plot_memorization_fraction_over_time,
        plot_training_loss,
        plot_unmemorized_distribution_by_rank,
        resolve_device,
    )

    return (
        BayesOptimalNextBitPredictor,
        Transformer,
        ZipfBitSequenceGenerator,
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
    num_sequences = 10_000
    sequence_length = 32
    zipf_exponent = 1.0
    batch_size = 256
    max_iters = 10_000
    learning_rate = 1e-3
    seed = 0
    device_name = "auto"
    eval_seed = 12345
    eval_max_sequences = min(num_sequences, 2000)
    eval_microbatch_size = 4096
    eval_interval = 10
    memorization_accuracy = 0.9
    return (
        batch_size,
        device_name,
        eval_interval,
        eval_max_sequences,
        eval_microbatch_size,
        eval_seed,
        learning_rate,
        max_iters,
        memorization_accuracy,
        num_sequences,
        seed,
        sequence_length,
        zipf_exponent,
    )


@app.cell
def _(torch):
    @torch.no_grad()
    def model_autoregressive_accuracies(model, tokens, microbatch_size):
        accuracies = []
        for start in range(0, tokens.shape[0], microbatch_size):
            stop = min(tokens.shape[0], start + microbatch_size)
            batch_tokens = tokens[start:stop]
            input_ids = batch_tokens[:, :-1]
            targets = batch_tokens[:, 1:]
            logits = model(input_ids)["logits"]
            predictions = logits.argmax(dim=-1)
            accuracies.append(predictions.eq(targets).float().mean(dim=1).cpu())
        return torch.cat(accuracies)

    return (model_autoregressive_accuracies,)


@app.cell
def _(
    BayesOptimalNextBitPredictor,
    Transformer,
    ZipfBitSequenceGenerator,
    batch_size,
    device_name,
    eval_interval,
    eval_max_sequences,
    eval_microbatch_size,
    eval_seed,
    learning_rate,
    make_torch_generator,
    max_iters,
    memorization_accuracy,
    model_autoregressive_accuracies,
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
    model = Transformer(
        vocab_size=2,
        max_seq_len=sequence_length - 1,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        mlp_ratio=4,
        mlp_num_layers=2,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    tracked_sequence_ids = torch.arange(eval_max_sequences, device=device)
    eval_tokens = data_generator.sample_from_sequence_ids(tracked_sequence_ids)
    tracked_sequence_ids_cpu = tracked_sequence_ids.cpu()

    bayes = BayesOptimalNextBitPredictor.from_generator(data_generator)
    bayes_eval_accuracies = bayes.autoregressive_accuracies(eval_tokens).cpu()
    print(
        "Bayes optimal mean next-bit accuracy:",
        f"{bayes_eval_accuracies.mean().item():.4f}",
    )

    presentation_counts = torch.zeros(num_sequences, dtype=torch.long)
    memorization_presentations = torch.full(
        (eval_max_sequences,),
        -1,
        dtype=torch.long,
    )
    accuracy_history = []
    losses = []

    model.train()
    for _iteration in range(max_iters):
        tokens, sequence_ids = data_generator.sample(
            batch_size=batch_size,
            return_sequence_ids=True,
        )
        presentation_counts += torch.bincount(
            sequence_ids.cpu(),
            minlength=num_sequences,
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
            eval_accuracies = model_autoregressive_accuracies(
                model,
                eval_tokens,
                microbatch_size=eval_microbatch_size,
            )
            bayes_accuracy_gap = eval_accuracies - bayes_eval_accuracies
            memorized = eval_accuracies > memorization_accuracy
            tracked_presentations = presentation_counts[:eval_max_sequences].clone()
            newly_memorized = (memorization_presentations < 0) & memorized
            memorization_presentations[newly_memorized] = (
                tracked_presentations[newly_memorized]
            )
            accuracy_history.append(
                {
                    "iteration": _iteration + 1,
                    "mean_accuracy": eval_accuracies.mean().item(),
                    "mean_model_minus_bayes_accuracy": bayes_accuracy_gap.mean().item(),
                    "num_memorized_sequences": int(memorized.sum().item()),
                    "num_sequences_ever_memorized": int(
                        (memorization_presentations >= 0).sum().item()
                    ),
                    "mean_presentations": tracked_presentations.double().mean().item(),
                    "memorization_accuracy": memorization_accuracy,
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
            if accuracy_history:
                print(
                    "  Bayes accuracy gap="
                    f"{accuracy_history[-1]['mean_model_minus_bayes_accuracy']:.4f}; "
                    "memorized sequences="
                    f"{accuracy_history[-1]['num_memorized_sequences']}"
                    f"/{eval_max_sequences} "
                    f"(threshold={memorization_accuracy:.2f})"
                )
    return (
        accuracy_history,
        eval_max_sequences,
        losses,
        memorization_presentations,
        presentation_counts,
        tracked_sequence_ids_cpu,
    )


@app.cell
def _(losses, plot_training_loss):
    _fig = plot_training_loss(losses)
    print(f"final loss: {losses[-1]:.4f}")
    _fig
    return


@app.cell
def _(accuracy_history, eval_max_sequences, plot_memorization_fraction_over_time):
    _fig = plot_memorization_fraction_over_time(
        history=accuracy_history,
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
    return (memorization_rows,)


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
