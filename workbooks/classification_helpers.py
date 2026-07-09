import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from base.bit_sequences import BayesOptimalSequenceClassifier
from base.estimators import BayesOptimalDistributionLabelClassifier


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(device_name)


def make_torch_generator(device, seed):
    if device.type not in ("cpu", "cuda"):
        return None
    return torch.Generator(device=device).manual_seed(seed)


def num_classes_for_label_scheme(label_scheme, num_tasks):
    if label_scheme == "binary":
        return 2
    if label_scheme == "identity":
        return num_tasks
    raise ValueError("label_scheme must be 'binary' or 'identity'")


def targets_for_sequence_labels(data_generator, sequence_ids, label_scheme):
    if label_scheme == "binary":
        return data_generator.labels[sequence_ids]
    if label_scheme == "identity":
        return sequence_ids
    raise ValueError("label_scheme must be 'binary' or 'identity'")


def targets_for_distribution_labels(data_generator, distribution_ids, label_scheme):
    if label_scheme == "binary":
        return data_generator.distribution_labels[distribution_ids]
    if label_scheme == "identity":
        return distribution_ids
    raise ValueError("label_scheme must be 'binary' or 'identity'")


def distribution_labels_for_scheme(data_generator, label_scheme):
    if label_scheme == "binary":
        return data_generator.distribution_labels
    if label_scheme == "identity":
        return torch.arange(
            data_generator.num_distributions,
            device=data_generator.device,
            dtype=torch.long,
        )
    raise ValueError("label_scheme must be 'binary' or 'identity'")


def make_distribution_label_classifier(data_generator, label_scheme):
    return BayesOptimalDistributionLabelClassifier(
        distributions=data_generator.distributions,
        distribution_labels=distribution_labels_for_scheme(
            data_generator,
            label_scheme,
        ),
        distribution_weights=data_generator.distribution_weights,
        num_classes=num_classes_for_label_scheme(
            label_scheme,
            data_generator.num_distributions,
        ),
    )


def sequence_baseline_accuracy(data_generator, eval_tokens, eval_targets, label_scheme):
    if label_scheme == "identity":
        return 1.0
    if label_scheme == "binary":
        bayes = BayesOptimalSequenceClassifier.from_generator(data_generator)
        predictions = bayes.predict(eval_tokens)
        return predictions.eq(eval_targets).float().mean().item()
    raise ValueError("label_scheme must be 'binary' or 'identity'")


@torch.no_grad()
def classification_losses(model, tokens, targets, microbatch_size):
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


@torch.no_grad()
def predictor_losses(predictor, tokens, targets, microbatch_size):
    losses = []
    for start in range(0, tokens.shape[0], microbatch_size):
        stop = min(tokens.shape[0], start + microbatch_size)
        losses.append(
            predictor.losses(
                tokens[start:stop],
                targets[start:stop],
            ).cpu()
        )
    return torch.cat(losses)


def mean_loss_by_task(losses, task_ids, task_counts, num_tasks):
    task_losses = torch.zeros(num_tasks, dtype=torch.float64)
    task_losses.scatter_add_(0, task_ids, losses.double())
    return task_losses / task_counts


def make_memorization_rows(
    eval_max_tasks,
    memorization_presentations,
    presentation_counts,
):
    return [
        {
            "task_rank": task_id + 1,
            "final_presentations": int(presentation_counts[task_id].item()),
            "presentations_until_memorization": (
                None
                if memorization_presentations[task_id].item() < 0
                else int(memorization_presentations[task_id].item())
            ),
        }
        for task_id in range(eval_max_tasks)
    ]


def plot_training_loss(losses):
    iterations = range(1, len(losses) + 1)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(iterations, losses, color="#4f7cac")
    ax.set_xlabel("step")
    ax.set_ylabel("cross entropy")
    ax.set_title("Training loss")
    ax.set_xscale("log")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_memorization_by_task_rank(memorization_presentations, eval_max_tasks):
    task_ranks = range(1, eval_max_tasks + 1)
    presentations = [
        None if count.item() < 0 else int(count.item())
        for count in memorization_presentations
    ]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(task_ranks, presentations, color="#4f7cac", alpha=0.2, s=5)
    ax.set_xlabel("task rank")
    ax.set_xscale("log")
    ax.set_ylabel("presentations until memorization")
    ax.set_yscale("log")
    ax.set_title("Memorization by task rank")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_memorization_fraction_over_time(
    history,
    total_tasks,
    memorized_key,
    title="Memorization fraction over time",
):
    iterations = [row["iteration"] for row in history]
    fractions = [row[memorized_key] / total_tasks for row in history]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(iterations, fractions, color="#4f7cac")
    ax.set_xlabel("step")
    ax.set_ylabel("memorized fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_unmemorized_distribution_by_rank(
    memorization_presentations,
    eval_max_tasks,
    num_bins=30,
):
    task_ranks = torch.arange(1, eval_max_tasks + 1)
    unmemorized_ranks = task_ranks[memorization_presentations < 0]
    bin_edges = torch.unique(
        torch.round(
            torch.logspace(
                0,
                torch.log10(torch.tensor(float(eval_max_tasks))).item(),
                steps=min(num_bins, eval_max_tasks) + 1,
            )
        ).long()
    )
    if bin_edges[0].item() != 1:
        bin_edges = torch.cat([torch.tensor([1]), bin_edges])
    if bin_edges[-1].item() != eval_max_tasks:
        bin_edges = torch.cat([bin_edges, torch.tensor([eval_max_tasks])])

    counts = []
    labels = []
    for index, (start, next_start) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        stop = next_start if index == len(bin_edges) - 2 else next_start - 1
        counts.append(
            int(((unmemorized_ranks >= start) & (unmemorized_ranks <= stop)).sum())
        )
        labels.append(f"{int(start)}-{int(stop)}")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.bar(range(len(counts)), counts, color="#c85f5f", alpha=0.85)
    ax.set_xlabel("task rank bin")
    ax.set_ylabel("unmemorized tasks")
    ax.set_title("Unmemorized task distribution by rank")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig
