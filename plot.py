from pathlib import Path
import matplotlib.pyplot as plt

def plot_history(history, ax=None):
    """Plot mislabeled rate against iteration and return the axes."""
    if not history:
        raise ValueError("history must contain at least one iteration")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    iterations = [row["iteration"] for row in history]
    error_rates = [row["mislabeled_rate"] for row in history]

    ax.plot(iterations, error_rates)
    ax.set_title("Classification Error")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mislabeled Rate")

    return ax

def plot_variant_comparison(
    signed_zero_history,
    signed_three_history,
    signed_n_history,
    unsigned_history,
    cut_history,
    classifier="Poisson",
    ax=None,
):
    """Compare signed edge-addition budgets with unsigned and cut baselines."""

    if classifier not in ["Poisson", "Laplace"]:
        raise ValueError("classifier must be 'Poisson' or 'Laplace'")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    histories = [
        (
            signed_zero_history,
            f"{classifier} signed — max 0 added edges per sign",
            "tab:blue",
            "-",
        ),
        (
            signed_three_history,
            f"{classifier} signed — max 3 added edges per sign",
            "tab:orange",
            "--",
        ),
        (
            signed_n_history,
            rf"{classifier} signed — max $n$ edges per sign",
            "tab:green",
            "-.",
        ),
        (unsigned_history, f"{classifier} unsigned", "tab:red", ":"),

        (cut_history, f"{classifier} cut edges", "tab:purple", (0, (3, 1, 1, 1))),
    ]

    for history, label, color, linestyle in histories:
        plot_history(history, ax=ax)
        line = ax.lines[-1]
        line.set_color(color)
        line.set_linestyle(linestyle)
        line.set_linewidth(2)
        line.set_label(label)

    ax.set_title(f"{classifier} Edge-Addition and Variant Comparison")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mislabeled Rate")
    ax.legend()
    return ax


def save_plot(history, path):
    # Ensures the destination directory exists:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    ax = plot_history(history)
    figure = ax.figure
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
