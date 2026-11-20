"""
Turn the training metrics (checkpoints/*.metrics.jsonl, written by tiny_dlm.training.train) into graphs.

    python scripts/plot_metrics.py                    # -> docs/assets/plots/*.png + docs/assets/results.json

    rhoa_accuracy.png     validation accuracy per dataset: WITH pre-training vs WITHOUT (one panel per dataset)
    rhoa_loss.png         decision loss (train and validation): with vs without pre-training
    rhoa_pretraining.png  stage 1: masked-word loss and masked-word accuracy
    melon.png             Mēlon: loss, and how often it picks the teacher's steering (exact / within one notch)
    health.png            learning rate and gradient norm of every run (spikes = unstable training)
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
NAMES = {"babi": "bAbI (2-6 options)", "clinc150": "CLINC150 (151 options)",
         "banking77": "Banking77 (77 options)", "csqa": "CommonsenseQA (5 options)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11,
    "axes.titlecolor": INK, "legend.frameon": False,
})


def read(path):
    """-> (train records, eval records), or ([], []) if the run does not exist."""
    if not os.path.exists(path):
        return [], []
    rows = [json.loads(line) for line in open(path)]
    return [r for r in rows if r["kind"] == "train"], [r for r in rows if r["kind"] == "eval"]


def line(ax, rows, key, color, label, style="-"):
    rows = [r for r in rows if key in r]
    if rows:
        xs, ys = [r["step"] for r in rows], [r[key] for r in rows]
        ax.plot(xs, ys, style, color=color, label=label)
        ax.annotate(f"{ys[-1]:.0%}" if "acc" in key else f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)


def save(fig, out, name, top=None):
    fig.tight_layout()
    if top:  # room above the panels for a shared legend
        fig.subplots_adjust(top=top)
    path = os.path.join(out, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--out", default="docs/assets/plots")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    runs = {name: read(os.path.join(args.ckpt_dir, f"{name}.pt.metrics.jsonl"))
            for name in ("rhoa", "rhoa_scratch", "rhoa_pretrain", "melon")}
    results = {}

    # 1. accuracy per dataset: with vs without pre-training
    _, pre_ev = runs["rhoa"]
    _, scr_ev = runs["rhoa_scratch"]
    tasks = [t for t in NAMES if any(f"val_acc/{t}" in r for r in pre_ev + scr_ev)]
    if tasks:
        fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.2), sharey=True)
        for ax, t in zip(axes if len(tasks) > 1 else [axes], tasks):
            line(ax, pre_ev, f"val_acc/{t}", BLUE, "with pre-training")
            line(ax, scr_ev, f"val_acc/{t}", ORANGE, "without pre-training")
            ax.set_title(NAMES[t])
            ax.set_xlabel("training step")
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.set_ylim(0, 1.05)
            for label, ev in (("with_pretraining", pre_ev), ("without_pretraining", scr_ev)):
                vals = [r[f"val_acc/{t}"] for r in ev if f"val_acc/{t}" in r]
                if vals:
                    results.setdefault("rhoa", {}).setdefault(t, {})[label] = {"final": vals[-1], "best": max(vals)}
        (axes[0] if len(tasks) > 1 else axes).set_ylabel("validation accuracy (all options)")
        fig.legend(*axes[0].get_legend_handles_labels() if len(tasks) > 1 else axes.get_legend_handles_labels(),
                   loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
        save(fig, args.out, "rhoa_accuracy.png", top=0.8)

        # 2. decision loss
        fig, ax = plt.subplots(figsize=(6, 3.4))
        line(ax, pre_ev, "val_loss", BLUE, "with pre-training · validation")
        line(ax, scr_ev, "val_loss", ORANGE, "without pre-training · validation")
        line(ax, pre_ev, "train_loss", BLUE, "with pre-training · train", "--")
        line(ax, scr_ev, "train_loss", ORANGE, "without pre-training · train", "--")
        ax.set(title="Rhoa · decision loss (all datasets)", xlabel="training step", ylabel="cross-entropy over options")
        ax.legend(fontsize=8)
        save(fig, args.out, "rhoa_loss.png")

    # 3. pre-training
    _, ev = runs["rhoa_pretrain"]
    if ev:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
        line(a1, ev, "train_loss", BLUE, "train")
        line(a1, ev, "val_loss", ORANGE, "validation")
        a1.set(title="Pre-training · masked-word loss", xlabel="training step", ylabel="cross-entropy over the vocabulary")
        a1.legend()
        line(a2, ev, "val_acc/masked_word", BLUE, "validation")
        a2.set(title="Pre-training · hidden word guessed exactly", xlabel="training step")
        a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        results["rhoa_pretrain"] = {"val_loss": ev[-1]["val_loss"], "masked_word_acc": ev[-1].get("val_acc/masked_word")}
        save(fig, args.out, "rhoa_pretraining.png")

    # 4. Mēlon
    _, ev = runs["melon"]
    if ev:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
        line(a1, ev, "train_loss", BLUE, "train")
        line(a1, ev, "val_loss", ORANGE, "validation")
        a1.set(title="Mēlon · loss", xlabel="training step", ylabel="cross-entropy over 15 options")
        a1.legend()
        line(a2, ev, "val_acc/exact", BLUE, "the teacher's exact option")
        line(a2, ev, "val_acc/within_one_notch", AQUA, "steering within one notch")
        a2.set(title="Mēlon · validation frames (unseen tracks)", xlabel="training step")
        a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        a2.set_ylim(0, 1.05)
        a2.legend()
        results["melon"] = {"exact": ev[-1].get("val_acc/exact"), "within_one_notch": ev[-1].get("val_acc/within_one_notch")}
        save(fig, args.out, "melon.png")

    # 5. training health: learning rate and gradient norm
    present = [(n, tr) for n, (tr, _) in runs.items() if tr]
    if present:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
        for (name, tr), color in zip(present, (BLUE, ORANGE, AQUA, YELLOW)):
            a1.plot([r["step"] for r in tr], [r["lr"] for r in tr], color=color, label=name)
            a2.plot([r["step"] for r in tr], [r["grad_norm"] for r in tr], color=color, label=name)
        a1.set(title="Learning rate (warmup + cosine)", xlabel="training step")
        a2.set(title="Gradient norm before clipping", xlabel="training step", yscale="log")
        a1.legend(fontsize=8)
        save(fig, args.out, "health.png")

    path = os.path.join(os.path.dirname(args.out.rstrip("/")), "results.json")
    merged = json.load(open(path)) if os.path.exists(path) else {}  # keep the test results of evaluate_all.py
    merged["validation"] = results
    with open(path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
