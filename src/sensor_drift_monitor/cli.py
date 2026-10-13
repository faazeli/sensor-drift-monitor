"""`sensor-drift-monitor benchmark | drift | all` — reproduce the README results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer

from . import data, experiments

app = typer.Typer(no_args_is_help=True, help="Sensor fault detection and drift monitoring on the UCI gas sensor array.")
DataOpt = typer.Option("data/Dataset", help="Folder containing batch1.dat ... batch10.dat")
OutOpt = typer.Option("results", help="Where to write CSVs, tables and figures")


@app.command()
def benchmark(data_dir: str = DataOpt, out: Path = OutOpt):
    """Inject synthetic faults into healthy data and score every detector."""
    out.mkdir(exist_ok=True)
    res = experiments.fault_benchmark(data.load(data_dir))
    res.to_csv(out / "fault_benchmark_raw.csv", index=False)
    summary = experiments.summarize_benchmark(res)
    summary.to_csv(out / "fault_benchmark_summary.csv")
    auc = summary["roc_auc"]
    (out / "fault_benchmark_auc.md").write_text(auc.to_markdown(floatfmt=".3f"), encoding="utf-8")
    typer.echo(auc.to_string())
    rec = summary["recall"]
    (out / "fault_benchmark_recall.md").write_text(rec.to_markdown(floatfmt=".2f"), encoding="utf-8")
    typer.echo("\nRecall at ~1% false-alarm threshold:\n" + rec.to_string())
    plot_heatmap(auc, out / "fault_benchmark_auc.png")


def plot_heatmap(auc, path: Path) -> None:
    auc = auc[auc.mean().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150)
    im = ax.imshow(auc.values, cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(auc.columns)), auc.columns, fontsize=8)
    ax.set_yticks(range(len(auc.index)), auc.index, fontsize=8)
    for i in range(auc.shape[0]):
        for j in range(auc.shape[1]):
            v = auc.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.8 else "black")
    ax.set_title("Fault detection ROC-AUC (5 seeds, 20% of samples faulted)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@app.command()
def drift(data_dir: str = DataOpt, out: Path = OutOpt):
    """Replay months 11-36 against models frozen on batches 1-2 (months 1-10)."""
    out.mkdir(exist_ok=True)
    study = experiments.drift_study(data.load(data_dir))
    study.to_csv(out / "drift_study.csv", index=False)
    (out / "drift_study.md").write_text(study.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
    corr = experiments.drift_correlations(study)
    (out / "drift_correlations.md").write_text(corr.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
    typer.echo(study.to_string(index=False))
    typer.echo("\n" + corr.to_string(index=False))
    typer.echo(f"\nFirst alarms: {experiments.first_alarm_batch(study)}")
    plot_drift(study, out / "drift_study.png")


def plot_drift(study, path: Path) -> None:
    x = range(len(study))
    labels = [
        f"{b}\n(m {m})" if not b.startswith("1-2") else "ref\n(m 1-10)" for b, m in zip(study.batch, study.months)
    ]
    fig, ax1 = plt.subplots(figsize=(9, 4.8), dpi=150)
    ax1.plot(x, study.gas_accuracy, "o-", color="#1f4e79", lw=2.2, label="Gas classifier accuracy (needs labels)")
    ax1.set_ylabel("Accuracy", color="#1f4e79")
    ax1.set_ylim(0, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(x, study.domain_auc, "s--", color="#c0504d", label="Domain-classifier AUC (label-free)")
    ax2.plot(x, study.t2_alarm_rate, "^--", color="#e3a21a", label="Hotelling T² alarm rate (label-free)")
    ax2.plot(x, study.spe_alarm_rate, "v:", color="#7f7f7f", label="PCA-SPE alarm rate (label-free)")
    ax2.set_ylabel("Drift signal")
    ax2.set_ylim(0, 1.02)
    ax1.set_xticks(list(x), labels, fontsize=8)
    ax1.set_xlabel("Batch (months since deployment)")
    ax1.set_title("Sensor drift: label-free monitors vs. the accuracy they are meant to protect", fontsize=10)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=False)
    for s in ("top",):
        ax1.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@app.command(name="all")
def run_all(data_dir: str = DataOpt, out: Path = OutOpt):
    """Run both studies."""
    benchmark(data_dir, out)
    drift(data_dir, out)


def main():
    app()
