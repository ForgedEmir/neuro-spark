"""Generate comparison charts from MLflow runs and log them as artifacts.

Reads all parent runs of the neuro-spark-eeg-v2 experiment, builds a comparison
figure of test metrics across runs, and logs it back as an artifact on a new
summary run. Also writes a standalone PNG to data/mlflow_report.png.
"""
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

TRACKING_URI = "sqlite:///mlruns/mlflow.db"
EXPERIMENT_NAME = "neuro-spark-eeg-v2"
OUTPUT_PNG = "data/mlflow_report.png"
METRICS = ["test_accuracy", "test_f1", "test_precision_weighted", "test_recall_weighted"]


def fetch_parent_runs():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment {EXPERIMENT_NAME} not found")
    runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time ASC"])
    parents = [r for r in runs if not r.data.tags.get("mlflow.parentRunId")]
    return [r for r in parents if all(m in r.data.metrics for m in METRICS)]


def build_figure(runs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [datetime.fromtimestamp(r.info.start_time / 1000).strftime("%H:%M:%S")
              for r in runs]
    x = range(len(runs))

    ax = axes[0]
    width = 0.2
    colors = ["#4C72B0", "#DD8452", "#55A467", "#C44E52"]
    for i, metric in enumerate(METRICS):
        values = [r.data.metrics[metric] for r in runs]
        offset = (i - 1.5) * width
        ax.bar([xi + offset for xi in x], values, width, label=metric, color=colors[i])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Test metrics per run")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.333, color="grey", linestyle="--", linewidth=1, label="Random (3 classes)")

    ax = axes[1]
    accuracy = [r.data.metrics["test_accuracy"] for r in runs]
    f1 = [r.data.metrics["test_f1"] for r in runs]
    ax.plot(labels, accuracy, marker="o", linewidth=2, label="test_accuracy", color="#4C72B0")
    ax.plot(labels, f1, marker="s", linewidth=2, label="test_f1", color="#DD8452")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Accuracy and F1 over runs")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"NeuroSpark EEG, {len(runs)} runs", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def main():
    runs = fetch_parent_runs()
    if not runs:
        print(f"No usable parent run found in {EXPERIMENT_NAME}")
        return

    print(f"Building report for {len(runs)} run(s)")
    fig = build_figure(runs)

    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=120, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG}")

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="comparison-report"):
        mlflow.log_figure(fig, "comparison.png")
        mlflow.log_param("n_runs_compared", len(runs))
        for i, run in enumerate(runs):
            for metric in METRICS:
                mlflow.log_metric(metric, run.data.metrics[metric], step=i)
        print("Logged comparison report as new MLflow run")

    plt.close(fig)


if __name__ == "__main__":
    main()
