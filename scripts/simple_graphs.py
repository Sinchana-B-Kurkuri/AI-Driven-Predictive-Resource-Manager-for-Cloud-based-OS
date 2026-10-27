"""Five simple bar charts of the headline results in the report.

Usage (from the repository root):

    python -m scripts.simple_graphs            # rerun the studies, then show the graphs
    python -m scripts.simple_graphs --no-rerun # only redraw from the CSVs already in results/

Each chart is a plain bar chart with the value printed on every bar.
Our method is drawn in blue, the alternatives in grey.

    1. CPU: how long urgent processes wait (priority-critical workloads)
    2. CPU: response time on I/O-bound workloads
    3. Memory: page faults per policy (locality pattern, 8 frames)
    4. Device: share of mild faults caught, per detector
    5. Device: SMART disk-failure model, real Backblaze data

Copies are saved to results/simple/.
"""

import argparse
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = "results"
OUT = os.path.join(RESULTS, "simple")
CHOSEN = "#2a78d6"
OTHER = "#b9b8b1"

STUDIES = [
    ["experiments.cpu_eval"],
    ["experiments.memory_eval"],
    ["experiments.model_selection_eval", "--only", "device"],
]

SCHEDULERS = ["AI-selected", "Priority", "SJF", "FCFS", "Round Robin"]

DETECTORS = {
    "transformer (in use)": "Transformer (ours)",
    "one_class_svm": "One-Class SVM",
    "z_score_max": "z-score",
    "local_outlier_factor": "LOF",
    "pca_reconstruction": "PCA",
    "isolation_forest": "Isolation Forest",
}

SMART_MODELS = {
    "random_forest": "Random Forest (ours)",
    "logistic_regression": "Logistic regression",
    "rule_baseline": "Simple rule",
}


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS, name))


def bar_chart(names, values, chosen, title, ylabel, filename, fmt="{:.1f}", ylim=None) -> None:
    """One plain bar chart: value on top of each bar, our bars in blue."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [CHOSEN if name in chosen else OTHER for name in names]
    bars = ax.bar(names, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(value),
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=15, ha="right", fontsize=11)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, filename), dpi=150)


def cpu_chart(workload, column, title, ylabel, filename) -> None:
    summary = read("cpu_summary.csv")
    sub = summary[summary.workload_type == workload].set_index("scheduler").loc[SCHEDULERS]
    bar_chart(SCHEDULERS, sub[f"{column}_mean"].tolist(), ["AI-selected"], title, ylabel, filename)


def memory_faults() -> None:
    summary = read("memory_replacement_summary.csv")
    sub = summary[(summary.pattern == "locality") & (summary.frames == 8)]
    order = ["FIFO", "LRU", "Predictive-protect", "Predictive-prefetch", "Optimal"]
    sub = sub.set_index("policy").loc[order]
    names = ["FIFO", "LRU", "Ours: protect", "Ours: prefetch", "Optimal (best possible)"]
    bar_chart(names, sub.page_faults_mean.tolist(), ["Ours: protect", "Ours: prefetch"],
              "Memory: page faults (lower is better)",
              "Page faults per 500 references", "3_memory_faults.png", fmt="{:.0f}")


def device_recall() -> None:
    d = read("model_selection_device.csv").set_index("model").loc[list(DETECTORS)]
    bar_chart(list(DETECTORS.values()), d["recall_severity 0.05"].tolist(), ["Transformer (ours)"],
              "Device: mild faults caught (higher is better)",
              "Share of faults caught", "4_device_recall.png", fmt="{:.0%}", ylim=(0, 1))


def smart() -> None:
    m = read("smart_metrics.csv").set_index("model").loc[list(SMART_MODELS)]
    ours = m.loc["random_forest"]
    bar_chart(list(SMART_MODELS.values()), m.roc_auc_mean.tolist(), ["Random Forest (ours)"],
              f"Device: disk-failure prediction (higher is better)\n"
              f"Ours flags {ours.drive_recall_mean:.1%} of failing drives, "
              f"median {ours.median_lead_days_mean:.0f} days before failure",
              "ROC-AUC", "5_smart.png", fmt="{:.3f}", ylim=(0.8, 0.92))


def main() -> None:
    parser = argparse.ArgumentParser(description="Five simple bar charts of the headline results.")
    parser.add_argument("--no-rerun", action="store_true", help="redraw from the existing CSVs only")
    args = parser.parse_args()
    if not args.no_rerun:
        for study in STUDIES:
            print("Running", " ".join(study), "...", flush=True)
            subprocess.run([sys.executable, "-m", *study], check=True)

    cpu_chart("PRIORITY_CRITICAL", "high_priority_waiting",
              "CPU: waiting time of urgent processes (lower is better)",
              "Average waiting time", "1_cpu_urgent_waiting.png")
    cpu_chart("IO_BOUND", "avg_response_time",
              "CPU: response time on I/O-bound work (lower is better)",
              "Average response time", "2_cpu_io_response.png")
    memory_faults()
    device_recall()
    smart()

    print(f"Saved copies in {OUT}/. Close the windows to finish.")
    plt.show()


if __name__ == "__main__":
    main()
