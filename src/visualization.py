import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURES_DIR
from .experiments import SAMPLED_REPORT_PATH, SYNTHETIC_RESULTS_PATH


def build_synthetic_dashboard_df(final_results=None, synthetic_results_path=SYNTHETIC_RESULTS_PATH):
    """Build synthetic dashboard metrics from in memory results or saved JSON """
    synthetic_rows = []
    if final_results is not None and isinstance(final_results, dict) and "runs" in final_results:
        src_runs = final_results["runs"]
    else:
        with open(synthetic_results_path, "r", encoding="utf-8") as file:
            src_runs = json.load(file)["runs"]

    for n_key, size_runs in src_runs.items():
        for strategy, run_data in size_runs.items():
            baseline_measurements = [t["baseline"]["llm_measurement"] for t in run_data["trials"]]
            rag_measurements = [t["graph_rag"]["llm_measurement"] for t in run_data["trials"]]
            total = len(baseline_measurements)
            if total == 0:
                continue

            b_conn = sum(1 for m in baseline_measurements if m["connectivity_correct"]) * 100.0 / total
            b_path = sum(1 for m in baseline_measurements if m["path_fidelity_correct"]) * 100.0 / total
            r_conn = sum(1 for m in rag_measurements if m["connectivity_correct"]) * 100.0 / total
            r_path = sum(1 for m in rag_measurements if m["path_fidelity_correct"]) * 100.0 / total

            synthetic_rows.append(
                {
                    "N": int(n_key),
                    "strategy": strategy,
                    "baseline_conn_pct": b_conn,
                    "baseline_path_pct": b_path,
                    "rag_conn_pct": r_conn,
                    "rag_path_pct": r_path,
                    "conn_delta": r_conn - b_conn,
                    "path_delta": r_path - b_path,
                }
            )

    return pd.DataFrame(synthetic_rows).sort_values(["N", "strategy"])


def build_real_dashboard_df(report_df=None, sampled_report_path=SAMPLED_REPORT_PATH):
    """Build real data dashboard metrics from report DataFrame or saved JSON """
    if report_df is not None and isinstance(report_df, pd.DataFrame) and len(report_df) > 0:
        real_df = report_df.copy()
    else:
        with open(sampled_report_path, "r", encoding="utf-8") as file:
            real_df = pd.DataFrame(json.load(file)["rows"])

    real_df["conn_delta"] = real_df["rag_conn_pct"] - real_df["baseline_conn_pct"]
    real_df["path_delta"] = real_df["rag_path_pct"] - real_df["baseline_path_pct"]
    real_df = real_df.sort_values(["dataset", "strategy"]).reset_index(drop=True)
    return real_df


def plot_synthetic_dashboard(
    final_results=None,
    synthetic_results_path=SYNTHETIC_RESULTS_PATH,
    figure_path=os.path.join(FIGURES_DIR, "synthetic_evaluation_dashboard.png"),
):
    """Create and save the synthetic evaluation dashboard """
    synthetic_df = build_synthetic_dashboard_df(
        final_results=final_results,
        synthetic_results_path=synthetic_results_path,
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    heat = synthetic_df.pivot(index="N", columns="strategy", values="path_delta")
    im = axes[0].imshow(heat.values, cmap="RdYlGn", aspect="auto", vmin=-50, vmax=50)
    axes[0].set_title("Synthetic: Path Fidelity Gain (RAG - Baseline)")
    axes[0].set_xticks(range(len(heat.columns)))
    axes[0].set_xticklabels(heat.columns, rotation=25, ha="right")
    axes[0].set_yticks(range(len(heat.index)))
    axes[0].set_yticklabels([str(x) for x in heat.index])
    axes[0].set_xlabel("Retrieval Strategy")
    axes[0].set_ylabel("Graph Size N")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            axes[0].text(j, i, f"{heat.values[i, j]:.0f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    size_avg = (
        synthetic_df.groupby("N")[["baseline_path_pct", "rag_path_pct", "baseline_conn_pct", "rag_conn_pct"]]
        .mean()
        .reset_index()
    )
    axes[1].plot(size_avg["N"], size_avg["baseline_path_pct"], marker="o", linewidth=2, label="Baseline Path")
    axes[1].plot(size_avg["N"], size_avg["rag_path_pct"], marker="o", linewidth=2, label="RAG Path")
    axes[1].plot(
        size_avg["N"],
        size_avg["baseline_conn_pct"],
        marker="s",
        linestyle="--",
        linewidth=1.8,
        label="Baseline Conn",
    )
    axes[1].plot(
        size_avg["N"],
        size_avg["rag_conn_pct"],
        marker="s",
        linestyle="--",
        linewidth=1.8,
        label="RAG Conn",
    )
    axes[1].set_title("Synthetic: Performance vs Graph Size (Avg over strategies)")
    axes[1].set_xlabel("Graph Size N")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    plt.tight_layout()
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.show()

    return {
        "figure": fig,
        "figure_path": figure_path,
        "synthetic_df": synthetic_df,
    }


def plot_real_dashboard(
    report_df=None,
    sampled_report_path=SAMPLED_REPORT_PATH,
    figure_path=os.path.join(FIGURES_DIR, "real_evaluation_dashboard.png"),
):
    """Create and save the real data evaluation dashboard """
    real_df = build_real_dashboard_df(
        report_df=report_df,
        sampled_report_path=sampled_report_path,
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    real_path_gain = real_df.pivot(index="dataset", columns="strategy", values="path_delta")
    real_path_gain.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Real Data: Path Fidelity Gain (RAG - Baseline)")
    axes[0].set_xlabel("Dataset")
    axes[0].set_ylabel("Gain (percentage points)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(title="Strategy")

    tradeoff = real_df.groupby("strategy")[["conn_delta", "path_delta"]].mean().reset_index()
    axes[1].axhline(0, color="gray", linewidth=1)
    axes[1].axvline(0, color="gray", linewidth=1)
    for _, row in tradeoff.iterrows():
        axes[1].scatter(row["conn_delta"], row["path_delta"], s=140)
        axes[1].annotate(
            row["strategy"],
            (row["conn_delta"], row["path_delta"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axes[1].set_title("Real Data: Strategy Trade-off")
    axes[1].set_xlabel("Avg Connectivity Gain (pp)")
    axes[1].set_ylabel("Avg Path Gain (pp)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.show()

    return {
        "figure": fig,
        "figure_path": figure_path,
        "real_df": real_df,
        "tradeoff": tradeoff,
    }


def plot_evaluation_dashboard(
    final_results=None,
    report_df=None,
    synthetic_results_path=SYNTHETIC_RESULTS_PATH,
    sampled_report_path=SAMPLED_REPORT_PATH,
    figure_path=os.path.join(FIGURES_DIR, "evaluation_dashboard.png"),
):
    """Create separate synthetic and real data dashboards """
    base_path, ext = os.path.splitext(figure_path)
    synthetic_figure_path = f"{base_path}_synthetic{ext or '.png'}"
    real_figure_path = f"{base_path}_real{ext or '.png'}"

    synthetic_dashboard = plot_synthetic_dashboard(
        final_results=final_results,
        synthetic_results_path=synthetic_results_path,
        figure_path=synthetic_figure_path,
    )
    real_dashboard = plot_real_dashboard(
        report_df=report_df,
        sampled_report_path=sampled_report_path,
        figure_path=real_figure_path,
    )

    return {
        "synthetic": synthetic_dashboard,
        "real": real_dashboard,
        "figure_paths": {
            "synthetic": synthetic_dashboard["figure_path"],
            "real": real_dashboard["figure_path"],
        },
        "synthetic_df": synthetic_dashboard["synthetic_df"],
        "real_df": real_dashboard["real_df"],
        "tradeoff": real_dashboard["tradeoff"],
    }
