import json
import os

import networkx as nx
import pandas as pd

from .config import OUTPUT_DIR
from .data_loading import load_test_graph, sample_connected_subgraph
from .evaluation import evaluate_baseline_on_trials, evaluate_framework_on_trials, generate_pathfinding_trials

SYNTHETIC_RESULTS_PATH = os.path.join(OUTPUT_DIR, "evaluation_results_graph_sizes.json")
SAMPLED_REPORT_PATH = os.path.join(OUTPUT_DIR, "comprehensive_report_cora_facebook_sampled.json")


def run_synthetic_graph_size_experiments(graph_sizes=None, retrieval_strategies=None, trials_per_size=10):
    """Run shared-trial synthetic experiments across graph sizes and retrieval strategies """

    if graph_sizes is None:
        graph_sizes = [20, 50, 100]
    if retrieval_strategies is None:
        retrieval_strategies = ["betweenness", "path_aware", "degree"]

    final_results = {
        "graph_sizes": graph_sizes,
        "retrieval_strategies": retrieval_strategies,
        "trials_per_size": trials_per_size,
        "runs": {},
    }

    for N in graph_sizes:
        graph = nx.erdos_renyi_graph(n=N, p=0.1)
        shared_trials = generate_pathfinding_trials(graph, num_trials=trials_per_size)
        print(f"\nPrecomputing shared baseline for synthetic_N{N} across {len(shared_trials)} trials...")
        shared_baseline = evaluate_baseline_on_trials(
            graph,
            shared_trials,
            num_nodes=N,
            graph_name=f"synthetic_N{N}",
        )
        final_results["runs"][str(N)] = {}
        for strategy in retrieval_strategies:
            final_results["runs"][str(N)][strategy] = evaluate_framework_on_trials(
                num_nodes=N,
                graph=graph,
                graph_name=f"synthetic_N{N}",
                dataset_strategy=strategy,
                trials=shared_trials,
                baseline_results=shared_baseline,
            )

    return final_results


def save_synthetic_results(final_results, output_path=SYNTHETIC_RESULTS_PATH):
    """Save full synthetic experiment results to JSON for later analysis """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(final_results, file, indent=2, ensure_ascii=False)
    return output_path


def build_synthetic_summary_rows(final_results):
    """Aggregate synthetic trial results into per size per strategy summary rows """
    summary_rows = []
    for N, size_runs in final_results["runs"].items():
        for strategy, run_data in size_runs.items():
            baseline_measurements = [trial["baseline"]["llm_measurement"] for trial in run_data["trials"]]
            rag_measurements = [trial["graph_rag"]["llm_measurement"] for trial in run_data["trials"]]

            def pct(value, total):
                return 100.0 * value / total if total else 0.0

            total = len(baseline_measurements)
            baseline_conn = sum(1 for m in baseline_measurements if m["connectivity_correct"])
            baseline_path = sum(1 for m in baseline_measurements if m["path_fidelity_correct"])
            rag_conn = sum(1 for m in rag_measurements if m["connectivity_correct"])
            rag_path = sum(1 for m in rag_measurements if m["path_fidelity_correct"])

            summary_rows.append(
                {
                    "N": int(N),
                    "strategy": strategy,
                    "trials": total,
                    "baseline_conn_pct": pct(baseline_conn, total),
                    "baseline_path_pct": pct(baseline_path, total),
                    "rag_conn_pct": pct(rag_conn, total),
                    "rag_path_pct": pct(rag_path, total),
                    "conn_delta": pct(rag_conn - baseline_conn, total),
                    "path_delta": pct(rag_path - baseline_path, total),
                    "baseline_conf": sum(m.get("confidence", 0.5) for m in baseline_measurements) / total
                    if total
                    else 0.0,
                    "rag_conf": sum(m.get("confidence", 0.5) for m in rag_measurements) / total if total else 0.0,
                }
            )
    return summary_rows


def build_synthetic_summary_df(final_results):
    return pd.DataFrame(build_synthetic_summary_rows(final_results)).sort_values(["N", "strategy"])


def build_synthetic_summary_pivot(summary_df):
    return summary_df.pivot(index="N", columns="strategy", values="rag_path_pct")


def run_sampled_dataset_report(trials=10, report_datasets=None, report_strategies=None):
    """ Run shared trial evaluation on sampled real datasets and build summary report """
    if report_datasets is None:
        report_datasets = [
            {"graph_name": "cora", "sample_nodes": 300, "trials": trials},
            {"graph_name": "facebook", "sample_nodes": 300, "trials": trials},
        ]
    if report_strategies is None:
        report_strategies = ["betweenness", "path_aware", "degree"]

    report_rows = []
    sampled_runs = {}

    for cfg in report_datasets:
        dataset_name = cfg["graph_name"]
        full_graph = load_test_graph(dataset_name)
        graph = sample_connected_subgraph(full_graph, target_nodes=cfg["sample_nodes"], seed=42)

        sampled_runs[dataset_name] = {}

        print(f"\n=== {dataset_name.upper()} sampled run ===")
        print(
            f"full: nodes={full_graph.number_of_nodes()} edges={full_graph.number_of_edges()} | "
            f"sampled: nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}"
        )
        shared_trials = generate_pathfinding_trials(graph, num_trials=cfg["trials"])
        print(f"Precomputing shared baseline for {dataset_name}_sampled across {len(shared_trials)} trials...")
        shared_baseline = evaluate_baseline_on_trials(
            graph,
            shared_trials,
            num_nodes=graph.number_of_nodes(),
            graph_name=f"{dataset_name}_sampled",
        )

        for strategy in report_strategies:
            run_result = evaluate_framework_on_trials(
                num_nodes=graph.number_of_nodes(),
                graph=graph,
                graph_name=f"{dataset_name}_sampled",
                dataset_strategy=strategy,
                trials=shared_trials,
                baseline_results=shared_baseline,
            )
            sampled_runs[dataset_name][strategy] = run_result

            baseline_measurements = [t["baseline"]["llm_measurement"] for t in run_result["trials"]]
            rag_measurements = [t["graph_rag"]["llm_measurement"] for t in run_result["trials"]]

            total = len(baseline_measurements)
            baseline_conn = sum(1 for m in baseline_measurements if m["connectivity_correct"])
            baseline_path = sum(1 for m in baseline_measurements if m["path_fidelity_correct"])
            rag_conn = sum(1 for m in rag_measurements if m["connectivity_correct"])
            rag_path = sum(1 for m in rag_measurements if m["path_fidelity_correct"])

            report_rows.append(
                {
                    "dataset": dataset_name,
                    "strategy": strategy,
                    "trials": cfg["trials"],
                    "sample_nodes": graph.number_of_nodes(),
                    "sample_edges": graph.number_of_edges(),
                    "baseline_conn_pct": 100.0 * baseline_conn / total if total else 0.0,
                    "baseline_path_pct": 100.0 * baseline_path / total if total else 0.0,
                    "rag_conn_pct": 100.0 * rag_conn / total if total else 0.0,
                    "rag_path_pct": 100.0 * rag_path / total if total else 0.0,
                    "baseline_avg_conf": sum(m.get("confidence", 0.5) for m in baseline_measurements) / total
                    if total
                    else 0.0,
                    "rag_avg_conf": sum(m.get("confidence", 0.5) for m in rag_measurements) / total
                    if total
                    else 0.0,
                }
            )

    report_df = pd.DataFrame(report_rows).sort_values(["dataset", "strategy"]).reset_index(drop=True)
    return report_df, sampled_runs, report_rows


def save_sampled_dataset_report(report_rows, output_path=SAMPLED_REPORT_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump({"rows": report_rows}, file, indent=2, ensure_ascii=False)
    return output_path
