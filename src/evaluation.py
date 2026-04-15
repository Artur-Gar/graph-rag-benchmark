import json
import random
import re
import time

import networkx as nx
import requests

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL
from .data_loading import load_test_graph


def local_generate(prompt, model_name=OLLAMA_MODEL, timeout=180, max_retries=3, backoff_sec=2):
    """ Call local LLM via Ollama API with retry on timeout/connection errors """
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()["response"]
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == max_retries - 1:
                raise
            sleep_for = backoff_sec * (2**attempt)
            print(f"local_generate retry {attempt + 1}/{max_retries - 1} in {sleep_for:.1f}s")
            time.sleep(sleep_for)


def linearize_graph(G):
    """Convert graph to incident-encoded text format"""
    nodes = list(G.nodes())
    if not nodes:
        return "The graph is empty."

    description = f"Graph with nodes: {', '.join(map(str, nodes))}.\n"
    for node in nodes:
        neighbors = list(G.neighbors(node))
        if neighbors:
            if len(neighbors) > 1:
                neighbors_str = ", ".join(map(str, neighbors[:-1])) + f", and {neighbors[-1]}"
            else:
                neighbors_str = str(neighbors[0])
            description += f"Node {node} -> [{neighbors_str}]\n"
        else:
            description += f"Node {node} -> isolated\n"
    return description


def retrieve_subgraph(G, source, target, k_hops=2, max_nodes=18, strategy="betweenness"):
    """
    IMPORTANT
    Retrieve context-aware subgraph via adaptive k-hop expansion with intelligent scoring.

    strategy:
      - "betweenness": prioritize bridge nodes
      - "path_aware": prioritize nodes close to both endpoints
      - "degree": prioritize hub nodes
    """
    if source not in G or target not in G:
        return G.subgraph([n for n in [source, target] if n in G]).copy()

    if not nx.has_path(G, source, target):
        return G.subgraph([source, target]).copy()

    try:
        path_nodes = set(nx.shortest_path(G, source, target))
    except nx.NetworkXNoPath:
        path_nodes = {source, target}

    relevant = path_nodes.copy()
    frontier = path_nodes.copy()

    for _ in range(k_hops):
        if len(relevant) >= max_nodes * 0.8:
            break
        new_frontier = set()
        for node in frontier:
            new_frontier.update(G.neighbors(node))
        frontier = new_frontier - relevant
        relevant.update(frontier)
        if not frontier:
            break

    if strategy == "betweenness":
        betweenness = nx.betweenness_centrality(G.subgraph(relevant))
        scores = {node: betweenness.get(node, 0.0) for node in relevant}
        for node in path_nodes:
            scores[node] += 1.0
    elif strategy == "path_aware":
        scores = {}
        for node in relevant:
            if node in path_nodes:
                scores[node] = 1e6
            else:
                d_src = nx.shortest_path_length(G, source, node) if nx.has_path(G, source, node) else float("inf")
                d_tgt = nx.shortest_path_length(G, node, target) if nx.has_path(G, node, target) else float("inf")
                scores[node] = -(d_src + d_tgt) + 0.01 * G.degree(node)
    else:
        scores = {node: G.degree(node) for node in relevant}
        for node in path_nodes:
            scores[node] = 1e6

    ranked = sorted(relevant, key=lambda n: scores.get(n, -1e9), reverse=True)
    kept = set(path_nodes)
    for node in ranked:
        if len(kept) >= max_nodes:
            break
        kept.add(node)

    return G.subgraph(kept).copy()


def generate_cog_prompt(graph_text, source, target):
    """Generate Chain-of-Graph prompt with endpoint constraints."""
    return f"""{graph_text}
        Question: Is there a path from Node {source} to Node {target}? If yes, provide the shortest path.

        Instruction: Step through neighbors of {source}, then trace connectivity to {target}.
        IMPORTANT: The shortest path MUST begin with {source} and end with {target}.

        Return JSON: {{"is_connected": bool, "shortest_path": [int, ...] or null}}
        Answer: 
    """


def generate_pathfinding_task(G):
    """Generate a pathfinding task with ground truth."""
    nodes = list(G.nodes())
    if len(nodes) < 2:
        raise ValueError("Graph must contain at least two nodes.")

    for _ in range(200):
        source, target = random.sample(nodes, 2)
        if nx.has_path(G, source, target):
            shortest_path = nx.shortest_path(G, source=source, target=target)
            truth_bin = "Yes"
            truth_path = " -> ".join(map(str, shortest_path))
            return source, target, truth_bin, truth_path

    raise RuntimeError("Could not find a connected source/target pair after multiple attempts.")


def generate_pathfinding_trials(G, num_trials=3):
    """Generate a reusable set of pathfinding trials for a graph """
    trials = []
    for _ in range(num_trials):
        source, target, truth_bin, truth_path = generate_pathfinding_task(G)
        trials.append(
            {
                "source": source,
                "target": target,
                "truth_bin": truth_bin,
                "truth_path": truth_path,
            }
        )
    return trials


def _print_evaluation_header(graph_name, num_nodes, graph, strategy_label):
    print(f"\n=== Dataset: {graph_name} | Graph Size N={num_nodes if graph is not None else 'auto'} | Strategy: {strategy_label} ===")
    print(f"Graph stats: nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")


def _parse_structured_response(response_text):
    """Extract JSON object from model response, including fenced blocks"""
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_path_regex(response_text):
    """ Regex fallback for common path formats"""
    patterns = [
        r"(\d+(?:\s*->\s*\d+)+)",
        r"\[(\d+(?:,\s*\d+)*)\]",
        r"path[:\s]+(\d+(?:[,\s]+\d+)*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if not match:
            continue
        path_str = match.group(1)
        if "->" in path_str:
            nodes = [int(n.strip()) for n in path_str.split("->")]
        else:
            nodes = [int(n.strip()) for n in re.split(r"[,\s]+", path_str) if n.strip().isdigit()]
        if nodes:
            return "Yes", nodes
    return None, None


def _coerce_node_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if re.fullmatch(r"[-+]?\d+", s):
            try:
                return int(s)
            except ValueError:
                return None
    return None


def _normalize_path_nodes(path):
    if path is None or not isinstance(path, list):
        return []
    out = []
    for item in path:
        node = _coerce_node_int(item)
        if node is not None:
            out.append(node)
    return out


def _extract_json_dict_from_response_text(response_text):
    """Deterministically parse JSON dict from response text/code blocks"""
    text = response_text.strip()

    block_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    for block in reversed(block_matches):
        try:
            obj = json.loads(block.strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    for match in reversed(list(re.finditer(r"\{[\s\S]*?\}", text))):
        candidate = match.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


def _validate_path_sequence(predicted_path, response_text):
    """ Check whether extracted path nodes are present in response text"""
    if not predicted_path:
        return True
    for node in predicted_path:
        if str(node) not in response_text:
            return False
    return True


def _llm_extract_response(response_text):
    """ Extract connectivity/path: deterministic parse first tehn LLM fallback second """
    parsed_direct = _extract_json_dict_from_response_text(response_text)
    if isinstance(parsed_direct, dict):
        is_conn = parsed_direct.get("is_connected")
        path = _normalize_path_nodes(parsed_direct.get("shortest_path"))
        if isinstance(is_conn, bool):
            return ("Yes" if is_conn else "No"), path

    prompt = f"""Extract connectivity and path from the response below.

        IMPORTANT:
        - is_connected: true if path exists, false otherwise
        - shortest_path: ONLY the sequence of node numbers from source to target
        * If path: [1, 2, 3, 4] (a list of integers in order)
        * If no path: null or []
        - Ignore any explanations; return ONLY the node numbers

        Response: {response_text}

        Return ONLY valid JSON: {{"is_connected": bool, "shortest_path": [int, ...] or null}}
    """

    raw = local_generate(prompt).strip()
    parsed = _parse_structured_response(raw)
    if isinstance(parsed, dict):
        is_conn = parsed.get("is_connected")
        path = _normalize_path_nodes(parsed.get("shortest_path"))
        if isinstance(is_conn, bool):
            return ("Yes" if is_conn else "No"), path

    fallback_conn, fallback_path = _extract_path_regex(response_text)
    if fallback_conn:
        return fallback_conn, _normalize_path_nodes(fallback_path)

    return None, None


def _llm_measure_prediction(response_text, truth_bin, truth_path):
    """ Extract prediction and score connectivity/path fidelity deterministically """
    pred_conn, pred_path = _llm_extract_response(response_text)

    if pred_conn is None or pred_path is None:
        return {"connectivity_correct": False, "path_fidelity_correct": False, "confidence": 0.0}

    path_valid = _validate_path_sequence(pred_path, response_text)
    conn_correct = pred_conn == truth_bin

    if truth_bin == "Yes":
        gt_nodes = [int(x.strip()) for x in truth_path.split("->")]
        path_correct = pred_path == gt_nodes
        if not path_correct and len(pred_path) == len(gt_nodes) and len(pred_path) > 0:
            path_correct = path_valid
    else:
        path_correct = pred_path == []

    confidence = 1.0 if path_valid else 0.5

    return {
        "connectivity_correct": bool(conn_correct),
        "path_fidelity_correct": bool(path_correct),
        "confidence": confidence,
        "path_validated": path_valid,
    }


def _run_baseline_trial(graph, source, target, truth_bin, truth_path):
    baseline_text = linearize_graph(graph)
    baseline_prompt = generate_cog_prompt(baseline_text, source, target)
    baseline_response = local_generate(baseline_prompt).strip()
    baseline_measure = _llm_measure_prediction(baseline_response, truth_bin, truth_path)
    return {
        "response_text": baseline_response,
        "llm_measurement": baseline_measure,
    }


def evaluate_baseline_on_trials(graph, trials, num_nodes=None, graph_name="synthetic"):
    """Evaluate the shared baseline once for a fixed set of trials """
    _print_evaluation_header(graph_name, num_nodes, graph, "all nodes (baseline)")
    baseline_results = []
    for i, trial in enumerate(trials):
        print(f"[Trial {i + 1}] Path {trial['source']}->{trial['target']} | Truth: {trial['truth_bin']}")
        baseline_results.append(
            _run_baseline_trial(
                graph,
                trial["source"],
                trial["target"],
                trial["truth_bin"],
                trial["truth_path"],
            )
        )
    return baseline_results


def _copy_baseline_result(baseline_result):
    return {
        "response_text": baseline_result["response_text"],
        "llm_measurement": dict(baseline_result["llm_measurement"]),
    }


def _run_graph_rag_trial(graph, source, target, truth_bin, truth_path, dataset_strategy="betweenness"):
    pruned_G = retrieve_subgraph(graph, source, target, k_hops=2, strategy=dataset_strategy)
    rag_text = linearize_graph(pruned_G)
    rag_prompt = f"""{rag_text}
                    Question: Is there a path from Node {source} to Node {target}? Provide the shortest path.

                    Focus Instructions:
                    1. Start from Node {source}, trace neighbors that move CLOSER to Node {target}
                    2. Each step, pick the neighbor that minimizes distance to {target}
                    3. The path MUST begin with {source} and end with {target}
                    4. Return the node sequence as JSON

                    Return JSON: {{"is_connected": bool, "shortest_path": [int, ...] or null}}
                    Answer:
                """
    rag_response = local_generate(rag_prompt).strip()
    rag_measure = _llm_measure_prediction(rag_response, truth_bin, truth_path)
    return {
        "response_text": rag_response,
        "llm_measurement": rag_measure,
    }


def evaluate_framework_on_trials(
    num_nodes=30,
    graph=None,
    graph_name="synthetic",
    dataset_strategy="betweenness",
    trials=None,
    baseline_results=None,
):
    """ Run evaluation on fixed set of trials optionally reusing baseline outputs """
    if graph is None:
        graph = load_test_graph(graph_name)
    if trials is None:
        trials = generate_pathfinding_trials(graph, num_trials=3)
    if baseline_results is not None and len(baseline_results) != len(trials):
        raise ValueError("baseline_results must have the same length as trials.")

    _print_evaluation_header(graph_name, num_nodes, graph, dataset_strategy)
    results = {"num_nodes": num_nodes, "num_trials": len(trials), "graph_name": graph_name, "trials": []}

    for i, trial in enumerate(trials):
        source = trial["source"]
        target = trial["target"]
        truth_bin = trial["truth_bin"]
        truth_path = trial["truth_path"]

        print(f"[Trial {i + 1}] Path {source}->{target} | Truth: {truth_bin}")

        if baseline_results is None:
            baseline_result = _run_baseline_trial(graph, source, target, truth_bin, truth_path)
        else:
            baseline_result = _copy_baseline_result(baseline_results[i])

        graph_rag_result = _run_graph_rag_trial(
            graph,
            source,
            target,
            truth_bin,
            truth_path,
            dataset_strategy=dataset_strategy,
        )

        trial_result = {
            "trial_index": i + 1,
            "source": source,
            "target": target,
            "ground_truth": {"connected": truth_bin, "shortest_path": truth_path},
            "baseline": baseline_result,
            "graph_rag": graph_rag_result,
        }
        results["trials"].append(trial_result)

    return results


def evaluate_framework(num_nodes=30, num_trials=3, graph=None, graph_name="synthetic", dataset_strategy="betweenness"):
    """Run evaluation: Baseline vs Graph-RAG"""
    if graph is None:
        graph = load_test_graph(graph_name)

    _print_evaluation_header(graph_name, num_nodes, graph, dataset_strategy)
    results = {"num_nodes": num_nodes, "num_trials": num_trials, "graph_name": graph_name, "trials": []}

    for i in range(num_trials):
        source, target, truth_bin, truth_path = generate_pathfinding_task(graph)

        print(f"[Trial {i + 1}] Path {source}->{target} | Truth: {truth_bin}")

        baseline_result = _run_baseline_trial(graph, source, target, truth_bin, truth_path)
        graph_rag_result = _run_graph_rag_trial(
            graph,
            source,
            target,
            truth_bin,
            truth_path,
            dataset_strategy=dataset_strategy,
        )

        trial_result = {
            "trial_index": i + 1,
            "source": source,
            "target": target,
            "ground_truth": {"connected": truth_bin, "shortest_path": truth_path},
            "baseline": baseline_result,
            "graph_rag": graph_rag_result,
        }
        results["trials"].append(trial_result)

    return results
