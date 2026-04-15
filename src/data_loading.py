import logging
import os
import random
import tarfile
import urllib.request

import networkx as nx
import requests

from .config import DATA_DIR


def _download_file(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return path
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    with open(path, "wb") as file:
        file.write(response.content)
    return path


def load_cora_graph() -> nx.Graph:
    """
    Downloads and loads the original Cora citation network as undirected graph
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cora_tgz_path = os.path.join(DATA_DIR, "cora.tgz")
    cora_dir = os.path.join(DATA_DIR, "cora")
    cites_path = os.path.join(cora_dir, "cora.cites")

    if not os.path.exists(cites_path):
        logging.info("Downloading Cora dataset from LINQS...")
        url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
        urllib.request.urlretrieve(url, cora_tgz_path)

        with tarfile.open(cora_tgz_path, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
        logging.info("Extraction complete.")

    graph = nx.Graph()
    with open(cites_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                # Cora cites format: <cited_paper_id> <citing_paper_id>
                source, target = parts
                graph.add_edge(int(source), int(target))

    if graph.number_of_nodes() == 0:
        raise ValueError("Graph is empty. Check the data parsing.")

    if not nx.is_connected(graph):
        largest_component = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_component).copy()

    # Relabel nodes to sequential integers (0, 1, 2, ..., N-1)
    # This is critical for LLM Token efficiency! (Prevents bloated IDs like '1061127')
    graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering="default")

    print(
        f"Cora Network LCC successfully loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges."
    )

    return graph


def load_facebook_graph():
    """Load Facebook Social Circles (SNAP) as an undirected graph """
    url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
    path = _download_file(url, os.path.join(DATA_DIR, "facebook_combined.txt.gz"))
    graph = nx.read_edgelist(path, nodetype=int)

    if graph.number_of_nodes() > 0 and not nx.is_connected(graph):
        largest_component = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_component).copy()

    return graph


def sample_connected_subgraph(G, target_nodes=300, seed=42):
    """Return a connected induced subgraph with up to target_nodes nodes."""
    if G.number_of_nodes() <= target_nodes:
        return G.copy()

    rng = random.Random(seed)

    # Work inside the largest connected component for stable pathfinding tasks
    lcc_nodes = max(nx.connected_components(G), key=len)
    H = G.subgraph(lcc_nodes).copy()

    if H.number_of_nodes() <= target_nodes:
        return H

    start = rng.choice(list(H.nodes()))
    selected = {start}
    frontier = [start]

    while frontier and len(selected) < target_nodes:
        node = frontier.pop(0)
        for nb in H.neighbors(node):
            if nb not in selected:
                selected.add(nb)
                frontier.append(nb)
                if len(selected) >= target_nodes:
                    break

    # If BFS neighborhood is still short - top up with random nodes from LCC
    if len(selected) < target_nodes:
        remaining = list(set(H.nodes()) - selected)
        rng.shuffle(remaining)
        selected.update(remaining[: target_nodes - len(selected)])

    sampled = H.subgraph(selected).copy()

    if not nx.is_connected(sampled):
        cc = max(nx.connected_components(sampled), key=len)
        sampled = sampled.subgraph(cc).copy()

    return sampled


def load_test_graph(dataset_name):
    """Load graph by dataset name or generate a synthetic graph """
    name = dataset_name.lower().strip()
    if name in {"synthetic", "er", "erdos", "erdos_renyi"}:
        return nx.erdos_renyi_graph(n=100, p=0.1)
    if name == "cora":
        return load_cora_graph()
    if name in {"facebook", "facebook_snap", "snap_facebook"}:
        return load_facebook_graph()
    raise ValueError(f"Unsupported dataset: {dataset_name}")
