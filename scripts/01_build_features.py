from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


META_COLUMNS = ["sample_id", "label", "talk_id", "persona_id", "profile_id", "source_file"]


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)
    if np.allclose(y, y[0]):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _time_to_fraction(values: list[float], fraction: float) -> int:
    if not values:
        return -1
    maximum = max(values)
    if maximum <= 0:
        return -1
    threshold = maximum * fraction
    for i, value in enumerate(values):
        if value >= threshold:
            return i
    return -1


def _jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def extract_one(path: Path, include_node_features: bool = True) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    meta = raw.get("input_meta") or {}
    ticks = raw.get("ticks") or []

    row: dict[str, Any] = {
        "sample_id": meta.get("sample_id") or path.parent.name,
        "label": meta.get("label"),
        "talk_id": meta.get("talk_id"),
        "persona_id": meta.get("persona_id"),
        "profile_id": meta.get("profile_id"),
        "source_file": path.as_posix(),
        "ticks": len(ticks),
    }

    active_counts: list[float] = []
    edge_counts: list[float] = []
    active_sets: list[set[int]] = []
    unique_nodes: set[int] = set()
    unique_edges: set[tuple[int, int]] = set()
    node_seen_count: dict[int, int] = {}
    neuron_type_counts = {"excitatory": 0, "inhibitory": 0, "modulatory": 0}
    k_values = {key: [] for key in neuron_type_counts}
    stim_norm_values = {key: [] for key in neuron_type_counts}

    max_node_id = -1

    for tick in ticks:
        active_nodes = {int(x) for x in (tick.get("active_nodes") or [])}
        edges = tick.get("edges_fired") or []
        node_states = tick.get("node_states") or []

        active_counts.append(float(len(active_nodes)))
        edge_counts.append(float(len(edges)))
        active_sets.append(active_nodes)

        unique_nodes.update(active_nodes)
        for node_id in active_nodes:
            node_seen_count[node_id] = node_seen_count.get(node_id, 0) + 1
            max_node_id = max(max_node_id, node_id)

        for edge in edges:
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                unique_edges.add((int(edge[0]), int(edge[1])))

        for state in node_states:
            node_id = state.get("node_id")
            if isinstance(node_id, int):
                max_node_id = max(max_node_id, node_id)

            neuron_type = state.get("neuron_type")
            if neuron_type not in neuron_type_counts:
                continue

            neuron_type_counts[neuron_type] += 1

            k = state.get("K")
            if isinstance(k, (int, float)):
                k_values[neuron_type].append(float(k))

            stim_vec = state.get("stim_vec")
            if isinstance(stim_vec, list) and stim_vec:
                arr = np.asarray(stim_vec, dtype=float)
                stim_norm_values[neuron_type].append(float(np.linalg.norm(arr)))

    network_size = max_node_id + 1 if max_node_id >= 0 else 0
    active_total = sum(active_counts)
    edge_total = sum(edge_counts)

    row.update(
        {
            "network_size_inferred": network_size,
            "active_mean": _safe_mean(active_counts),
            "active_std": _safe_std(active_counts),
            "active_max": max(active_counts, default=0.0),
            "active_min": min(active_counts, default=0.0),
            "active_auc": float(active_total),
            "active_slope_all": _slope(active_counts),
            "active_slope_early10": _slope(active_counts[:10]),
            "active_mean_early10": _safe_mean(active_counts[:10]),
            "active_mean_late10": _safe_mean(active_counts[-10:]),
            "time_to_50pct_active_max": _time_to_fraction(active_counts, 0.50),
            "time_to_90pct_active_max": _time_to_fraction(active_counts, 0.90),
            "zero_active_tick_ratio": (
                float(sum(v == 0 for v in active_counts) / len(active_counts))
                if active_counts
                else 0.0
            ),
            "active_ratio_mean": (
                float(_safe_mean(active_counts) / network_size) if network_size else 0.0
            ),
            "unique_active_nodes": len(unique_nodes),
            "unique_active_node_ratio": (
                float(len(unique_nodes) / network_size) if network_size else 0.0
            ),
            "edges_mean": _safe_mean(edge_counts),
            "edges_std": _safe_std(edge_counts),
            "edges_max": max(edge_counts, default=0.0),
            "edges_auc": float(edge_total),
            "edges_slope_early10": _slope(edge_counts[:10]),
            "unique_edges": len(unique_edges),
            "edge_reuse_ratio": (
                float(1.0 - (len(unique_edges) / edge_total)) if edge_total else 0.0
            ),
        }
    )

    consecutive_jaccard = [
        _jaccard(active_sets[i - 1], active_sets[i])
        for i in range(1, len(active_sets))
    ]
    row["active_set_jaccard_mean"] = _safe_mean(consecutive_jaccard)
    row["active_set_jaccard_std"] = _safe_std(consecutive_jaccard)

    total_state_rows = sum(neuron_type_counts.values())
    for neuron_type, short in [
        ("excitatory", "exc"),
        ("inhibitory", "inh"),
        ("modulatory", "mod"),
    ]:
        count = neuron_type_counts[neuron_type]
        row[f"{short}_state_count"] = count
        row[f"{short}_state_ratio"] = (
            float(count / total_state_rows) if total_state_rows else 0.0
        )
        row[f"{short}_K_mean"] = _safe_mean(k_values[neuron_type])
        row[f"{short}_K_std"] = _safe_std(k_values[neuron_type])
        row[f"{short}_stim_norm_mean"] = _safe_mean(stim_norm_values[neuron_type])
        row[f"{short}_stim_norm_std"] = _safe_std(stim_norm_values[neuron_type])

    if include_node_features and network_size:
        denominator = max(1, len(ticks))
        for node_id in range(network_size):
            row[f"node_freq_{node_id:03d}"] = node_seen_count.get(node_id, 0) / denominator

    return row


def find_raw_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.rglob("raw_trace.json"))
    if not files and raw_dir.is_file() and raw_dir.name.endswith(".json"):
        files = [raw_dir]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert nested EmoNet raw TRACE JSON files into one sample-level feature table."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument(
        "--no-node-features",
        action="store_true",
        help="Skip per-node activation frequency columns.",
    )
    args = parser.parse_args()

    files = find_raw_files(args.raw_dir)
    if not files:
        raise SystemExit(f"No raw_trace.json found under: {args.raw_dir}")

    rows = [
        extract_one(path, include_node_features=not args.no_node_features)
        for path in files
    ]
    df = pd.DataFrame(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"raw files : {len(files)}")
    print(f"rows      : {len(df)}")
    print(f"columns   : {len(df.columns)}")
    print(f"saved     : {args.output}")
    if "label" in df:
        print("\nlabel counts")
        print(df["label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
