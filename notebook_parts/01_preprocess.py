def safe_mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def safe_std(values):
    return float(np.std(values)) if len(values) else 0.0


def slope(values):
    y = np.asarray(values, dtype=float)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sum(x * x)
    return float(np.sum(x * y) / denom) if denom > 0 else 0.0


def time_to_fraction(values, fraction):
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


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def signed_log1p(x):
    x = float(x)
    return np.sign(x) * np.log1p(abs(x))


def extract_trace(path, expected_nodes=EXPECTED_NODES):
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    meta = raw.get("input_meta") or {}
    ticks = raw.get("ticks") or []

    metadata = {
        "sample_id": meta.get("sample_id") or path.parent.name,
        "label": meta.get("label"),
        "talk_id": meta.get("talk_id"),
        "persona_id": meta.get("persona_id"),
        "profile_id": meta.get("profile_id"),
        "source_file": path.as_posix(),
    }

    active_counts = []
    active_sets = []
    edge_counts = []
    unique_nodes = set()
    unique_edges = set()
    node_seen = np.zeros(expected_nodes, dtype=int)

    type_counts = {"excitatory": 0, "inhibitory": 0, "modulatory": 0}
    type_logk = {key: [] for key in type_counts}

    tick_logk_mean = []
    tick_stim_mean = []
    max_node_id = -1

    for tick in ticks:
        active = {int(x) for x in (tick.get("active_nodes") or [])}
        edges = tick.get("edges_fired") or []
        states = tick.get("node_states") or []

        active_counts.append(len(active))
        active_sets.append(active)
        edge_counts.append(len(edges))
        unique_nodes.update(active)

        for node_id in active:
            max_node_id = max(max_node_id, node_id)
            if 0 <= node_id < expected_nodes:
                node_seen[node_id] += 1

        for edge in edges:
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                unique_edges.add((int(edge[0]), int(edge[1])))

        tick_logk = []
        tick_stim = []

        for state in states:
            node_id = state.get("node_id")
            if isinstance(node_id, int):
                max_node_id = max(max_node_id, node_id)

            neuron_type = state.get("neuron_type")
            if neuron_type in type_counts:
                type_counts[neuron_type] += 1

            k = state.get("K")
            if isinstance(k, (int, float)):
                lk = signed_log1p(k)
                tick_logk.append(lk)
                if neuron_type in type_logk:
                    type_logk[neuron_type].append(lk)

            stim = state.get("stim_vec")
            if isinstance(stim, list) and len(stim) == 4:
                try:
                    tick_stim.append(np.asarray(stim, dtype=float))
                except (TypeError, ValueError):
                    pass

        tick_logk_mean.append(safe_mean(tick_logk))
        if tick_stim:
            tick_stim_mean.append(np.mean(np.vstack(tick_stim), axis=0))
        else:
            tick_stim_mean.append(np.full(4, np.nan))

    inferred_nodes = max(expected_nodes, max_node_id + 1 if max_node_id >= 0 else 0)
    tick_count = len(ticks)
    active_total = float(np.sum(active_counts))
    edge_total = float(np.sum(edge_counts))

    consecutive_jaccard = [
        jaccard(active_sets[i - 1], active_sets[i])
        for i in range(1, len(active_sets))
    ]

    feature = {
        "sample_id": metadata["sample_id"],
        "ticks": tick_count,
        "network_size_inferred": inferred_nodes,
        "active_mean": safe_mean(active_counts),
        "active_std": safe_std(active_counts),
        "active_max": max(active_counts, default=0),
        "active_auc": active_total,
        "active_ratio_mean": safe_mean(active_counts) / inferred_nodes if inferred_nodes else 0.0,
        "active_slope_all": slope(active_counts),
        "active_slope_early10": slope(active_counts[:10]),
        "active_mean_early10": safe_mean(active_counts[:10]),
        "active_mean_late10": safe_mean(active_counts[-10:]),
        "time_to_50pct_active_max": time_to_fraction(active_counts, 0.50),
        "time_to_90pct_active_max": time_to_fraction(active_counts, 0.90),
        "zero_active_tick_ratio": np.mean(np.asarray(active_counts) == 0) if active_counts else 1.0,
        "unique_active_nodes": len(unique_nodes),
        "unique_active_node_ratio": len(unique_nodes) / inferred_nodes if inferred_nodes else 0.0,
        "edges_mean": safe_mean(edge_counts),
        "edges_std": safe_std(edge_counts),
        "edges_max": max(edge_counts, default=0),
        "edges_auc": edge_total,
        "edges_slope_all": slope(edge_counts),
        "edges_slope_early10": slope(edge_counts[:10]),
        "edge_per_active": edge_total / active_total if active_total else 0.0,
        "unique_edges": len(unique_edges),
        "edge_reuse_ratio": 1.0 - len(unique_edges) / edge_total if edge_total else 0.0,
        "active_set_jaccard_mean": safe_mean(consecutive_jaccard),
        "active_set_jaccard_std": safe_std(consecutive_jaccard),
        "logK_mean": safe_mean(tick_logk_mean),
        "logK_std": safe_std(tick_logk_mean),
        "logK_slope": slope(tick_logk_mean),
    }

    total_type_states = sum(type_counts.values())
    for neuron_type, short in [
        ("excitatory", "exc"),
        ("inhibitory", "inh"),
        ("modulatory", "mod"),
    ]:
        feature[f"{short}_state_ratio"] = (
            type_counts[neuron_type] / total_type_states if total_type_states else 0.0
        )
        feature[f"{short}_logK_mean"] = safe_mean(type_logk[neuron_type])
        feature[f"{short}_logK_std"] = safe_std(type_logk[neuron_type])

    stim_matrix = np.vstack(tick_stim_mean) if tick_stim_mean else np.empty((0, 4))
    for j in range(4):
        values = stim_matrix[:, j] if len(stim_matrix) else np.asarray([])
        valid = values[np.isfinite(values)] if len(values) else values
        feature[f"stim{j}_mean"] = safe_mean(valid)
        feature[f"stim{j}_std"] = safe_std(valid)
        feature[f"stim{j}_slope"] = slope(valid)

    denom = max(tick_count, 1)
    rates = node_seen / denom
    feature["node_activation_rate_mean"] = float(rates.mean())
    feature["node_activation_rate_std"] = float(rates.std())
    feature["variable_node_ratio"] = float(np.mean(rates * (1 - rates) > 0.05))
    feature["high_persistence_node_ratio"] = float(np.mean(rates >= 0.90))

    p = np.clip(rates, 1e-12, 1 - 1e-12)
    feature["node_binary_entropy_mean"] = float(
        np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p)))
    )

    for node_id, rate in enumerate(rates):
        feature[f"node_freq_{node_id:03d}"] = float(rate)

    return metadata, feature


raw_files = sorted(RAW_DIR.rglob("raw_trace.json"))
metadata_rows = []
feature_rows = []

for i, path in enumerate(raw_files, start=1):
    meta, feature = extract_trace(path)
    metadata_rows.append(meta)
    feature_rows.append(feature)
    if i % 20 == 0 or i == len(raw_files):
        print(f"processed {i:3d}/{len(raw_files)}")

metadata = pd.DataFrame(metadata_rows)
features = pd.DataFrame(feature_rows)

metadata.to_csv(PROCESSED_DIR / "metadata_labels.csv", index=False, encoding="utf-8-sig")
features.to_csv(PROCESSED_DIR / "trace_features.csv", index=False, encoding="utf-8-sig")

print("metadata shape:", metadata.shape)
print("feature shape :", features.shape)

# label과 무관한 품질 기준으로 실패/빈 TRACE 제거
qc_mask = (features["ticks"] >= 3) & (features["unique_active_nodes"] > 0)
removed = features.loc[~qc_mask, ["sample_id", "ticks", "unique_active_nodes"]]
blind_features = features.loc[qc_mask].reset_index(drop=True)

print("usable samples :", len(blind_features))
print("removed samples:", len(removed))
if len(removed):
    print(removed.to_string(index=False))

feature_cols = [c for c in blind_features.columns if c != "sample_id"]
X_df = blind_features[feature_cols].apply(pd.to_numeric, errors="coerce")
X_df = X_df.apply(lambda s: s.fillna(s.median()))

# 거의 변하지 않는 feature는 거리 계산에 의미가 없으므로 제거
std = X_df.std(ddof=0)
keep_cols = std[std > 1e-12].index.tolist()
X_df = X_df[keep_cols]

mean = X_df.mean()
std = X_df.std(ddof=0).replace(0, 1.0)
X_z = ((X_df - mean) / std).to_numpy(dtype=float)

print("features before low-variance removal:", len(feature_cols))
print("features after  low-variance removal:", len(keep_cols))
print("standardized matrix:", X_z.shape)


def pca_numpy(X, variance_target=0.90):
    X = np.asarray(X, dtype=float)
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    eigenvalues = (S ** 2) / max(len(X) - 1, 1)
    explained = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained)
    n_components = int(np.searchsorted(cumulative, variance_target) + 1)
    components = Vt[:n_components]
    scores = X_centered @ components.T
    return scores, components, explained, cumulative


X_pca, pca_components, explained, cumulative = pca_numpy(X_z, PCA_VARIANCE_TARGET)
pca_summary = pd.DataFrame({
    "PC": np.arange(1, len(explained) + 1),
    "explained_variance_ratio": explained,
    "cumulative_ratio": cumulative,
})

print("PCA components used:", X_pca.shape[1])
print("cumulative explained variance:", cumulative[X_pca.shape[1] - 1])


def ascii_hist(values, bins=10, width=36):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=bins)
    max_count = max(counts.max(), 1)
    lines = []
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        bar = "█" * int(round(width * count / max_count))
        lines.append(f"{left:9.3f} ~ {right:9.3f} | {bar} {count}")
    return "\n".join(lines)


for col in ["active_ratio_mean", "edges_mean", "logK_mean", "active_set_jaccard_mean"]:
    if col in blind_features:
        print("\n[" + col + "]")
        print(ascii_hist(blind_features[col], bins=10))

pca_summary.head(15)
