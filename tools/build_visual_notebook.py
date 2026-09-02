import json
from pathlib import Path


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip() + "\n",
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip() + "\n",
    }


cells = [
md(r'''
# isThereEmotion — RAW TRACE에서 감정 관련 잠재 군집 찾기

> **연구 질문**  
> EmoNet의 RAW 뉴런 TRACE를 감정 정보 없이 분석했을 때 안정적인 내부 상태 군집이 발견되는가?  
> 또한 발견된 군집은 외부 감정 범주와 우연 이상의 연관성을 가지는가?

이 노트북 하나에서 **RAW JSON 전처리 → 라벨 블라인드 EDA → 시각화 → PCA → K-means → 계층적 군집화 → 군집 안정성 → 감정 라벨 사후 검증 → permutation test**까지 수행한다.

## 사용 도구

- **분석 계산:** NumPy, pandas
- **시각화만:** matplotlib
- **파일 처리:** Python 표준 라이브러리 `json`, `pathlib`

`matplotlib`은 이미 계산된 값의 표시만 담당하며 군집, PCA, 통계량 계산에는 사용하지 않는다.

## 분석 원칙

1. 군집 생성과 군집 수 선택에는 감정 `label`을 사용하지 않는다.
2. `top_emotions`, `dominant_global_signal`, latent `z_*`, 신경전달물질 모사값, style/LLM response 등 후단 감정 해석 결과는 feature에서 제외한다.
3. 군집 품질과 안정성을 먼저 확정한 뒤에만 감정 label을 공개한다.
4. 최종 결론은 “모델이 감정을 느낀다”가 아니라 **TRACE 내부에 감정과 체계적으로 연관된 잠재 구조가 존재하는지**에 한정한다.
'''),
code(r'''
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 120)

SEED = 42
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

EXPECTED_NODES = 256
PCA_VARIANCE_TARGET = 0.90
K_VALUES = range(2, 11)
N_INIT = 20
N_STABILITY = 30
N_PERMUTATIONS = 2000
N_NEGATIVE = 30

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("RAW directory:", RAW_DIR.resolve())
print("raw_trace.json files:", len(list(RAW_DIR.rglob("raw_trace.json"))))
'''),
md(r'''
## 1. RAW TRACE 전처리

각 `raw_trace.json`은 여러 tick과 뉴런 상태가 중첩된 계층형 데이터이다. 이를 sample-level 수치 특징으로 바꾼다.

사용하는 RAW 정보는 `active_nodes`, `edges_fired`, `node_states.neuron_type`, `K`, `stim_vec`, tick 순서이다. 감정 label은 별도 metadata에 보관하고 **feature table과 물리적으로 분리**한다.

전처리에는 JSON 통합, 실패 TRACE 제거, 새로운 열 생성, `K`의 log 변환, 결측치 중앙값 처리, 저분산 열 제거, Z-score 표준화가 포함된다.
'''),
code(r'''
def safe_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else 0.0


def safe_std(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.std()) if len(arr) else 0.0


def slope(values):
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)
    y = y[mask]
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
        tick_stim_mean.append(
            np.mean(np.vstack(tick_stim), axis=0)
            if tick_stim else np.full(4, np.nan)
        )

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
'''),
md(r'''
## 2. 라벨을 보지 않는 데이터 품질 확인과 시각화

이 단계에서는 감정 label을 사용하지 않는다.

먼저 실패하거나 사실상 비어 있는 TRACE를 제거하고, 대표적인 수치 특징의 분포를 확인한다. 그래프는 **어떤 감정인지 색을 입히지 않고** 전체 데이터 분포만 보여준다.
'''),
code(r'''
qc_mask = (features["ticks"] >= 3) & (features["unique_active_nodes"] > 0)
removed = features.loc[~qc_mask, ["sample_id", "ticks", "unique_active_nodes"]]
blind_features = features.loc[qc_mask].reset_index(drop=True)

print("usable samples :", len(blind_features))
print("removed samples:", len(removed))
if len(removed):
    print(removed.to_string(index=False))

plot_features = [
    "ticks",
    "active_ratio_mean",
    "edges_mean",
    "logK_mean",
]

for col in plot_features:
    plt.figure(figsize=(7, 4))
    plt.hist(blind_features[col].to_numpy(dtype=float), bins=12)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
'''),
md(r'''
## 3. 결측값 처리, 저분산 제거, 표준화

- 결측값은 전체 데이터의 중앙값으로 대체한다.
- 거의 일정한 열은 군집 거리에 의미 있는 정보를 주지 못하므로 제거한다.
- 서로 다른 단위를 가진 특징을 공정하게 비교하기 위해 Z-score 표준화를 적용한다.

감정 label은 여전히 사용하지 않는다.
'''),
code(r'''
feature_cols = [c for c in blind_features.columns if c != "sample_id"]
X_df = blind_features[feature_cols].apply(pd.to_numeric, errors="coerce")

medians = X_df.median(axis=0)
X_df = X_df.fillna(medians)

variances = X_df.var(axis=0, ddof=0)
keep_cols = variances[variances > 1e-10].index.tolist()
X_df = X_df[keep_cols]

X = X_df.to_numpy(dtype=float)
mu = X.mean(axis=0)
sigma = X.std(axis=0)
sigma[sigma == 0] = 1.0
Xz = (X - mu) / sigma

print("features before low-variance removal:", len(feature_cols))
print("features after  low-variance removal:", len(keep_cols))
print("standardized shape:", Xz.shape)
'''),
md(r'''
## 4. PCA — 고차원 TRACE 구조를 라벨 없이 보기

PCA는 NumPy의 SVD로 직접 계산한다. 먼저 설명분산을 확인하고, 누적 설명분산 90% 이상을 설명하는 PC들을 이후 군집 분석에 사용한다.

첫 PCA 산점도에서도 감정 label은 사용하지 않는다.
'''),
code(r'''
def pca_numpy(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    scores = U * S
    explained = (S ** 2) / np.sum(S ** 2)
    return scores, explained, Vt


pca_scores, explained, components = pca_numpy(Xz)
cumulative = np.cumsum(explained)
n_pc = int(np.searchsorted(cumulative, PCA_VARIANCE_TARGET) + 1)
X_pca = pca_scores[:, :n_pc]

print("PCs for >= 90% variance:", n_pc)
print("first 10 explained variance ratios:")
print(pd.Series(explained[:10], index=[f"PC{i+1}" for i in range(min(10, len(explained)))]))

plt.figure(figsize=(7, 4))
plt.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o")
plt.axhline(PCA_VARIANCE_TARGET, linestyle="--")
plt.axvline(n_pc, linestyle="--")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA cumulative explained variance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], alpha=0.8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Blind PCA projection — no emotion labels")
plt.tight_layout()
plt.show()
'''),
md(r'''
## 5. K-means를 NumPy로 직접 구현

`k=2~10`을 모두 시험하고 **Silhouette score만** 이용해 군집 수를 선택한다. 감정 label은 군집 수 선택에 사용하지 않는다.
'''),
code(r'''
def pairwise_distances(X):
    X = np.asarray(X, dtype=float)
    sq = np.sum(X * X, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2 * (X @ X.T)
    dist2 = np.maximum(dist2, 0.0)
    return np.sqrt(dist2)


def kmeans_once(X, k, seed=0, max_iter=300):
    rng = np.random.default_rng(seed)
    n = len(X)
    centers = X[rng.choice(n, size=k, replace=False)].copy()

    labels = np.full(n, -1, dtype=int)
    for _ in range(max_iter):
        d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        new_centers = centers.copy()
        for c in range(k):
            members = X[labels == c]
            if len(members):
                new_centers[c] = members.mean(axis=0)
            else:
                new_centers[c] = X[rng.integers(0, n)]
        centers = new_centers

    inertia = float(np.sum((X - centers[labels]) ** 2))
    return labels, centers, inertia


def kmeans(X, k, seed=0, n_init=20):
    best = None
    for i in range(n_init):
        result = kmeans_once(X, k, seed=seed + i)
        if best is None or result[2] < best[2]:
            best = result
    return best


def silhouette_score_numpy(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    D = pairwise_distances(X)
    unique = np.unique(labels)
    scores = []

    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        a = D[i, same].mean() if np.any(same) else 0.0

        b_values = []
        for c in unique:
            if c == labels[i]:
                continue
            other = labels == c
            if np.any(other):
                b_values.append(D[i, other].mean())
        b = min(b_values) if b_values else 0.0

        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)

    return float(np.mean(scores))


k_rows = []
k_results = {}

for k in K_VALUES:
    labels_k, centers_k, inertia_k = kmeans(X_pca, k, seed=SEED, n_init=N_INIT)
    sil_k = silhouette_score_numpy(X_pca, labels_k)
    k_rows.append({"k": k, "silhouette": sil_k, "inertia": inertia_k})
    k_results[k] = (labels_k, centers_k, inertia_k)

k_table = pd.DataFrame(k_rows)
best_k = int(k_table.loc[k_table["silhouette"].idxmax(), "k"])
best_labels = k_results[best_k][0]
best_silhouette = float(k_table.loc[k_table["k"] == best_k, "silhouette"].iloc[0])

print(k_table.to_string(index=False))
print("\nselected k:", best_k)
print("best silhouette:", best_silhouette)

plt.figure(figsize=(7, 4))
plt.plot(k_table["k"], k_table["silhouette"], marker="o")
plt.axvline(best_k, linestyle="--")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("K-means cluster quality by k")
plt.xticks(list(K_VALUES))
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], c=best_labels, alpha=0.85)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"K-means clusters in PCA space — k={best_k}")
plt.tight_layout()
plt.show()
'''),
md(r'''
## 6. 두 번째 알고리즘 — 계층적 군집화

K-means와 다른 방식에서도 유사한 구조가 나오는지 확인하기 위해 **average-linkage hierarchical clustering**을 직접 구현한다.

두 군집 알고리즘의 결과 일치도는 ARI로 비교한다.
'''),
code(r'''
def hierarchical_average_linkage(X, k):
    X = np.asarray(X, dtype=float)
    n = len(X)
    D = pairwise_distances(X)

    clusters = {i: [i] for i in range(n)}
    sizes = {i: 1 for i in range(n)}
    active = set(range(n))
    next_id = n

    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = D[i, j]

    def get_dist(a, b):
        return dist[(min(a, b), max(a, b))]

    while len(active) > k:
        active_list = sorted(active)
        best_pair = None
        best_distance = np.inf

        for idx, a in enumerate(active_list):
            for b in active_list[idx + 1:]:
                d = get_dist(a, b)
                if d < best_distance:
                    best_distance = d
                    best_pair = (a, b)

        a, b = best_pair
        new_id = next_id
        next_id += 1
        clusters[new_id] = clusters[a] + clusters[b]
        sizes[new_id] = sizes[a] + sizes[b]

        others = [c for c in active if c not in (a, b)]
        for c in others:
            d_new = (
                sizes[a] * get_dist(a, c) + sizes[b] * get_dist(b, c)
            ) / sizes[new_id]
            dist[(min(new_id, c), max(new_id, c))] = d_new

        active.remove(a)
        active.remove(b)
        active.add(new_id)

    labels = np.empty(n, dtype=int)
    for label, cluster_id in enumerate(sorted(active)):
        labels[clusters[cluster_id]] = label
    return labels


def comb2(n):
    n = np.asarray(n, dtype=float)
    return n * (n - 1) / 2


def adjusted_rand_index(labels_a, labels_b):
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    ua = np.unique(a)
    ub = np.unique(b)

    table = np.zeros((len(ua), len(ub)), dtype=int)
    for i, va in enumerate(ua):
        for j, vb in enumerate(ub):
            table[i, j] = np.sum((a == va) & (b == vb))

    sum_comb = np.sum(comb2(table))
    row_comb = np.sum(comb2(table.sum(axis=1)))
    col_comb = np.sum(comb2(table.sum(axis=0)))
    total_comb = comb2(len(a))

    expected = row_comb * col_comb / total_comb if total_comb else 0.0
    maximum = 0.5 * (row_comb + col_comb)
    denom = maximum - expected
    return float((sum_comb - expected) / denom) if denom else 1.0


hier_labels = hierarchical_average_linkage(X_pca, best_k)
hier_silhouette = silhouette_score_numpy(X_pca, hier_labels)
method_ari = adjusted_rand_index(best_labels, hier_labels)

print("hierarchical silhouette:", hier_silhouette)
print("K-means vs hierarchical ARI:", method_ari)

plt.figure(figsize=(7, 5))
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], c=hier_labels, alpha=0.85)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"Hierarchical clusters in PCA space — k={best_k}")
plt.tight_layout()
plt.show()
'''),
md(r'''
## 7. 군집 안정성

K-means의 초기 중심(seed)을 바꿔도 비슷한 군집이 유지되는지 ARI로 확인한다. 이 단계에서도 감정 label은 사용하지 않는다.
'''),
code(r'''
stability_rows = []
reference_labels = best_labels.copy()

for seed in range(N_STABILITY):
    labels_seed, _, _ = kmeans(X_pca, best_k, seed=seed * 101 + 7, n_init=1)
    ari = adjusted_rand_index(reference_labels, labels_seed)
    stability_rows.append({"seed": seed, "ARI_vs_reference": ari})

stability = pd.DataFrame(stability_rows)
print(stability.describe().to_string())

plt.figure(figsize=(7, 4))
plt.plot(stability["seed"], stability["ARI_vs_reference"], marker="o")
plt.ylim(-0.05, 1.05)
plt.xlabel("Seed run")
plt.ylabel("ARI vs reference clustering")
plt.title("K-means cluster stability across seeds")
plt.tight_layout()
plt.show()
'''),
md(r'''
## 8. Negative control — 구조를 깨뜨리면 군집도 약해지는가?

각 feature 열의 sample 순서를 독립적으로 섞으면 원래 sample 내부에서 함께 움직이던 특징 조합이 깨진다. 이 데이터에서 얻은 최대 Silhouette와 실제 데이터의 최대 Silhouette를 비교한다.

이 단계 역시 감정 label과 무관한 검증이다.
'''),
code(r'''
rng = np.random.default_rng(SEED)
negative_best_sil = []

for rep in range(N_NEGATIVE):
    shuffled = X_pca.copy()
    for j in range(shuffled.shape[1]):
        shuffled[:, j] = rng.permutation(shuffled[:, j])

    rep_scores = []
    for k in K_VALUES:
        labels_rep, _, _ = kmeans(shuffled, k, seed=rep + 1000, n_init=5)
        rep_scores.append(silhouette_score_numpy(shuffled, labels_rep))
    negative_best_sil.append(max(rep_scores))

negative_best_sil = np.asarray(negative_best_sil)
print("observed best silhouette:", best_silhouette)
print("negative-control mean   :", negative_best_sil.mean())
print("negative-control max    :", negative_best_sil.max())

plt.figure(figsize=(7, 4))
plt.hist(negative_best_sil, bins=10, alpha=0.8)
plt.axvline(best_silhouette, linewidth=2, label="Observed")
plt.xlabel("Best silhouette after feature shuffling")
plt.ylabel("Count")
plt.title("Negative-control distribution")
plt.legend()
plt.tight_layout()
plt.show()
'''),
md(r'''
# 9. 여기서 처음 감정 label 공개

위 단계까지는 감정 label을 전혀 사용하지 않았다. 이제 고정된 군집 결과와 외부 감정 label의 관계를 검증한다.

세부 감정 label의 표본 수가 작더라도, permutation test에서는 **같은 label 빈도 구조를 그대로 유지한 채 label의 위치만 무작위로 섞기 때문에** 우연한 대응인지 직접 비교할 수 있다.
'''),
code(r'''
used_ids = blind_features["sample_id"].to_numpy()
label_map = metadata.set_index("sample_id")["label"]
y_label = np.asarray([label_map.get(sid) for sid in used_ids], dtype=object)

validation_table = pd.DataFrame({
    "sample_id": used_ids,
    "cluster": best_labels,
    "emotion_label": y_label,
})

print("emotion label counts:")
print(validation_table["emotion_label"].value_counts(dropna=False).to_string())

cross = pd.crosstab(validation_table["cluster"], validation_table["emotion_label"])
print("\ncluster × emotion label table:")
print(cross.to_string())
'''),
md(r'''
## 10. 군집과 감정의 연관성 — NMI + permutation test

NMI는 군집과 감정 label이 얼마나 많은 정보를 공유하는지 측정한다. 하지만 관측 NMI 값 자체만으로는 충분하지 않다.

그래서 cluster는 고정하고 감정 label만 무작위로 2,000번 섞어 **같은 label 분포에서 우연히 얻을 수 있는 NMI 분포**를 만든다.
'''),
code(r'''
def normalized_mutual_information(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    ua, a_inv = np.unique(a, return_inverse=True)
    ub, b_inv = np.unique(b, return_inverse=True)

    table = np.zeros((len(ua), len(ub)), dtype=float)
    for i in range(len(a)):
        table[a_inv[i], b_inv[i]] += 1.0

    table /= table.sum()
    pa = table.sum(axis=1)
    pb = table.sum(axis=0)

    mi = 0.0
    for i in range(len(ua)):
        for j in range(len(ub)):
            p = table[i, j]
            if p > 0 and pa[i] > 0 and pb[j] > 0:
                mi += p * np.log(p / (pa[i] * pb[j]))

    ha = -np.sum(pa[pa > 0] * np.log(pa[pa > 0]))
    hb = -np.sum(pb[pb > 0] * np.log(pb[pb > 0]))
    denom = np.sqrt(ha * hb)
    return float(mi / denom) if denom > 0 else 0.0


observed_nmi = normalized_mutual_information(best_labels, y_label)

rng = np.random.default_rng(SEED)
perm_nmi = np.empty(N_PERMUTATIONS, dtype=float)
for i in range(N_PERMUTATIONS):
    perm_nmi[i] = normalized_mutual_information(best_labels, rng.permutation(y_label))

p_value_nmi = (1 + np.sum(perm_nmi >= observed_nmi)) / (N_PERMUTATIONS + 1)

print("observed NMI:", observed_nmi)
print("permutation mean:", perm_nmi.mean())
print("permutation p-value:", p_value_nmi)

plt.figure(figsize=(7, 4))
plt.hist(perm_nmi, bins=25, alpha=0.8)
plt.axvline(observed_nmi, linewidth=2, label="Observed NMI")
plt.xlabel("NMI under shuffled emotion labels")
plt.ylabel("Count")
plt.title("Permutation test: cluster–emotion association")
plt.legend()
plt.tight_layout()
plt.show()
'''),
md(r'''
## 11. 독립적인 보강 증거 — 같은 감정끼리 TRACE가 더 가까운가?

군집 알고리즘을 거치지 않고 PCA 공간의 모든 sample 쌍을 직접 비교한다.

- 같은 감정 label을 가진 sample 간 평균 거리
- 다른 감정 label을 가진 sample 간 평균 거리

를 비교하고, label permutation으로 그 차이가 우연인지 검증한다.
'''),
code(r'''
D = pairwise_distances(X_pca)
upper_i, upper_j = np.triu_indices(len(X_pca), k=1)
pair_dist = D[upper_i, upper_j]

same_mask = y_label[upper_i] == y_label[upper_j]
observed_same = float(pair_dist[same_mask].mean()) if np.any(same_mask) else np.nan
observed_diff = float(pair_dist[~same_mask].mean()) if np.any(~same_mask) else np.nan
observed_gap = observed_diff - observed_same

rng = np.random.default_rng(SEED + 1)
perm_gap = np.empty(N_PERMUTATIONS, dtype=float)
for i in range(N_PERMUTATIONS):
    yp = rng.permutation(y_label)
    same_p = yp[upper_i] == yp[upper_j]
    same_mean = pair_dist[same_p].mean() if np.any(same_p) else np.nan
    diff_mean = pair_dist[~same_p].mean() if np.any(~same_p) else np.nan
    perm_gap[i] = diff_mean - same_mean

p_value_gap = (1 + np.sum(perm_gap >= observed_gap)) / (N_PERMUTATIONS + 1)

print("mean distance — same emotion     :", observed_same)
print("mean distance — different emotion:", observed_diff)
print("distance gap (different - same)  :", observed_gap)
print("permutation p-value              :", p_value_gap)

plt.figure(figsize=(6, 4))
plt.bar(["Same emotion", "Different emotion"], [observed_same, observed_diff])
plt.ylabel("Mean pairwise distance")
plt.title("TRACE distance by emotion agreement")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.hist(perm_gap, bins=25, alpha=0.8)
plt.axvline(observed_gap, linewidth=2, label="Observed distance gap")
plt.xlabel("Different − same distance under shuffled labels")
plt.ylabel("Count")
plt.title("Permutation test: same-emotion proximity")
plt.legend()
plt.tight_layout()
plt.show()
'''),
md(r'''
## 12. 결과 요약 및 저장

결론은 다음 순서로 해석한다.

1. **Silhouette가 낮고 negative control과 차이가 작다** → 뚜렷한 잠재 군집 증거가 부족하다.
2. **안정적인 군집은 있으나 감정 NMI가 permutation보다 높지 않다** → 내부 구조는 있지만 감정 구조라고 보기 어렵다.
3. **안정적인 군집 + 감정과 유의한 NMI + 같은 감정끼리 더 가까움** → TRACE 내부에 감정과 체계적으로 연관된 잠재 구조가 있다는 복수의 증거가 된다.

어떤 경우에도 이 결과만으로 EmoNet이 실제로 감정을 “느낀다”고 주장하지 않는다.
'''),
code(r'''
summary = pd.DataFrame([
    {
        "usable_samples": len(X_pca),
        "raw_feature_count": len(feature_cols),
        "kept_feature_count": len(keep_cols),
        "pca_components_90pct": n_pc,
        "selected_k": best_k,
        "kmeans_silhouette": best_silhouette,
        "hierarchical_silhouette": hier_silhouette,
        "kmeans_vs_hierarchical_ARI": method_ari,
        "seed_stability_ARI_mean": stability["ARI_vs_reference"].mean(),
        "negative_control_silhouette_mean": negative_best_sil.mean(),
        "observed_NMI": observed_nmi,
        "NMI_permutation_p": p_value_nmi,
        "same_emotion_distance": observed_same,
        "different_emotion_distance": observed_diff,
        "distance_gap": observed_gap,
        "distance_permutation_p": p_value_gap,
    }
])

summary.to_csv(RESULTS_DIR / "summary.csv", index=False, encoding="utf-8-sig")
k_table.to_csv(RESULTS_DIR / "k_selection.csv", index=False, encoding="utf-8-sig")
stability.to_csv(RESULTS_DIR / "cluster_stability.csv", index=False, encoding="utf-8-sig")
validation_table.to_csv(RESULTS_DIR / "cluster_emotion_mapping.csv", index=False, encoding="utf-8-sig")

print(summary.T.to_string(header=False))
print("\nSaved results to:", RESULTS_DIR.resolve())
'''),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path("isThereEmotion.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1),
    encoding="utf-8",
)

Path("requirements.txt").write_text("numpy\npandas\nmatplotlib\n", encoding="utf-8")
print("built isThereEmotion.ipynb with", len(cells), "cells")
