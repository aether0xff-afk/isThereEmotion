# 여기부터 감정 label 봉인 해제: K*, PCA, 군집 결과를 먼저 확정한 뒤에만 실행한다.
valid_ids = blind_features["sample_id"].tolist()
validation_meta = (
    metadata
    .set_index("sample_id")
    .loc[valid_ids]
    .reset_index()
)

emotion = validation_meta["label"].astype(str).to_numpy()

print("emotion classes:", pd.Series(emotion).nunique())
print("samples        :", len(emotion))
print("\nlabel counts")
print(pd.Series(emotion).value_counts().to_string())


def nmi_score(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)

    contingency = np.zeros((ai.max() + 1, bi.max() + 1), dtype=float)
    np.add.at(contingency, (ai, bi), 1.0)

    n = contingency.sum()
    pxy = contingency / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    expected = px @ py
    nonzero = pxy > 0

    mutual_information = np.sum(
        pxy[nonzero] * np.log(pxy[nonzero] / expected[nonzero])
    )

    px1 = px.ravel()
    py1 = py.ravel()
    hx = -np.sum(px1[px1 > 0] * np.log(px1[px1 > 0]))
    hy = -np.sum(py1[py1 > 0] * np.log(py1[py1 > 0]))
    denom = np.sqrt(hx * hy)
    return float(mutual_information / denom) if denom > 0 else 0.0


def permutation_test_nmi(cluster_labels, emotion_labels, repeats=2000, seed=42):
    observed = nmi_score(cluster_labels, emotion_labels)
    rng = np.random.default_rng(seed)
    null = np.empty(repeats, dtype=float)

    for i in range(repeats):
        shuffled = rng.permutation(emotion_labels)
        null[i] = nmi_score(cluster_labels, shuffled)

    p = (1 + np.sum(null >= observed)) / (repeats + 1)
    return observed, null, float(p)


nmi_observed, nmi_null, nmi_p = permutation_test_nmi(
    kmeans_labels,
    emotion,
    repeats=N_PERMUTATIONS,
    seed=SEED,
)

print("\nCluster ↔ emotion")
print("observed NMI :", nmi_observed)
print("null mean    :", nmi_null.mean())
print("null 95%     :", np.quantile(nmi_null, 0.95))
print("p-value      :", nmi_p)

cluster_emotion_table = pd.crosstab(
    pd.Series(kmeans_labels, name="cluster"),
    pd.Series(emotion, name="emotion"),
)
print("\ncluster × emotion")
print(cluster_emotion_table.to_string())


def emotion_distance_test(X, emotion_labels, repeats=2000, seed=42):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(emotion_labels)
    D = pairwise_euclidean(X)
    upper = np.triu_indices(len(X), 1)
    pair_distance = D[upper]
    same = labels[upper[0]] == labels[upper[1]]

    if not same.any() or same.all():
        raise ValueError("same/different emotion pair comparison is not possible.")

    same_mean = float(pair_distance[same].mean())
    different_mean = float(pair_distance[~same].mean())
    # 양수일수록 같은 감정 sample끼리 더 가깝다.
    observed_gap = different_mean - same_mean

    rng = np.random.default_rng(seed)
    null = np.empty(repeats, dtype=float)

    for i in range(repeats):
        shuffled = rng.permutation(labels)
        shuffled_same = shuffled[upper[0]] == shuffled[upper[1]]
        null[i] = (
            pair_distance[~shuffled_same].mean()
            - pair_distance[shuffled_same].mean()
        )

    p = (1 + np.sum(null >= observed_gap)) / (repeats + 1)
    return same_mean, different_mean, observed_gap, null, float(p)


same_dist, diff_dist, distance_gap, distance_null, distance_p = emotion_distance_test(
    X_pca,
    emotion,
    repeats=N_PERMUTATIONS,
    seed=SEED,
)

print("\nSame emotion vs different emotion distance")
print("same-emotion mean distance     :", same_dist)
print("different-emotion mean distance:", diff_dist)
print("distance gap (different - same):", distance_gap)
print("null 95%                       :", np.quantile(distance_null, 0.95))
print("p-value                        :", distance_p)

assignments = pd.DataFrame({
    "sample_id": blind_features["sample_id"],
    "cluster_kmeans": kmeans_labels,
    "cluster_hierarchical": hier_labels,
    "emotion_label_after_unblinding": emotion,
})

summary = pd.DataFrame([
    {"metric": "selected_k_blind", "value": K_STAR, "p_value": np.nan},
    {"metric": "kmeans_silhouette", "value": observed_silhouette, "p_value": control_p},
    {"metric": "kmeans_vs_hierarchical_ARI", "value": method_ari, "p_value": np.nan},
    {"metric": "seed_stability_median_ARI", "value": np.median(seed_ari), "p_value": np.nan},
    {"metric": "noise_stability_median_ARI", "value": np.median(noise_ari), "p_value": np.nan},
    {"metric": "cluster_emotion_NMI", "value": nmi_observed, "p_value": nmi_p},
    {"metric": "emotion_distance_gap", "value": distance_gap, "p_value": distance_p},
])

assignments.to_csv(
    RESULTS_DIR / "cluster_assignments.csv",
    index=False,
    encoding="utf-8-sig",
)
summary.to_csv(
    RESULTS_DIR / "validation_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nSaved results")
print("-", RESULTS_DIR / "blind_cluster_comparison.csv")
print("-", RESULTS_DIR / "cluster_stability.csv")
print("-", RESULTS_DIR / "cluster_assignments.csv")
print("-", RESULTS_DIR / "validation_summary.csv")

summary
