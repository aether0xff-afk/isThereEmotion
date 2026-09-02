def pairwise_euclidean(X):
    X = np.asarray(X, dtype=float)
    sq = np.sum(X * X, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2 * X @ X.T, 0.0)
    return np.sqrt(d2)


def kmeans_pp_init(X, k, rng):
    n = len(X)
    centers_idx = [int(rng.integers(n))]
    closest_d2 = np.sum((X - X[centers_idx[0]]) ** 2, axis=1)

    for _ in range(1, k):
        total = closest_d2.sum()
        if total <= 1e-15:
            candidates = [i for i in range(n) if i not in centers_idx]
            idx = int(rng.choice(candidates))
        else:
            idx = int(rng.choice(n, p=closest_d2 / total))

        centers_idx.append(idx)
        d2 = np.sum((X - X[idx]) ** 2, axis=1)
        closest_d2 = np.minimum(closest_d2, d2)

    return X[centers_idx].copy()


def kmeans_once(X, k, seed=0, max_iter=300, tol=1e-8):
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    centers = kmeans_pp_init(X, k, rng)
    labels = np.full(len(X), -1, dtype=int)

    for _ in range(max_iter):
        d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1)

        new_centers = np.empty_like(centers)
        min_d2 = d2[np.arange(len(X)), new_labels]
        used = set()

        for c in range(k):
            members = X[new_labels == c]
            if len(members):
                new_centers[c] = members.mean(axis=0)
            else:
                order = np.argsort(min_d2)[::-1]
                idx = next(int(i) for i in order if int(i) not in used)
                used.add(idx)
                new_centers[c] = X[idx]

        shift = np.sqrt(np.sum((new_centers - centers) ** 2, axis=1)).max()
        centers = new_centers

        if np.array_equal(new_labels, labels) or shift < tol:
            labels = new_labels
            break
        labels = new_labels

    inertia = float(np.sum((X - centers[labels]) ** 2))
    return labels, centers, inertia


def kmeans_numpy(X, k, n_init=20, seed=42):
    master = np.random.default_rng(seed)
    best = None

    for s in master.integers(0, 2**31 - 1, size=n_init):
        result = kmeans_once(X, k, int(s))
        if best is None or result[2] < best[2]:
            best = result

    return best


def silhouette_score_numpy(X, labels, distance_matrix=None):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    D = pairwise_euclidean(X) if distance_matrix is None else distance_matrix
    clusters = np.unique(labels)

    if len(clusters) < 2 or len(clusters) >= len(labels):
        return np.nan

    scores = np.zeros(len(labels), dtype=float)

    for i in range(len(labels)):
        same = np.where(labels == labels[i])[0]
        same = same[same != i]

        if len(same) == 0:
            scores[i] = 0.0
            continue

        a = D[i, same].mean()
        b = min(D[i, labels == c].mean() for c in clusters if c != labels[i])
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0

    return float(scores.mean())


def hierarchical_average_labels_by_k(X, ks):
    X = np.asarray(X, dtype=float)
    n = len(X)
    wanted = set(ks)
    D0 = pairwise_euclidean(X)

    # 병합마다 새 cluster ID가 필요하므로 최대 2n-1개의 노드를 사용한다.
    max_nodes = 2 * n - 1
    D = np.full((max_nodes, max_nodes), np.inf, dtype=float)
    D[:n, :n] = D0
    np.fill_diagonal(D, np.inf)

    sizes = np.zeros(max_nodes, dtype=int)
    sizes[:n] = 1
    members = {i: [i] for i in range(n)}
    active = list(range(n))
    labels_by_k = {}
    next_id = n

    while len(active) > 1:
        idx = np.asarray(active, dtype=int)
        sub = D[np.ix_(idx, idx)]
        ai, bi = np.unravel_index(np.argmin(sub), sub.shape)
        a, b = int(idx[ai]), int(idx[bi])

        new = next_id
        next_id += 1
        sizes[new] = sizes[a] + sizes[b]
        members[new] = members[a] + members[b]

        # Average linkage: 두 기존 cluster의 크기로 가중한 평균 거리.
        for c in active:
            if c in (a, b):
                continue
            d = (sizes[a] * D[a, c] + sizes[b] * D[b, c]) / sizes[new]
            D[new, c] = D[c, new] = d

        active = [c for c in active if c not in (a, b)] + [new]

        if len(active) in wanted:
            labels = np.empty(n, dtype=int)
            for label, cluster_id in enumerate(active):
                labels[members[cluster_id]] = label
            labels_by_k[len(active)] = labels.copy()

        if len(active) <= min(wanted):
            break

    return labels_by_k


D = pairwise_euclidean(X_pca)

kmeans_results = {}
kmeans_rows = []

for k in K_VALUES:
    labels, centers, inertia = kmeans_numpy(X_pca, k, n_init=N_INIT, seed=SEED)
    sil = silhouette_score_numpy(X_pca, labels, D)
    kmeans_results[k] = labels
    kmeans_rows.append({
        "k": k,
        "kmeans_silhouette": sil,
        "kmeans_inertia": inertia,
        "kmeans_min_cluster_size": int(pd.Series(labels).value_counts().min()),
    })

hierarchical_results = hierarchical_average_labels_by_k(X_pca, K_VALUES)
hier_rows = []

for k in K_VALUES:
    labels = hierarchical_results[k]
    sil = silhouette_score_numpy(X_pca, labels, D)
    hier_rows.append({
        "k": k,
        "hierarchical_silhouette": sil,
        "hierarchical_min_cluster_size": int(pd.Series(labels).value_counts().min()),
    })

comparison = pd.DataFrame(kmeans_rows).merge(pd.DataFrame(hier_rows), on="k")
comparison["mean_silhouette"] = comparison[
    ["kmeans_silhouette", "hierarchical_silhouette"]
].mean(axis=1)

# 감정 label을 보지 않고 두 알고리즘 평균 silhouette가 가장 높은 K를 고정한다.
K_STAR = int(comparison.loc[comparison["mean_silhouette"].idxmax(), "k"])
comparison.to_csv(RESULTS_DIR / "blind_cluster_comparison.csv", index=False, encoding="utf-8-sig")

print("K* selected without emotion labels:", K_STAR)
print(comparison.to_string(index=False))


def adjusted_rand_index(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)

    contingency = np.zeros((ai.max() + 1, bi.max() + 1), dtype=int)
    np.add.at(contingency, (ai, bi), 1)

    def comb2(x):
        x = np.asarray(x, dtype=float)
        return x * (x - 1) / 2

    sum_ij = comb2(contingency).sum()
    sum_a = comb2(contingency.sum(axis=1)).sum()
    sum_b = comb2(contingency.sum(axis=0)).sum()
    total = comb2(len(a))
    expected = sum_a * sum_b / total if total else 0.0
    maximum = 0.5 * (sum_a + sum_b)
    denom = maximum - expected
    return float((sum_ij - expected) / denom) if abs(denom) > 1e-15 else 1.0


kmeans_labels = kmeans_results[K_STAR]
hier_labels = hierarchical_results[K_STAR]
method_ari = adjusted_rand_index(kmeans_labels, hier_labels)

cluster_sizes = pd.DataFrame({
    "kmeans": pd.Series(kmeans_labels).value_counts().sort_index(),
    "hierarchical": pd.Series(hier_labels).value_counts().sort_index(),
})

print("\nK-means ↔ Hierarchical ARI:", method_ari)
print(cluster_sizes.to_string())

# K-means 초기 중심 변화에 대한 안정성.
seed_ari = []
for s in range(N_STABILITY):
    labels_s, _, _ = kmeans_once(X_pca, K_STAR, seed=s)
    seed_ari.append(adjusted_rand_index(kmeans_labels, labels_s))

# PCA 좌표에 작은 5% 잡음을 넣어도 같은 군집이 유지되는지 확인.
rng = np.random.default_rng(SEED)
noise_ari = []
scale = X_pca.std(axis=0)
scale[scale == 0] = 1.0

for _ in range(N_STABILITY):
    X_noisy = X_pca + rng.normal(0.0, 0.05, size=X_pca.shape) * scale
    labels_noisy, _, _ = kmeans_numpy(
        X_noisy,
        K_STAR,
        n_init=10,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    noise_ari.append(adjusted_rand_index(kmeans_labels, labels_noisy))

stability = pd.DataFrame({
    "test": ["seed_change", "5pct_feature_noise"],
    "mean_ARI": [np.mean(seed_ari), np.mean(noise_ari)],
    "median_ARI": [np.median(seed_ari), np.median(noise_ari)],
    "min_ARI": [np.min(seed_ari), np.min(noise_ari)],
    "max_ARI": [np.max(seed_ari), np.max(noise_ari)],
})
stability.to_csv(RESULTS_DIR / "cluster_stability.csv", index=False, encoding="utf-8-sig")

print("\nStability")
print(stability.to_string(index=False))


def shuffled_feature_control(X, k, observed_silhouette, repeats=100, seed=42):
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    null_scores = []

    for _ in range(repeats):
        X_null = X.copy()
        # 각 PCA 축의 주변분포는 유지하면서 sample별 축 조합을 깨뜨린다.
        for j in range(X_null.shape[1]):
            rng.shuffle(X_null[:, j])

        labels_null, _, _ = kmeans_numpy(
            X_null,
            k,
            n_init=5,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        null_scores.append(silhouette_score_numpy(X_null, labels_null))

    null_scores = np.asarray(null_scores)
    p = (1 + np.sum(null_scores >= observed_silhouette)) / (len(null_scores) + 1)
    return null_scores, float(p)


observed_silhouette = silhouette_score_numpy(X_pca, kmeans_labels, D)
control_silhouette, control_p = shuffled_feature_control(
    X_pca,
    K_STAR,
    observed_silhouette,
    repeats=100,
    seed=SEED,
)

print("\nNegative control")
print("observed silhouette:", observed_silhouette)
print("control mean       :", control_silhouette.mean())
print("control max        :", control_silhouette.max())
print("empirical p-value  :", control_p)
