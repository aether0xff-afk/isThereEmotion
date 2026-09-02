from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


META_COLUMNS = {
    "sample_id",
    "label",
    "talk_id",
    "persona_id",
    "profile_id",
    "source_file",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/eda"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df["label"].value_counts().rename_axis("label").reset_index(
        name="count"
    ).to_csv(args.output_dir / "label_counts.csv", index=False, encoding="utf-8-sig")

    numeric = df.select_dtypes(include="number")
    numeric.describe().T.to_csv(
        args.output_dir / "numeric_summary.csv", encoding="utf-8-sig"
    )

    counts = df["label"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Samples per emotion label")
    ax.set_xlabel("label")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(args.output_dir / "label_counts.png", dpi=180)
    plt.close(fig)

    core_features = [
        col
        for col in [
            "active_mean",
            "active_std",
            "active_slope_early10",
            "active_set_jaccard_mean",
            "edges_mean",
            "edges_std",
            "exc_state_ratio",
            "inh_state_ratio",
            "mod_state_ratio",
        ]
        if col in df.columns
    ]
    if core_features:
        grouped = df.groupby("label")[core_features].mean()
        grouped.to_csv(
            args.output_dir / "label_feature_means.csv", encoding="utf-8-sig"
        )

    feature_columns = [
        col
        for col in df.columns
        if col not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(df) >= 3 and len(feature_columns) >= 2:
        x = df[feature_columns].fillna(0.0)
        x_scaled = StandardScaler().fit_transform(x)
        coords = PCA(n_components=2, random_state=0).fit_transform(x_scaled)
        pca_df = pd.DataFrame(
            {
                "PC1": coords[:, 0],
                "PC2": coords[:, 1],
                "label": df["label"].astype(str).to_numpy(),
                "sample_id": df["sample_id"].astype(str).to_numpy(),
            }
        )
        pca_df.to_csv(args.output_dir / "pca.csv", index=False, encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(8, 6))
        for label, sub in pca_df.groupby("label"):
            ax.scatter(sub["PC1"], sub["PC2"], label=label, alpha=0.75)
        ax.set_title("PCA of TRACE-derived features")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if pca_df["label"].nunique() <= 15:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output_dir / "pca.png", dpi=180)
        plt.close(fig)

    print(f"EDA outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
