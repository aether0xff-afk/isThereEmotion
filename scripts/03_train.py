from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


META_COLUMNS = {
    "sample_id",
    "label",
    "talk_id",
    "persona_id",
    "profile_id",
    "source_file",
}


def choose_split(df: pd.DataFrame, y: pd.Series, seed: int):
    group_col = None
    for candidate in ("talk_id", "profile_id", "persona_id"):
        if candidate in df.columns and df[candidate].notna().sum() == len(df):
            if df[candidate].nunique() < len(df):
                group_col = candidate
                break

    min_class_count = int(y.value_counts().min())
    if group_col is not None and min_class_count >= 2:
        n_splits = min(5, min_class_count)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        groups = df[group_col].astype(str)
        train_idx, test_idx = next(splitter.split(df, y, groups))
        return train_idx, test_idx, f"StratifiedGroupKFold(group={group_col}, n_splits={n_splits})"

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.25, random_state=seed
    )
    train_idx, test_idx = next(splitter.split(df, y))
    return train_idx, test_idx, "StratifiedShuffleSplit(test_size=0.25)"


def evaluate(name, model, x_train, x_test, y_train, y_test, out_dir: Path):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
    }

    labels = sorted(set(y_test) | set(pred))
    cm = confusion_matrix(y_test, pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        out_dir / f"confusion_{name}.csv", encoding="utf-8-sig"
    )

    report = classification_report(
        y_test, pred, labels=labels, output_dict=True, zero_division=0
    )
    with (out_dir / f"classification_report_{name}.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return metrics


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
        default=Path("results"),
    )
    parser.add_argument(
        "--min-class-size",
        type=int,
        default=2,
        help="Classes smaller than this are excluded before splitting.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    if "label" not in df:
        raise SystemExit("features.csv has no 'label' column.")

    counts = df["label"].value_counts()
    keep_labels = counts[counts >= args.min_class_size].index
    df = df[df["label"].isin(keep_labels)].copy()

    if df["label"].nunique() < 2:
        raise SystemExit(
            "Need at least two labels after filtering. "
            "Inspect label counts and adjust --min-class-size."
        )

    y = df["label"].astype(str)
    feature_columns = [
        col
        for col in df.columns
        if col not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]
    x = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    train_idx, test_idx, split_name = choose_split(df, y, args.seed)
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=args.seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=-1,
        ),
    }

    rows = []
    for name, model in models.items():
        rows.append(
            evaluate(
                name,
                model,
                x_train,
                x_test,
                y_train,
                y_test,
                args.output_dir,
            )
        )

    metrics = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    split_info = {
        "split": split_name,
        "seed": args.seed,
        "n_total_after_filter": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "labels": sorted(y.unique().tolist()),
        "feature_count": len(feature_columns),
        "excluded_from_X": sorted(META_COLUMNS),
    }
    with (args.output_dir / "split_info.json").open("w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    print(split_name)
    print(f"samples={len(df)}, features={len(feature_columns)}, labels={y.nunique()}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
