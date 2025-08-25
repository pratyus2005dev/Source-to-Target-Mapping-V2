from __future__ import annotations
from typing import Dict, List
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump

from .ddl_parser import load_ddl
from .data_loader import load_tables
from .featurizer import column_features, to_frame
from .utils import coarse_type
from .model import build_model


def normalize_name(name: str) -> str:
    """Normalize table or column name: lowercase, strip spaces, replace spaces with underscores."""
    return name.strip().lower().replace(" ", "_")


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all column names of a DataFrame."""
    df.columns = [normalize_name(c) for c in df.columns]
    return df


def build_training_pairs(
    table_pairs: List[List[str]],
    src: Dict[str, pd.DataFrame],
    tgt: Dict[str, pd.DataFrame],
    ddl: Dict[str, Dict[str, str]],
    synonyms: Dict[str, List[str]],
    negative_ratio: int = 3
) -> pd.DataFrame:
    rows = []

    # Normalize table names in dictionaries
    src = {normalize_name(k): normalize_dataframe_columns(v) for k, v in src.items()}
    tgt = {normalize_name(k): normalize_dataframe_columns(v) for k, v in tgt.items()}
    ddl = {normalize_name(k): {normalize_name(col): typ for col, typ in v.items()} for k, v in ddl.items()}
    synonyms = {normalize_name(k): [normalize_name(col) for col in v] for k, v in synonyms.items()}

    for s_table, t_table in table_pairs:
        s_table_norm = normalize_name(s_table)
        t_table_norm = normalize_name(t_table)

        s_df = src.get(s_table_norm)
        t_df = tgt.get(t_table_norm)
        s_ddl = ddl.get(s_table_norm, {})
        t_ddl = ddl.get(t_table_norm, {})

        if s_df is None:
            print(f"Warning: Source table '{s_table}' not found. Skipping.")
            continue
        if t_df is None:
            print(f"Warning: Target table '{t_table}' not found. Skipping.")
            continue

        s_cols = list(s_df.columns)
        t_cols = list(t_df.columns)

        for s_col in s_cols:
            key = f"{s_table_norm}::{s_col}"
            pos_targets = [c.split("::")[-1] for c in synonyms.get(key, [])]

            # Positive examples
            for t_col in pos_targets:
                if t_col not in t_cols:
                    print(f"Warning: Column '{t_col}' not found in target table '{t_table}' DDL or DataFrame. Skipping.")
                    continue
                feats = column_features(
                    s_table, s_col, s_df[s_col], s_ddl.get(s_col, "string"),
                    t_table, t_col, t_df[t_col], t_ddl.get(t_col, "string")
                )
                feats.update({"label": 1, "s_table": s_table, "s_col": s_col, "t_table": t_table, "t_col": t_col})
                rows.append(feats)

            # Negative examples
            neg_candidates = [c for c in t_cols if c not in pos_targets]
            n_neg = min(len(neg_candidates), negative_ratio * max(1, len(pos_targets)))
            for t_col in neg_candidates[:n_neg]:
                feats = column_features(
                    s_table, s_col, s_df[s_col], s_ddl.get(s_col, "string"),
                    t_table, t_col, t_df[t_col], t_ddl.get(t_col, "string")
                )
                feats.update({"label": 0, "s_table": s_table, "s_col": s_col, "t_table": t_table, "t_col": t_col})
                rows.append(feats)

    if not rows:
        raise ValueError("No training pairs were created. Check table_pairs, source/target tables, and synonyms.")

    return to_frame(rows)


def train(
    ddl_path: str,
    source_root: str, source_files: Dict[str, str],
    target_root: str, target_files: Dict[str, str],
    table_pairs: List[List[str]],
    synonyms_path: str,
    negative_ratio: int = 3,
    test_size: float = 0.2,
    random_state: int = 42,
    model_out: str = "model.joblib"
) -> Dict:
    print("Loading DDL...")
    ddl = load_ddl(ddl_path)

    print("Loading source tables...")
    src = load_tables(source_root, source_files)

    print("Loading target tables...")
    tgt = load_tables(target_root, target_files)

    print(f"Loading synonyms from {synonyms_path}...")
    with open(synonyms_path, "r", encoding="utf-8") as f:
        synonyms = json.load(f)

    print("Building training dataframe...")
    df = build_training_pairs(table_pairs, src, tgt, ddl, synonyms, negative_ratio)

    feature_cols = [c for c in df.columns if c not in ("label", "s_table", "s_col", "t_table", "t_col")]
    X = df[feature_cols].values
    y = df["label"].values

    print(f"Total samples: {len(df)}, Features: {len(feature_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Training model...")
    model = build_model(random_state)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    score = model.score(X_test, y_test)
    print(f"Test accuracy: {score:.4f}")

    print(f"Saving model to {model_out}...")
    dump({"model": model, "features": feature_cols}, model_out)

    return {
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "test_score": float(score),
        "model_out": model_out
    }
