from __future__ import annotations
import json
import re
from typing import Dict, List, Optional
import pandas as pd
from joblib import load
from .ddl_parser import load_ddl
from .data_loader import load_tables
from .featurizer import column_features
from rapidfuzz import fuzz  # pip install rapidfuzz

def normalize_name(name: str) -> str:
    """Normalize table/column names for consistent comparison."""
    name = name.lower()
    name = re.sub(r"[\s\-\.\:]+", "_", name)
    name = re.sub(r"__+", "_", name)
    return name.strip("_")

def predict_mappings(
    ddl_path: str,
    source_root: str,
    source_files: Dict[str, str],
    target_root: str,
    target_files: Dict[str, str],
    model_in: str,
    out_csv: str,
    out_json: str,
    source_label: str = "Guidewire",
    target_label: str = "InsureNow",
    table_pairs: Optional[List[List[str]]] = None,
    top_n_per_source_col: Optional[int] = None,
    **kwargs  # safely accept any extra args (threshold, top_k, synonyms_path, etc.)
):
    """
    Predict all possible mappings between source and target tables/columns.
    Combines ML score and fuzzy name similarity into a single combined_match_score.
    Can optionally keep only top-N matches per source column.
    """

    # Load model
    mdl = load(model_in)
    model = mdl["model"]
    feat_cols = mdl["features"]

    # Load schema
    ddl = load_ddl(ddl_path)
    src = load_tables(source_root, source_files)
    tgt = load_tables(target_root, target_files)

    rows = []

    # Determine table pairs
    if table_pairs is None:
        table_pairs = [[s_table, t_table] for s_table in src.keys() for t_table in tgt.keys()]

    for s_table_raw, t_table_raw in table_pairs:
        if s_table_raw not in src or t_table_raw not in tgt:
            continue

        s_table_norm = normalize_name(s_table_raw)
        t_table_norm = normalize_name(t_table_raw)

        s_df = src[s_table_raw]
        t_df = tgt[t_table_raw]

        for s_col_raw in s_df.columns:
            s_col_norm = normalize_name(s_col_raw)
            s_series = s_df[s_col_raw].astype(str)

            for t_col_raw in t_df.columns:
                t_col_norm = normalize_name(t_col_raw)

                # Fuzzy similarity
                fuzzy_score = fuzz.token_sort_ratio(s_col_raw, t_col_raw) / 100.0

                # ML features
                try:
                    feats = column_features(
                        s_table_norm,
                        s_col_norm,
                        s_series,
                        "string",
                        t_table_norm,
                        t_col_norm,
                        t_df[t_col_raw].astype(str),
                        ddl.get(t_table_raw, {}).get(t_col_raw, "string")
                    )
                except Exception:
                    continue

                X = [[feats[c] for c in feat_cols]]

                try:
                    if hasattr(model, "predict_proba"):
                        ml_score = float(model.predict_proba(X)[0, 1])
                    else:
                        ml_score = float(model.decision_function(X)[0])
                except Exception:
                    ml_score = 0.0

                combined_match_score = 0.6 * ml_score + 0.4 * fuzzy_score

                rows.append({
                    "source_system": source_label,
                    "source_table_raw": s_table_raw,
                    "source_table_norm": s_table_norm,
                    "source_column_raw": s_col_raw,
                    "source_column_norm": s_col_norm,
                    "target_system": target_label,
                    "target_table_raw": t_table_raw,
                    "target_table_norm": t_table_norm,
                    "target_column_raw": t_col_raw,
                    "target_column_norm": t_col_norm,
                    "ml_score": round(ml_score, 4),
                    "fuzzy_name_score": round(fuzzy_score, 4),
                    "combined_match_score": round(combined_match_score, 4)
                })

    df = pd.DataFrame(rows)

    # Optionally keep only top-N matches per source column
    if top_n_per_source_col and not df.empty:
        df = df.sort_values(
            by=["source_table_norm", "source_column_norm", "combined_match_score"],
            ascending=[True, True, False]
        )
        df = df.groupby(["source_table_raw", "source_column_raw"], group_keys=False).head(top_n_per_source_col)

    # Sort final output
    if not df.empty:
        df = df.sort_values(
            by=["source_table_norm", "target_table_norm", "combined_match_score"],
            ascending=[True, True, False]
        )

    # Save CSV + JSON
    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    return {"rows": len(df), "csv": out_csv, "json": out_json}
