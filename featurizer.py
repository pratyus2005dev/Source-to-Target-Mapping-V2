from __future__ import annotations
from typing import Dict, List, Union
import pandas as pd
import numpy as np
from .utils import tokens, jaccard, seq_ratio, coarse_type, string_value_jaccard, numeric_stat_similarity, date_range_overlap

def column_features(
    s_table: str, s_col: Union[str, List[str]], s_series: pd.DataFrame, s_dtype: str,
    t_table: str, t_col: Union[str, List[str]], t_series: pd.DataFrame, t_dtype: str
) -> Dict[str, float]:
    """
    s_col, t_col can be either str (single column) or List[str] (multi-column)
    s_series, t_series are pd.DataFrame when multi-column, or pd.Series if single
    """
    # If multi-column, combine into single series as strings
    if isinstance(s_col, list):
        s_series_combined = s_series[s_col].astype(str).agg(" ".join, axis=1)
        s_name = " ".join(s_col)
    else:
        s_series_combined = s_series
        s_name = s_col

    if isinstance(t_col, list):
        t_series_combined = t_series[t_col].astype(str).agg(" ".join, axis=1)
        t_name = " ".join(t_col)
    else:
        t_series_combined = t_series
        t_name = t_col

    s_type = coarse_type(s_dtype)
    t_type = coarse_type(t_dtype)

    feats = {
        "name_seq_ratio": seq_ratio(s_name, t_name),
        "name_token_jaccard": jaccard(tokens(s_name), tokens(t_name)),
        "coarse_type_match": 1.0 if s_type == t_type else 0.0,
        "null_rate_diff": abs(s_series_combined.isna().mean() - t_series_combined.isna().mean()),
        "strlen_med_diff": 0.0,
        "string_val_jaccard": 0.0,
        "numeric_stat_sim": 0.0,
        "date_overlap": 0.0,
    }

    if s_type == "string" and t_type == "string":
        s_len = s_series_combined.dropna().astype(str).str.len()
        t_len = t_series_combined.dropna().astype(str).str.len()
        feats["strlen_med_diff"] = float(abs(s_len.median() - t_len.median()) / max(s_len.median(), t_len.median(), 1))
        feats["string_val_jaccard"] = string_value_jaccard(s_series_combined, t_series_combined)
    elif s_type == "numeric" and t_type == "numeric":
        feats["numeric_stat_sim"] = numeric_stat_similarity(s_series_combined, t_series_combined)
    elif s_type == "date" and t_type == "date":
        feats["date_overlap"] = date_range_overlap(s_series_combined, t_series_combined)

    return feats

def to_frame(feature_rows: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(feature_rows)
