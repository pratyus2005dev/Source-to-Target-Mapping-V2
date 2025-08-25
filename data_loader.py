from __future__ import annotations
import os
import pandas as pd
from typing import Dict, Union, List

def load_tables(root: str, file_map: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    out = {}
    for table, fname in file_map.items():
        path = os.path.join(root, fname)
        df = pd.read_csv(path, low_memory=False)
        # Convert object columns that look like dates to datetime
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="ignore")
            except Exception:
                pass
        out[table] = df
    return out

def profile_column(series: Union[pd.Series, pd.DataFrame]) -> dict:
    """
    Profiles a single column or multiple columns (DataFrame)
    """
    if isinstance(series, pd.DataFrame):
        series = series.astype(str).agg(" ".join, axis=1)

    s = series.dropna()
    info = {
        "n": int(series.size),
        "null_ratio": float(series.isna().mean()),
        "dtype": str(series.dtype),
    }
    if s.empty:
        return {**info, "unique": 0}

    info["unique"] = int(s.nunique(dropna=True))

    if pd.api.types.is_string_dtype(series):
        lengths = s.astype(str).str.len()
        info["strlen_med"] = float(lengths.median())
        info["sample_values"] = [str(v)[:64] for v in s.sample(min(100, len(s)), random_state=0).unique()[:50]]
    elif pd.api.types.is_numeric_dtype(series):
        info["mean"] = float(s.mean())
        info["std"] = float(s.std() or 0.0)
    elif pd.api.types.is_datetime64_any_dtype(series):
        s_dt = pd.to_datetime(s, errors="coerce")
        info["min"] = str(s_dt.min())
        info["max"] = str(s_dt.max())
    return info
