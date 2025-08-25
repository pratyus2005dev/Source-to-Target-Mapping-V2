import json
import io
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yaml


st.set_page_config(page_title="Schema Mapping Dashboard", layout="wide")
st.sidebar.title("Inputs")

default_csv = "outputs/mapping_suggestions.csv"
default_json = "outputs/mapping_suggestions.json"

source_system = st.sidebar.text_input("Source System Name", value="Guidewire")
target_system = st.sidebar.text_input("Target System Name", value="InsureNow")

csv_file = st.sidebar.file_uploader("Upload mapping_suggestions.csv", type=["csv"])
json_file = st.sidebar.file_uploader("Upload mapping_suggestions.json", type=["json"])


config_path = Path("configs/config.yaml")
if not config_path.exists():
    st.error("config.yaml not found! Please create a config.yaml with system file mappings.")
    st.stop()

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

def resolve_table_path(table_name, system):
    """
    Resolves CSV path from config.yaml instead of guessing filenames.
    system: 'guidewire' or 'insurenow'
    """
    system_key = "source" if system.lower() == "guidewire" else "target"
    root = Path(config[system_key]["root"])
    files_map = config[system_key]["files"]

    if table_name not in files_map:
        return None

    file_path = root / files_map[table_name]
    return file_path if file_path.exists() else None


if csv_file is None and Path(default_csv).exists():
    df = pd.read_csv(default_csv)
elif csv_file is not None:
    df = pd.read_csv(csv_file)
else:
    st.warning("Please upload mapping_suggestions.csv or place it at outputs/mapping_suggestions.csv")
    st.stop()


if json_file is None and Path(default_json).exists():
    with open(default_json, "r", encoding="utf-8") as f:
        details = json.load(f)
elif json_file is not None:
    details = json.load(json_file)
else:
    details = None


expected_cols = {
    "source_table": ["source_table", "source_table_raw", "src_table", "sourceTable"],
    "source_column": ["source_column", "source_column_raw", "src_column", "sourceColumn"],
    "target_table": ["target_table", "target_table_raw", "tgt_table", "targetTable"],
    "predicted_target_column": ["predicted_target_column", "target_column_raw", "predictedTargetColumn", "prediction"],
    "ml_score": ["ml_score", "mlScore"],
    "fuzzy_name_score": ["fuzzy_name_score", "fuzzyScore", "fuzzy_name"],
    "combined_match_score": ["combined_match_score", "combinedScore", "score"]
}
col_map = {}
for canon, alts in expected_cols.items():
    for c in df.columns:
        if c in alts:
            col_map[c] = canon
            break
df = df.rename(columns=col_map)


required_cols = ["source_table", "source_column", "target_table", "predicted_target_column"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing expected columns in CSV: {missing}. Please check pipeline output headers.")
    st.stop()


df["source_system"] = source_system
df["target_system"] = target_system

def pick_best(row):
    scores = {
        "ML Score": row.get("ml_score", np.nan),
        "Fuzzy Score": row.get("fuzzy_name_score", np.nan),
        "Combined Score": row.get("combined_match_score", np.nan),
    }
    best_type = max(scores, key=lambda k: scores[k] if pd.notna(scores[k]) else -1)
    return pd.Series({"best_score": scores[best_type], "best_score_type": best_type})

best_scores = df.apply(pick_best, axis=1)
df = pd.concat([df, best_scores], axis=1)


st.title("Schema Mapping Dashboard")
st.caption(f"{source_system} → {target_system} | Best-score based mapping view")

cols_to_show = [
    "source_system", "source_table", "source_column",
    "target_system", "target_table", "predicted_target_column",
    "best_score", "best_score_type"
]

# ---------- Column Mapping Explorer ----------
st.subheader("Column Mapping Explorer")
search_col = st.text_input("Enter a source or target column name to find mappings:")

if search_col:
    bi_matches = df[
        df["source_column"].str.contains(search_col, case=False, na=False) |
        df["predicted_target_column"].str.contains(search_col, case=False, na=False)
    ].copy()

    bi_matches = bi_matches.sort_values("source_column")

    if not bi_matches.empty:
        st.write("🔍 Found the following matches:")
        st.dataframe(bi_matches[cols_to_show], use_container_width=True)

        selected_col = st.selectbox(
            "Select a column to view details:",
            sorted(set(bi_matches["source_column"]).union(bi_matches["predicted_target_column"]))
        )

        
        table_info = []
        if selected_col in bi_matches["source_column"].values:
            source_table = bi_matches.loc[bi_matches["source_column"] == selected_col, "source_table"].iloc[0]
            mapped_target_col = bi_matches.loc[bi_matches["source_column"] == selected_col, "predicted_target_column"].iloc[0]
            target_table = bi_matches.loc[bi_matches["source_column"] == selected_col, "target_table"].iloc[0]
            table_info.append(("Source", source_table, selected_col, "guidewire"))
            table_info.append(("Mapped Target", target_table, mapped_target_col, "insurenow"))
        elif selected_col in bi_matches["predicted_target_column"].values:
            target_table = bi_matches.loc[bi_matches["predicted_target_column"] == selected_col, "target_table"].iloc[0]
            mapped_source_col = bi_matches.loc[bi_matches["predicted_target_column"] == selected_col, "source_column"].iloc[0]
            source_table = bi_matches.loc[bi_matches["predicted_target_column"] == selected_col, "source_table"].iloc[0]
            table_info.append(("Target", target_table, selected_col, "insurenow"))
            table_info.append(("Mapped Source", source_table, mapped_source_col, "guidewire"))

        
        for label, tbl_name, col_name, system in table_info:
            tbl_path = resolve_table_path(tbl_name, system)
            if tbl_path and tbl_path.exists():
                try:
                    tbl_df = pd.read_csv(tbl_path)
                    if col_name in tbl_df.columns:
                        st.markdown(f"**{label} Column: `{col_name}` in table `{tbl_name}` ({system})**")
                        st.write(f"- Data Type: {tbl_df[col_name].dtype}")
                        st.write(f"- Total Rows: {len(tbl_df)}")
                        # Show first 10 unique non-null values as a table
                        sample_vals = tbl_df[col_name].dropna().unique()[:10]
                        st.dataframe(pd.DataFrame({col_name: sample_vals}))
                    else:
                        st.warning(f"Column '{col_name}' not found in {tbl_path.name}.")
                except Exception as e:
                    st.warning(f"Could not load {tbl_path.name}: {e}")
            else:
                st.warning(f"No CSV found for table '{tbl_name}' in {system}. Check config.yaml.")
    else:
        st.error("No matches found for your search.")



tables_source = sorted(df["source_table"].dropna().unique())
tables_target = sorted(df["target_table"].dropna().unique())

st.sidebar.markdown("---")
thresh = st.sidebar.slider(
    "Confidence threshold (matched if best_score ≥ threshold)",
    0.0, 1.0, 0.5, 0.01
)
src_filter = st.sidebar.multiselect("Filter by source table(s)", tables_source, default=tables_source)
tgt_filter = st.sidebar.multiselect("Filter by target table(s)", tables_target, default=tables_target)

df = df[df["source_table"].isin(src_filter) & df["target_table"].isin(tgt_filter)].copy()
df["is_matched"] = df["best_score"] >= thresh


total_src_cols = len(df)
matched = int(df["is_matched"].sum())
not_matched = total_src_cols - matched
match_pct = (matched / total_src_cols * 100.0) if total_src_cols else 0.0
num_source_tables = df["source_table"].nunique()
num_target_tables = df["target_table"].nunique()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Source Columns (filtered)", total_src_cols)
c2.metric("Matched (score ≥ threshold)", matched)
c3.metric("Not Matched", not_matched)
c4.metric("% Matched", f"{match_pct:.1f}%")
c5.metric("Source Tables", num_source_tables)
c6.metric("Target Tables", num_target_tables)


ch1, ch2, ch3 = st.columns([1, 1, 1])

with ch1:
    st.subheader("Matched vs Not Matched")
    fig, ax = plt.subplots()
    ax.pie([matched, not_matched], labels=["Matched", "Not Matched"], autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

with ch2:
    st.subheader("Match Rate by Source Table")
    by_src = df.groupby("source_table").agg(total=("source_column","count"), matched=("is_matched","sum")).reset_index()
    by_src["match_rate"] = np.where(by_src["total"] > 0, by_src["matched"]/by_src["total"]*100, 0)
    fig2, ax2 = plt.subplots()
    ax2.bar(by_src["source_table"], by_src["match_rate"])
    ax2.set_ylabel("Match Rate (%)")
    ax2.set_xticklabels(by_src["source_table"], rotation=45, ha="right")
    st.pyplot(fig2)

with ch3:
    st.subheader("Best Score Distribution")
    fig3, ax3 = plt.subplots()
    df["best_score"].dropna().plot(kind="hist", bins=12, ax=ax3)
    ax3.axvline(thresh, linestyle="--", color="red")
    ax3.set_xlabel("Best Score")
    st.pyplot(fig3)


st.subheader("Low-Confidence Mappings")
low_conf = df[~df["is_matched"] | df["predicted_target_column"].isna()].copy()
st.dataframe(low_conf[cols_to_show], use_container_width=True)

buf = io.BytesIO()
low_conf[cols_to_show].to_csv(buf, index=False)
st.download_button("Download Low-Confidence Review CSV", data=buf.getvalue(),
                   file_name="low_confidence_review.csv", mime="text/csv")

st.subheader("Top Matches")
top = df[df["is_matched"]].sort_values("best_score", ascending=False)
st.dataframe(top[cols_to_show], use_container_width=True)

buf2 = io.BytesIO()
top[cols_to_show].to_csv(buf2, index=False)
st.download_button("Download Matched CSV", data=buf2.getvalue(),
                   file_name="matched_columns.csv", mime="text/csv")

st.subheader("Table-to-Table Match Summary")
table_summary = df.groupby(["source_table","target_table"]).agg(
    total_columns=("source_column","count"),
    matched_columns=("is_matched","sum"),
    avg_best_score=("best_score","mean"),
    max_best_score=("best_score","max")
).reset_index()
st.dataframe(table_summary, use_container_width=True)

buf3 = io.BytesIO()
table_summary.to_csv(buf3, index=False)
st.download_button("Download Table Match Summary", data=buf3.getvalue(),
                   file_name="table_match_summary.csv", mime="text/csv")
