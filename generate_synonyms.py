import json
import csv
from pathlib import Path

# --- CONFIGURATION ---
source_root = "data/guidewire"
target_root = "data/insurenow"

source_files = {
    "Policy": "Guidewire_Policy.csv",
    "Customer": "Guidewire_Customer.csv",
    "Address": "Guidewire_CustomerAddress.csv",
    "Claim": "Guidewire_Claim.csv",
    "Broker": "Guidewire_Broker.csv",
    "Coverage": "Guidewire_PolicyCoverage.csv",
    "Payment": "Guidewire_Payment.csv",
    "PBroker": "Guidewire_PolicyBroker.csv"
}

target_files = {
    "Policy": "InsureNow_Contract.csv",
    "Customer": "InsureNow_Client.csv",
    "Address": "InsureNow_ClientAddress.csv",
    "Claim": "InsureNow_Incident.csv",
    "Broker": "InsureNow_Agent.csv",
    "Coverage": "InsureNow_Coverage.csv",
    "Payment": "InsureNow_Transaction.csv",
    "PBroker": "InsureNow_ContractAgent.csv"
}

# --- HELPERS ---
def normalize_col(name: str) -> str:
    return name.lower().replace("_", "").strip()

def read_csv_headers(file_path: Path):
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        return next(reader)

def generate_synonyms(source_root, target_root, source_files, target_files):
    synonyms = {}
    for table in source_files.keys():
        source_path = Path(source_root) / source_files[table]
        target_path = Path(target_root) / target_files[table]

        source_cols = read_csv_headers(source_path)
        target_cols = read_csv_headers(target_path)

        for s_col in source_cols:
            s_norm = normalize_col(s_col)
            best_match = None
            for t_col in target_cols:
                t_norm = normalize_col(t_col)
                if s_norm == t_norm or s_norm in t_norm or t_norm in s_norm:
                    best_match = t_col
                    break
            # fallback to mapping to itself if no match
            if not best_match:
                best_match = s_col

            # flatten format: Table::Column -> [Table::Column]
            synonyms[f"{table}::{s_col}"] = [f"{table}::{best_match}"]

    return synonyms

# --- SAVE ---
configs_dir = Path("configs")
configs_dir.mkdir(exist_ok=True)

synonyms_json = generate_synonyms(source_root, target_root, source_files, target_files)
with open(configs_dir / "synonyms.json", "w", encoding="utf-8") as f:
    json.dump(synonyms_json, f, indent=2)

print("✅ configs/synonyms.json generated in flattened Table::Column format!")
