"""The unified per-drug table.

One row per candidate drug, keyed by `drugbank_id` (NaN for candidates DrugBank does not know). Columns:
- identity and model score: drug_name, drugbank_id, match_type, in_drugbank, ModelAvg4, n_cond, the condition columns
- `db_*`: DrugBank structured fields and text (citations stripped, missing = empty/NaN)
- deterministic features: routes, no_route_info, tox_text_present, tox_tag_present, one bool column per flag
- `fda_*` / `chembl_*`: added later by optional steps that merge on drugbank_id

Everything a drug's score depends on is in this file. It contains DrugBank text, so it lives under ../data/DB/.
"""
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from drugbank import is_missing
from features import toxicity_info

# table column -> prompt field
DB_TEXT = {
    "db_description": "description", "db_indication": "indication", "db_mechanism_of_action": "mechanism_of_action",
    "db_pharmacodynamics": "pharmacodynamics", "db_absorption": "absorption", "db_protein_binding": "protein_binding",
    "db_volume_of_distribution": "volume_of_distribution", "db_metabolism": "metabolism", "db_half_life": "half_life",
    "db_clearance": "clearance", "db_route_of_elimination": "route_of_elimination", "db_toxicity": "toxicity",
}
DB_PIPE = {"db_targets": "targets", "db_affected_organisms": "affected_organisms"}    # '|'-separated in the table
FDA_SECTIONS = [("fda_boxed_warning", "Boxed warning"), ("fda_warnings", "Warnings"),
                ("fda_adverse_reactions", "Adverse reactions")]
NO_FDA = "Not available"

# columns left out of the readable TSV
LONG_COLS = set(DB_TEXT) | set(DB_PIPE) | {"db_categories"} | {c for c, _ in FDA_SECTIONS}


def _s(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def _cap(s: str, cap: int) -> str:
    return s if len(s) <= cap else s[:cap].rstrip() + " [truncated]"


def _txt(v, cap: int, pipe=False) -> str:
    if is_missing(v):
        return "UNKNOWN"
    s = _s(v)
    if pipe:
        s = ", ".join(x.strip() for x in s.split("|") if x.strip())
    return _cap(s, cap)


def fda_block(row, cap: int) -> str:
    """FDA label sections (if the optional fda_* columns exist and are filled) as one labelled text block."""
    parts = []
    for col, label in FDA_SECTIONS:
        v = _s(row.get(col)).strip()
        if v:
            parts.append(f"{label}: {_cap(v, cap)}")
    return "\n".join(parts) if parts else NO_FDA


def records_from_table(table: pd.DataFrame, max_field_chars: int = 8000, fda_max_chars: int = 4000) -> List[dict]:
    """LLM input records (one per drug that DrugBank knows). Capping and 'UNKNOWN' filling happen here, not in the table."""
    recs = []
    for _, r in table[table["in_drugbank"].astype(bool)].iterrows():
        fda = fda_block(r, fda_max_chars)
        tox = toxicity_info(r.get("db_toxicity"), _s(r.get("db_categories")))
        rec = {
            "drugbank_id": r["drugbank_id"], "drug_name": r["drug_name"],
            "drug_type": _txt(r.get("db_type"), 50),
            "atc": _s(r.get("db_atc")).replace("|", ", ") or "UNKNOWN",
            "category_tags": _s(r.get("db_category_tags")) or "None specified",
            "routes": _s(r.get("routes")) or "UNKNOWN",
            "fda_label": fda,
            "tox_input_present": bool(tox["tox_text_present"] or tox["tox_tag_present"] or fda != NO_FDA),
        }
        for col, field in DB_TEXT.items():
            rec[field] = _txt(r.get(col), max_field_chars)
        for col, field in DB_PIPE.items():
            rec[field] = _txt(r.get(col), max_field_chars, pipe=True)
        recs.append(rec)
    return recs


def light_view(table: pd.DataFrame) -> pd.DataFrame:
    """The table without the long text columns: readable in a spreadsheet and free of DrugBank prose."""
    return table[[c for c in table.columns if c not in LONG_COLS]]


def merge_columns(table: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Add (or overwrite) columns from `new`, whose index is drugbank_id. Rows with no entry in `new` get NaN."""
    out = table.drop(columns=[c for c in new.columns if c in table.columns])
    return out.join(new, on="drugbank_id")


def save_table(table: pd.DataFrame, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "drug_table.pkl").exists():                    # keep the previous version, one step back
        shutil.copy2(out / "drug_table.pkl", out / "drug_table.prev.pkl")
    table.to_pickle(out / "drug_table.pkl")
    light_view(table).to_csv(out / "drug_table_light.tsv", sep="\t", index=False)


def load_table(path) -> pd.DataFrame:
    return pd.read_pickle(path)
