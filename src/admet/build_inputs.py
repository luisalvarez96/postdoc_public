"""Step 1: candidates + DrugBank -> drug_table.pkl (the unified per-drug table, see table.py).

Run from src/:  python admet/build_inputs.py
Outputs contain DrugBank text, so the default output directory is under ../data/DB/ (untracked in git).
Optional steps (FDA labels, ChEMBL) later add fda_* / chembl_* columns to the same table.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drugbank import clean_and_deduplicate_categories, clean_citations, load_drugbank  # noqa: E402
from features import FLAG_NAMES, deterministic_flags, toxicity_info  # noqa: E402
from table import DB_PIPE, DB_TEXT, save_table  # noqa: E402

CONDITIONS = ["endos", "iAstro", "iNeuron", "AT"]       # MIRs excluded (collaborators discarded them)

# DrugBank column in drugbank.parse_drugbank_xml -> table column
DB_COPY = {
    "description": "db_description", "indication": "db_indication", "mechanism-of-action": "db_mechanism_of_action",
    "pharmacodynamics": "db_pharmacodynamics", "absorption": "db_absorption", "protein_binding": "db_protein_binding",
    "volume_of_distribution": "db_volume_of_distribution", "metabolism": "db_metabolism", "half_life": "db_half_life",
    "clearance": "db_clearance", "route_of_elimination": "db_route_of_elimination", "toxicity": "db_toxicity",
    "targets": "db_targets", "affected-organisms": "db_affected_organisms",
}
assert set(DB_COPY.values()) == set(DB_TEXT) | set(DB_PIPE)
DB_IDS = {"unii": "db_unii", "chembl_id": "db_chembl_id", "pubchem_cid": "db_pubchem_cid", "cas_number": "db_cas_number"}


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _routes(row) -> list:
    out = set()
    for col in ("dosage_routes", "product_routes"):
        v = row.get(col)
        if isinstance(v, (list, tuple, np.ndarray)):
            out.update(str(x) for x in v)
    return sorted(out)


def load_aliases(path) -> dict:
    """aliases.csv: drug_name, drugbank_name, verified. Only rows with verified == yes are applied."""
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    skipped = df[df["verified"].astype(str).str.lower() != "yes"]
    if len(skipped):
        print(f"aliases: {len(skipped)} unverified rows skipped (set verified=yes to apply): "
              + ", ".join(skipped["drug_name"].astype(str)))
    ok = df[df["verified"].astype(str).str.lower() == "yes"]
    return {_norm(a): _norm(b) for a, b in zip(ok["drug_name"], ok["drugbank_name"])}


def match_candidates(pri: pd.DataFrame, db: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    """Add drugbank_id + match_type to the candidate table (alias > exact name > unique synonym)."""
    name_key = db["name"].map(_norm)
    by_name = pd.Series(db["drugbank_id"].values, index=name_key).loc[lambda s: ~s.index.duplicated()]
    syn = {}
    for dbid, s in zip(db["drugbank_id"], db["synonyms"].fillna("")):
        for x in str(s).split("|"):
            if x.strip():
                syn.setdefault(_norm(x), set()).add(dbid)

    def one(name):
        k = _norm(name)
        a = aliases.get(k)
        if a is not None and a in by_name.index:
            return by_name[a], "alias"
        if k in by_name.index:
            return by_name[k], "name"
        ids = syn.get(k, set())
        if len(ids) == 1:
            return next(iter(ids)), "synonym"
        return None, ("ambiguous synonym" if ids else "unmatched")

    res = pri["drug_name"].map(one)
    out = pri.copy()
    out["drugbank_id"] = [r[0] for r in res]
    out["match_type"] = [r[1] for r in res]
    return out


def build_table(db: pd.DataFrame, pri: pd.DataFrame, aliases: dict, conditions=CONDITIONS) -> pd.DataFrame:
    pri = pri.copy()
    cond = pri[conditions].apply(pd.to_numeric, errors="coerce")
    pri["ModelAvg4"] = cond.mean(axis=1).round(2)
    pri["n_cond"] = cond.notna().sum(axis=1)
    if "ModelAvg" in pri.columns:                       # the file's own column averages 5 conditions incl. MIRs
        pri = pri.rename(columns={"ModelAvg": "ModelAvg_file"})
    cand = match_candidates(pri, db, aliases)
    dbi = db.set_index("drugbank_id")

    rows = []
    for _, c in cand.iterrows():
        row = c.to_dict()
        known = c["drugbank_id"] is not None and c["drugbank_id"] in dbi.index
        row["in_drugbank"] = bool(known)
        if not known:
            row.update({"routes": "", "no_route_info": True, "tox_text_present": False, "tox_tag_present": False,
                        **{f: False for f in FLAG_NAMES}})
            rows.append(row)
            continue
        r = dbi.loc[c["drugbank_id"]].to_dict()
        cats = clean_citations(r.get("categories")) or ""
        routes = _routes(r)
        row.update({
            "db_name": r.get("name"), "db_type": r.get("type"), "db_groups": r.get("groups") or "", "db_atc": r.get("atc") or "",
            "db_categories": cats, "db_category_tags": clean_and_deduplicate_categories(cats),
            **{tcol: clean_citations(r.get(dcol)) for dcol, tcol in DB_COPY.items()},
            **{tcol: r.get(dcol) for dcol, tcol in DB_IDS.items()},
            "routes": "; ".join(routes), "no_route_info": not routes,
            **toxicity_info(clean_citations(r.get("toxicity")), cats),
            **deterministic_flags(c["drug_name"], r.get("type"), r.get("atc"), cats, routes),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drugbank_xml", default="<path/to/drugbank_full_database/full database.xml>")
    p.add_argument("--cache", default="../data/DB/drugbank_parsed.pkl")
    p.add_argument("--rebuild_cache", action="store_true", help="re-parse the XML (needed to get unii/chembl ids)")
    p.add_argument("--priorities", default="../PrelResults/drug_priorities.csv")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--aliases", default=str(Path(__file__).resolve().parent / "aliases.csv"))
    p.add_argument("--out_dir", default="../data/DB/admet")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    db = load_drugbank(a.drugbank_xml, a.cache, a.rebuild_cache)
    pri = pd.read_csv(a.priorities, sep="\t")
    table = build_table(db, pri, load_aliases(a.aliases), a.conditions)
    save_table(table, a.out_dir)
    table[table["match_type"].isin(["unmatched", "ambiguous synonym"])][["drug_name", "match_type"]].to_csv(
        Path(a.out_dir) / "unmatched.tsv", sep="\t", index=False)

    known = table["in_drugbank"]
    print(f"candidates: {len(table)} | matched: {int(known.sum())} | unmatched: {int((~known).sum())} (see unmatched.tsv)")
    print("match types:", table["match_type"].value_counts().to_dict())
    print("flag counts:", {f: int(table[f].sum()) for f in FLAG_NAMES})
    print("no toxicity info (tier forced to 'unknown'):",
          int((known & ~(table["tox_text_present"] | table["tox_tag_present"])).sum()))
    if "db_unii" in table:
        print("identifiers present: unii", int(table["db_unii"].notna().sum()), "| chembl",
              int(table["db_chembl_id"].notna().sum()), "| pubchem", int(table["db_pubchem_cid"].notna().sum()))
    print("wrote", Path(a.out_dir) / "drug_table.pkl", "and drug_table_light.tsv")


if __name__ == "__main__":
    main()
