"""Optional step: add ChEMBL potency (IC50/EC50/ED50/Ki/Kd) to the unified table, via the ChEMBL web API.

Run from src/ on a machine with internet:  python admet/add_chembl.py --limit 3      (try 3 drugs first)
Uses the ChEMBL ID that DrugBank lists for each drug (db_chembl_id; run build_inputs.py --rebuild_cache once to get it).
Only the ChEMBL ID is sent. Responses are summarised and cached on disk per molecule (resumable, cheap to re-run).

Two views are kept apart, and cytotoxicity is excluded from both:
- antiviral activity: the assay's target organism is a virus, or the assay description is about antiviral activity;
  Nipah / henipavirus / Hendra results are listed first and flagged (chembl_nipah_hit);
- target potency: binding or functional assays against protein targets (not necessarily the drug's mechanism target).
A doxorubicin IC50 in a cancer cell line is cytotoxicity, not potency, so cell-line assays are not reported.
Only exact ('=') values in nM are used; values with a data-validity comment are dropped.
"""
import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table import load_table, merge_columns, save_table  # noqa: E402
from web import JsonCache, get_json  # noqa: E402

HOST = "https://www.ebi.ac.uk"
BASE = HOST + "/chembl/api/data"
SCHEMA_VERSION = 1
TYPES = ["IC50", "EC50", "ED50", "Ki", "Kd"]
PAGE = 1000

VIRUS_ORG = re.compile(r"virus|viridae|\bHIV\b|retrovir", re.I)
ANTIVIRAL_DESC = re.compile(r"antivir|virus|viral|replicon|plaque|cytopathic", re.I)
CYTOTOX_DESC = re.compile(r"cytotox|cell viability|toxicity|cytostatic|antiproliferat", re.I)
NIPAH = re.compile(r"nipah|henipa|hendra", re.I)

CHEMBL_COLS = ["chembl_potency_summary", "chembl_n_activities", "chembl_n_antiviral", "chembl_n_target",
               "chembl_best_nM_antiviral", "chembl_best_nM_target", "chembl_nipah_hit", "chembl_truncated",
               "chembl_release", "chembl_error"]


def fetch_activities(get, chembl_id, max_activities):
    """Activity rows (already filtered server-side) and whether the list was truncated."""
    params = {"molecule_chembl_id": chembl_id, "standard_type__in": ",".join(TYPES), "standard_relation": "=",
              "standard_units": "nM", "data_validity_comment__isnull": "true", "limit": PAGE}
    r = get(BASE + "/activity.json", params)
    acts = list(r.get("activities", []))
    while r.get("page_meta", {}).get("next") and len(acts) < max_activities:
        r = get(urljoin(HOST, r["page_meta"]["next"]), None)
        acts += r.get("activities", [])
    total = int(r.get("page_meta", {}).get("total_count", len(acts)))
    return acts[:max_activities], total > max_activities


def fetch_targets(get, ids):
    out = {}
    ids = sorted({i for i in ids if i})
    for i in range(0, len(ids), 50):
        r = get(BASE + "/target.json", {"target_chembl_id__in": ",".join(ids[i:i + 50]), "limit": PAGE})
        for t in r.get("targets", []):
            out[t["target_chembl_id"]] = {"type": t.get("target_type"), "organism": t.get("organism"),
                                          "name": t.get("pref_name")}
    return out


def _nm(v) -> str:
    return f"{v:.3g}"


def _group_text(df, label_col, top=3, first=None):
    """'label: TYPE best nM (best of n=.., median nM; assay)' for the top groups, ordered by best potency."""
    rows = []
    for label, g in df.groupby(label_col):
        b = g.loc[g["value"].idxmin()]
        rows.append((first is not None and bool(first(g)), b["value"], label, len(g), g["value"].median(),
                     b["standard_type"], b["assay_chembl_id"]))
    rows.sort(key=lambda t: (not t[0], t[1]))
    return "; ".join(f"{lab}: {typ} {_nm(best)} nM (best of n={n}, median {_nm(med)} nM; {assay})"
                     for _, best, lab, n, med, typ, assay in rows[:top])


def summarize(acts, targets) -> dict:
    """Turn raw ChEMBL activity rows into the chembl_* fields."""
    out = {c: None for c in CHEMBL_COLS}
    out.update(chembl_n_activities=0, chembl_n_antiviral=0, chembl_n_target=0, chembl_nipah_hit=False)
    if not acts:
        return out
    df = pd.DataFrame(acts)
    for c in ("standard_type", "standard_value", "standard_units", "standard_relation", "assay_chembl_id", "assay_type",
              "assay_description", "target_chembl_id", "target_organism", "target_pref_name"):
        if c not in df:
            df[c] = None
    df["value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df[(df["standard_relation"] == "=") & (df["standard_units"] == "nM") & (df["value"] > 0)].copy()
    out["chembl_n_activities"] = len(df)
    if df.empty:
        return out
    tmap = df["target_chembl_id"].map(lambda x: targets.get(x, {}))
    df["t_type"] = tmap.map(lambda d: (d.get("type") or "").upper())
    df["t_org"] = df["target_organism"].where(df["target_organism"].notna(), tmap.map(lambda d: d.get("organism")))
    df["t_org"] = df["t_org"].fillna("")
    df["t_name"] = df["target_pref_name"].where(df["target_pref_name"].notna(), tmap.map(lambda d: d.get("name")))
    df["t_name"] = df["t_name"].fillna(df["target_chembl_id"]).fillna("unknown target")
    desc = df["assay_description"].fillna("")

    cyto = desc.str.contains(CYTOTOX_DESC) & ~desc.str.contains("antivir", case=False)
    org_virus = df["t_org"].str.contains(VIRUS_ORG)
    anti = (org_virus | desc.str.contains(ANTIVIRAL_DESC)) & ~cyto
    nip = (df["t_org"] + " " + desc).str.contains(NIPAH)
    is_protein = df["t_type"].map(lambda t: t == "SINGLE PROTEIN" or t.startswith("PROTEIN"))
    tgt = ~anti & ~cyto & is_protein & df["assay_type"].isin(["B", "F"])

    a, t = df[anti].copy(), df[tgt].copy()
    a["label"] = np.where(a["t_org"].str.contains(VIRUS_ORG), a["t_org"], "virus (cell-based assay)")
    a["nipah"] = nip[anti].values
    out.update(chembl_n_antiviral=len(a), chembl_n_target=len(t), chembl_nipah_hit=bool(a["nipah"].any()))
    parts = []
    if len(a):
        out["chembl_best_nM_antiviral"] = float(a["value"].min())
        parts.append("Antiviral - " + _group_text(a, "label", first=lambda g: g["nipah"].any()))
    if len(t):
        out["chembl_best_nM_target"] = float(t["value"].min())
        parts.append("Targets - " + _group_text(t, "t_name"))
    out["chembl_potency_summary"] = " | ".join(parts) if parts else None
    return out


def fetch_for_molecule(get, chembl_id, max_activities) -> dict:
    acts, truncated = fetch_activities(get, chembl_id, max_activities)
    targets = fetch_targets(get, [a.get("target_chembl_id") for a in acts])
    res = summarize(acts, targets)
    res["chembl_truncated"] = bool(truncated)
    return res


def run(table: pd.DataFrame, get, *, cache: JsonCache, max_activities=5000, limit=None, refresh=False,
        max_failures=5, log=print) -> pd.DataFrame:
    try:
        release = get(BASE + "/status.json", None).get("chembl_db_version")
    except Exception as e:  # noqa: BLE001 - the release label is nice to have, not essential
        log(f"could not read the ChEMBL release ({e}); continuing")
        release = None
    rows = table[table["in_drugbank"].astype(bool)]
    rows = rows[rows["db_chembl_id"].map(lambda x: isinstance(x, str) and bool(x.strip()))] \
        if "db_chembl_id" in rows else rows.iloc[0:0]
    if limit:
        rows = rows.head(limit)
    results, live, failed, consecutive = {}, 0, 0, 0
    for i, (_, r) in enumerate(rows.iterrows(), 1):
        cid = r["db_chembl_id"].strip()
        key = f"v{SCHEMA_VERSION}_{cid}"
        res = None if refresh else cache.get(key)
        if res is None:
            try:
                res = fetch_for_molecule(get, cid, max_activities)
                res["chembl_release"] = release
                cache.put(key, res)
                live += 1
                consecutive = 0
            except Exception as e:  # noqa: BLE001 - recorded per drug, not cached, retried on the next run
                res = {**summarize([], {}), "chembl_error": f"{type(e).__name__}: {e}"[:300]}
                failed += 1
                consecutive += 1
        results[r["drugbank_id"]] = res
        if consecutive >= max_failures:
            log(f"stopping: {consecutive} consecutive failures (is the ChEMBL web service up?). Progress is cached; "
                "re-run later to continue.")
            break
        if i % 25 == 0:
            log(f"  {i}/{len(rows)} molecules ({live} fetched live, {failed} failed)")
    new = pd.DataFrame.from_dict(results, orient="index").reindex(columns=CHEMBL_COLS)
    new["chembl_fetched_at"] = time.strftime("%Y-%m-%d")
    out = merge_columns(table, new)
    for c in ("chembl_nipah_hit", "chembl_truncated"):
        out[c] = out[c].map(lambda v: bool(v) if pd.notna(v) else False)
    for c in ("chembl_n_activities", "chembl_n_antiviral", "chembl_n_target"):
        out[c] = out[c].fillna(0).astype(int)
    return out


def summary(table: pd.DataFrame) -> str:
    k = table["in_drugbank"].astype(bool)
    has_id = table["db_chembl_id"].map(lambda x: isinstance(x, str) and bool(x.strip())) if "db_chembl_id" in table else k & False
    return (f"{int(k.sum())} drugs | with ChEMBL ID: {int(has_id.sum())} | with qualifying activities: "
            f"{int((table['chembl_n_activities'] > 0).sum())} | antiviral results: {int((table['chembl_n_antiviral'] > 0).sum())} "
            f"| Nipah/henipavirus hits: {int(table['chembl_nipah_hit'].sum())} | target potency: "
            f"{int((table['chembl_n_target'] > 0).sum())} | failed: {int(table['chembl_error'].notna().sum())}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default="../data/DB/admet/drug_table.pkl")
    p.add_argument("--cache_dir", default="../data/DB/admet/cache/chembl")
    p.add_argument("--sleep", type=float, default=0.2, help="pause after each live request, in seconds")
    p.add_argument("--retries", type=int, default=3, help="retries per request on 429/5xx/network errors")
    p.add_argument("--max_failures", type=int, default=5, help="stop after this many consecutive failed molecules")
    p.add_argument("--max_activities", type=int, default=5000, help="cap per molecule (chembl_truncated records it)")
    p.add_argument("--limit", type=int, default=None, help="only the first N drugs with a ChEMBL ID (test run)")
    p.add_argument("--refresh", action="store_true", help="ignore the cache")
    return p.parse_args(argv)


def main(argv=None, get=None):
    a = parse_args(argv)
    table = load_table(a.table)
    if "db_chembl_id" not in table.columns:
        raise SystemExit("no db_chembl_id column: run build_inputs.py --rebuild_cache first")
    if get is None:
        get = lambda url, params, **kw: get_json(url, params, sleep=a.sleep, retries=a.retries, **kw)  # noqa: E731
    out = run(table, get, cache=JsonCache(a.cache_dir), max_activities=a.max_activities, limit=a.limit,
              refresh=a.refresh, max_failures=a.max_failures)
    if a.limit:
        print("NOTE: --limit was used; the output table only has chembl_* data for those drugs")
    save_table(out, Path(a.table).parent)
    print(summary(out))


if __name__ == "__main__":
    main()
