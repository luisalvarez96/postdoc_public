"""Offline tests for add_fda_labels.py and add_chembl.py using fake API responses (no network).

Run from src/:  python admet/test_fetchers.py
The fake responses follow the documented shape of the openFDA label and ChEMBL REST APIs; the real schemas still need
to be confirmed once with `--limit 3` on a machine with internet.
"""
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import add_chembl as ch  # noqa: E402
import add_fda_labels as fda  # noqa: E402
from web import HttpError, JsonCache  # noqa: E402

# ------------------------------------------------------------------ openFDA
def label(unii, generic=None, **sections):
    return {"openfda": {"unii": unii, "generic_name": generic or []}, "effective_time": "20230415", "set_id": "S-" + "-".join(unii),
            **sections}


FDA_A = {   # search by UNII (newest first)
    "UNII_SINGLE": [label(["UNII_SINGLE", "OTHER"], warnings=["combo warn"]),
                    label(["UNII_SINGLE"], warnings_and_cautions=["single  warn"], adverse_reactions=["single adverse"])],
    "UNII_COMBO": [label(["UNII_COMBO", "OTHER"], warnings=["combo only"])],
}
FDA_B = {   # same, restricted to labels with a boxed warning
    "UNII_SINGLE": [label(["UNII_SINGLE", "OTHER"], boxed_warning=["COMBO BOX"]),
                    label(["UNII_SINGLE"], boxed_warning=["BOXED", "SINGLE"])],
    "UNII_COMBO": [label(["UNII_COMBO", "OTHER"], boxed_warning=["COMBO ONLY BOX"])],
}
calls = []


def fake_fda(url, params, **kw):
    calls.append(params["search"])
    s, limit = params["search"], params["limit"]
    boxed = "_exists_:boxed_warning" in s
    m = re.search(r'openfda\.unii:"([^"]+)"', s)
    if m:
        key = m.group(1)
        if key == "UNII_WIDE":      # 60 labels; the first 8 are combinations, a single-ingredient one comes later
            if boxed:
                return None
            labs = [label(["UNII_WIDE", f"X{i}"]) for i in range(8)] + [label(["UNII_WIDE"], warnings=["wide warn"])]
            return {"meta": {"results": {"total": 60}}, "results": labs[:limit]}
        labs = (FDA_B if boxed else FDA_A).get(key)
    else:
        name = re.search(r'openfda\.generic_name:"([^"]+)"', s).group(1)
        labs = None if name != "namedrug" or boxed else [label(["U"], generic=["NAMEDRUG"], warnings=["name warn"])]
    if not labs:
        return None
    return {"meta": {"results": {"total": len(labs)}}, "results": labs[:limit]}


r = fda.fetch_for_drug(fake_fda, "UNII_SINGLE", "Singledrug")
assert r["fda_match"] == "unii" and r["fda_single_ingredient_found"] and r["fda_has_boxed_warning"]
assert r["fda_boxed_warning"] == "BOXED\n\nSINGLE" and r["fda_warnings"] == "single warn"     # combo label ignored
assert r["fda_adverse_reactions"] == "single adverse" and r["fda_label_date"] == "2023-04-15" and r["fda_n_boxed"] == 2

r = fda.fetch_for_drug(fake_fda, "UNII_COMBO", "Combodrug", name_fallback=False)
assert r["fda_match"] == "unii" and not r["fda_single_ingredient_found"]
assert r["fda_boxed_combination_only"] and not r["fda_has_boxed_warning"] and r["fda_boxed_warning"] is None
assert r["fda_warnings"] is None                                           # no text from a combination label

r = fda.fetch_for_drug(fake_fda, "UNII_WIDE", "Widedrug")
assert r["fda_single_ingredient_found"] and r["fda_warnings"] == "wide warn" and r["fda_n_labels"] == 60   # widened to 50

r = fda.fetch_for_drug(fake_fda, "UNII_UNKNOWN", "namedrug")                # UNII finds nothing -> name fallback
assert r["fda_match"] == "name" and r["fda_warnings"] == "name warn" and not r["fda_has_boxed_warning"]
r = fda.fetch_for_drug(fake_fda, None, "namedrug", name_fallback=False)
assert r["fda_match"] is None and r["fda_n_labels"] == 0
r = fda.fetch_for_drug(fake_fda, "UNII_COMBO", "namedrug")                  # combination via UNII, single via name -> name wins
assert r["fda_match"] == "name" and r["fda_single_ingredient_found"]
print("openFDA extraction ok")

table = pd.DataFrame({
    "drug_name": ["A", "B", "C", "D"], "drugbank_id": ["DB1", "DB2", "DB3", None], "in_drugbank": [True, True, True, False],
    "db_unii": ["UNII_SINGLE", "UNII_COMBO", None, None], "db_name": ["Singledrug", "Combodrug", "namedrug", None],
})
tmp = Path(tempfile.mkdtemp())
calls.clear()
out = fda.run(table, fake_fda, cache=JsonCache(tmp / "fda"), log=lambda *_: None)
n1 = len(calls)
assert bool(out.loc[0, "fda_has_boxed_warning"]) and not out.loc[1, "fda_has_boxed_warning"] and out.loc[2, "fda_match"] == "name"
assert out.loc[3, "fda_match"] is None or pd.isna(out.loc[3, "fda_match"])             # not in DrugBank: untouched
assert out["fda_has_boxed_warning"].dtype == bool and out["fda_n_labels"].dtype.kind == "i"
fda.run(table, fake_fda, cache=JsonCache(tmp / "fda"), log=lambda *_: None)
assert len(calls) == n1, "second run must be served from the cache"
print(fda.summary(out))

# the table feeds the prompt: FDA text counts as toxicity input and appears in the label block
import table as tbl  # noqa: E402
recs = {x["drug_name"]: x for x in tbl.records_from_table(out.assign(db_toxicity=None, db_categories=""))}
assert recs["A"]["fda_label"].startswith("Boxed warning: BOXED") and recs["A"]["tox_input_present"]
assert recs["B"]["fda_label"] == "Not available" and not recs["B"]["tox_input_present"]   # combination-only: no text used
print("FDA -> prompt ok")

# ------------------------------------------------------------------ ChEMBL
def act(i, typ, val, tid, org, desc, atype="F", aid=None, units="nM", rel="="):
    return {"activity_id": i, "standard_type": typ, "standard_value": str(val), "standard_units": units,
            "standard_relation": rel, "assay_chembl_id": aid or f"A{i}", "assay_type": atype, "assay_description": desc,
            "target_chembl_id": tid, "target_organism": org, "target_pref_name": org if tid != "T_EGFR" else "EGFR"}


ACTS = [
    act(1, "EC50", 120, "T_NIPAH", "Nipah virus", "Antiviral activity against Nipah virus"),
    act(2, "EC50", 300, "T_NIPAH", "Nipah virus", "Antiviral activity against Nipah virus"),
    act(3, "IC50", 50, "T_FLU", "Influenza A virus", "Inhibition of influenza A virus replication"),
    act(4, "IC50", 2, "T_EGFR", "Homo sapiens", "Inhibition of EGFR", atype="B"),
    act(5, "IC50", 10, "T_EGFR", "Homo sapiens", "Inhibition of EGFR", atype="B"),
    act(6, "IC50", 5, "T_CELL", "Homo sapiens", "Cytotoxicity against A549 cells"),
    act(7, "EC50", 7, "T_CELL", "Homo sapiens", "Antiviral activity against HIV-1 in MT-4 cells"),
    act(8, "IC50", 1, "T_EGFR", "Homo sapiens", "Inhibition of EGFR", atype="B", units="ug.mL-1"),
]
TARGETS = {"T_NIPAH": ("ORGANISM", "Nipah virus"), "T_FLU": ("ORGANISM", "Influenza A virus"),
           "T_EGFR": ("SINGLE PROTEIN", "Homo sapiens"), "T_CELL": ("CELL-LINE", "Homo sapiens")}
chembl_calls = []


def fake_chembl(url, params=None, **kw):
    chembl_calls.append(url)
    if url.endswith("/status.json"):
        return {"chembl_db_version": "ChEMBL_TEST"}
    if "activity.json" in url:
        if params and params["molecule_chembl_id"] == "CHEMBLBAD":
            raise HttpError("simulated failure")
        if params and params["molecule_chembl_id"] == "CHEMBLNONE":
            return {"activities": [], "page_meta": {"total_count": 0, "next": None}}
        assert params is None or (params["standard_units"] == "nM" and "IC50" in params["standard_type__in"])
        if params:   # first page
            return {"activities": ACTS[:4], "page_meta": {"total_count": len(ACTS), "next": "/chembl/api/data/activity.json?offset=4"}}
        return {"activities": ACTS[4:], "page_meta": {"total_count": len(ACTS), "next": None}}
    if "target.json" in url:
        ids = params["target_chembl_id__in"].split(",")
        return {"targets": [{"target_chembl_id": i, "target_type": TARGETS[i][0], "organism": TARGETS[i][1],
                             "pref_name": TARGETS[i][1]} for i in ids if i in TARGETS]}
    raise AssertionError(url)


res = ch.fetch_for_molecule(fake_chembl, "CHEMBL1", 5000)
assert res["chembl_n_activities"] == 7                     # the ug/mL row is dropped
assert res["chembl_n_antiviral"] == 4 and res["chembl_n_target"] == 2 and res["chembl_nipah_hit"]
assert res["chembl_best_nM_antiviral"] == 7.0 and res["chembl_best_nM_target"] == 2.0
s = res["chembl_potency_summary"]
assert s.startswith("Antiviral - Nipah virus: EC50 120 nM (best of n=2, median 210 nM; A1)"), s     # Nipah listed first
assert "virus (cell-based assay): EC50 7 nM" in s and "Influenza A virus: IC50 50 nM" in s
assert s.index("virus (cell-based assay)") < s.index("Influenza A virus")
assert "Targets - EGFR: IC50 2 nM (best of n=2, median 6 nM; A4)" in s
assert "Cytotoxicity" not in s and "A5" not in s.split("Targets")[0]
assert res["chembl_truncated"] is False
assert ch.fetch_for_molecule(fake_chembl, "CHEMBL1", 4)["chembl_truncated"] is True
assert ch.fetch_for_molecule(fake_chembl, "CHEMBLNONE", 5000)["chembl_potency_summary"] is None
print("ChEMBL summary ok")

table2 = pd.DataFrame({
    "drug_name": ["A", "B", "C", "D", "E"], "drugbank_id": ["DB1", "DB2", "DB3", "DB4", None],
    "in_drugbank": [True, True, True, True, False],
    "db_chembl_id": ["CHEMBL1", None, "CHEMBLBAD", "CHEMBLNONE", None],
})
chembl_calls.clear()
out2 = ch.run(table2, fake_chembl, cache=JsonCache(tmp / "chembl"), log=lambda *_: None)
assert out2.loc[0, "chembl_release"] == "ChEMBL_TEST" and out2.loc[0, "chembl_n_activities"] == 7
assert out2["chembl_nipah_hit"].tolist() == [True, False, False, False, False]
assert isinstance(out2.loc[2, "chembl_error"], str) and "simulated failure" in out2.loc[2, "chembl_error"]
assert out2.loc[1, "chembl_n_activities"] == 0 and pd.isna(out2.loc[1, "chembl_potency_summary"])
n_first = len(chembl_calls)
out3 = ch.run(table2, fake_chembl, cache=JsonCache(tmp / "chembl"), log=lambda *_: None)
new_calls = [u for u in chembl_calls[n_first:] if "status" not in u]
assert all("activity.json" in u for u in new_calls) and len(new_calls) == 1, new_calls   # only the failed molecule is retried
print(ch.summary(out2))

# the report picks up the ChEMBL summary
import report as rep  # noqa: E402
from scoring import load_points  # noqa: E402
from features import FLAG_NAMES  # noqa: E402
t = out2.assign(ModelAvg4=[70.0] * 5, routes="Oral", db_type="small molecule", **{f: False for f in FLAG_NAMES})
df = rep.build_report(t, {}, load_points(HERE / "scoring_points.csv"))
assert df.set_index("Drug").loc["A", "ED50/IC50"].startswith("[ChEMBL] Antiviral - Nipah virus")
# circuit breaker: a dead service must not make a run crawl through every drug
attempts = []


def dead(url, params=None, **kw):
    attempts.append(url)
    raise HttpError("HTTP 500")


many = pd.DataFrame({"drug_name": list("ABCDEFGH"), "drugbank_id": [f"DB{i}" for i in range(8)], "in_drugbank": True,
                     "db_chembl_id": [f"CHEMBL{i}" for i in range(8)], "db_unii": [f"U{i}" for i in range(8)],
                     "db_name": list("abcdefgh")})
attempts.clear()
o = ch.run(many, dead, cache=JsonCache(tmp / "dead_chembl"), max_failures=3, log=lambda *_: None)
assert len([u for u in attempts if "activity" in u]) == 3 and o["chembl_error"].notna().sum() == 3, attempts
attempts.clear()
o = fda.run(many, dead, cache=JsonCache(tmp / "dead_fda"), max_failures=3, log=lambda *_: None)
assert len(attempts) == 3 and o["fda_error"].notna().sum() == 3
print("circuit breaker ok")
print("\nall fetcher tests passed")
