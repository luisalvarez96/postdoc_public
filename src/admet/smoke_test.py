"""Synthetic-data smoke test for the ADMET pipeline (no DrugBank data, no LLM server needed).

Run from src/:  python admet/smoke_test.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import annotate as ann_mod  # noqa: E402
import build_inputs as bi  # noqa: E402
import report as rep  # noqa: E402
import table as tbl  # noqa: E402
from features import FLAG_NAMES, route_tokens  # noqa: E402
from schemas import NOT_REPORTED, DrugAnnotation  # noqa: E402
from scoring import load_points, rescale_0_100, score_row  # noqa: E402
from verify import numbers_supported, quote_supported  # noqa: E402

COLS = ["description", "indication", "pharmacodynamics", "toxicity", "mechanism-of-action", "metabolism", "absorption",
        "half_life", "protein_binding", "clearance", "volume_of_distribution", "route_of_elimination",
        "affected-organisms", "targets", "categories", "groups", "synonyms", "atc"]


def drug(i, name, typ, atc="", cats="", routes=(), tox="", syn="", mech="Binds target X."):
    row = {c: "" for c in COLS}
    row.update({"drugbank_id": f"DB{i:05d}", "name": name, "type": typ, "atc": atc, "categories": cats,
                "toxicity": tox, "synonyms": syn, "mechanism-of-action": mech, "indication": "Used for testing.",
                "n_dosages": len(routes), "dosage_routes": list(routes), "n_products": len(routes),
                "product_routes": list(routes)})
    return row


DB = pd.DataFrame([
    drug(1, "Examplovir", "small molecule", "J05AX99", "Antiviral Agents || Hepatotoxic agents", ["Oral"],
         "The oral LD50 in rats is 1500 mg/kg. Elevated liver enzymes were reported in some patients."),
    drug(2, "Fakemab", "biotech", "L01FE01", "Antibodies || Antineoplastic Agents",
         ["Intravenous", "Intravenous; Parenteral"], "The intravenous LD<sub>50</sub> is > 300 mg/kg in mice."),
    drug(3, "Testvax Vaccine", "biotech", "J07BB01", "Vaccines", ["Intramuscular"]),
    drug(4, "Contrastol", "small molecule", "V08AA01", "Contrast Media", ["Intravenous"]),
    drug(5, "Creamex", "small molecule", "D01AC01", "Antifungal Agents", ["Topical", "Cutaneous"]),
    drug(6, "Aliasdrug", "small molecule", "", "", ["Respiratory (Inhalation)"], "Severe cardiotoxicity has been reported."),
    drug(7, "Synonymdrug", "small molecule", "", "", [], syn="Syn Alpha|Syn Beta"),
])
PRI = pd.DataFrame({
    "drug_name": ["Examplovir", "Fakemab", "Testvax Vaccine", "Contrastol", "Creamex", "CODE-1", "Syn Alpha", "Ghost"],
    "endos": [80, 85, 90, 70, 60, 75, 65, np.nan, ], "iAstro": [80, 85, 90, 70, 60, 75, 65, np.nan],
    "iNeuron": [80, 85, 90, 70, 60, 75, 65, 50], "AT": [80, 85, 90, 70, 60, 75, 65, 50],
})

# ---------------------------------------------------------------- verify
assert quote_supported("Elevated liver enzymes", "Elevated liver enzymes were reported in some patients.")
assert quote_supported("intravenous LD50 is > 300 mg/kg", "The intravenous LD<sub>50</sub> is > 300 mg/kg in mice.")
assert not quote_supported("causes severe kidney failure in children", "Elevated liver enzymes were reported.")
assert not quote_supported("", "anything")
assert numbers_supported("LD50 1500 mg/kg", "oral LD50 in rats is 1,500 mg/kg")[0]
assert not numbers_supported("IC50 5 nM", "no potency data")[0]
assert route_tokens(["Intravenous; Parenteral", "Oral"]) == {"intravenous", "parenteral", "oral"}
print("verify ok")

# ---------------------------------------------------------------- build_inputs
tmp = Path(tempfile.mkdtemp())
alias_csv = tmp / "aliases.csv"
alias_csv.write_text("drug_name,drugbank_name,verified,note\nCODE-1,Aliasdrug,yes,\nGhost2,Examplovir,no,\n")
table = bi.build_table(DB, PRI, bi.load_aliases(alias_csv))
# optional columns that later steps (FDA labels, ChEMBL) would add to the same table
table["fda_boxed_warning"] = np.where(table["drug_name"] == "Syn Alpha", "WARNING: Fatal hepatotoxicity has occurred.", None)
table["fda_has_boxed_warning"] = table["fda_boxed_warning"].notna()
table["chembl_potency_summary"] = np.where(table["drug_name"] == "Fakemab", "IC50 5 nM vs EGFR (n=3, CHEMBL1234)", None)
records = tbl.records_from_table(table)
feats = table
f = table.set_index("drug_name")
assert f.loc["CODE-1", "match_type"] == "alias" and f.loc["Syn Alpha", "match_type"] == "synonym"
assert f.loc["Ghost", "match_type"] == "unmatched" and not f.loc["Ghost", "in_drugbank"]
assert f.loc["Ghost", "ModelAvg4"] == 50 and f.loc["Ghost", "n_cond"] == 2      # mean of the non-NaN conditions
assert f.loc["Examplovir", "oral_or_inhalation"] and f.loc["Examplovir", "tox_tag_present"]
assert f.loc["Fakemab", "antibody"] and f.loc["Fakemab", "chemotherapy"] and f.loc["Fakemab", "iv_only"]
assert not f.loc["Fakemab", "other_biotech"] and not f.loc["Fakemab", "vaccine"]
assert f.loc["Testvax Vaccine", "vaccine"] and not f.loc["Testvax Vaccine", "other_biotech"]
assert f.loc["Contrastol", "diagnostic_agent"] and f.loc["Contrastol", "iv_only"]
assert f.loc["Creamex", "topical_only"] and not f.loc["Creamex", "tox_text_present"]
assert f.loc["CODE-1", "oral_or_inhalation"]                                      # inhalation route
assert f.loc["Syn Alpha", "no_route_info"] and not f.loc["Syn Alpha", "iv_only"] and not f.loc["Syn Alpha", "topical_only"]
assert len(records) == 7 and all(set(ann_mod.PROMPT_FIELDS) <= set(r) for r in records)
rec_by = {r["drug_name"]: r for r in records}
assert rec_by["Syn Alpha"]["tox_input_present"] and rec_by["Syn Alpha"]["fda_label"].startswith("Boxed warning: WARNING")
assert rec_by["Creamex"]["fda_label"] == "Not available" and not rec_by["Creamex"]["tox_input_present"]
assert rec_by["Examplovir"]["routes"] == "Oral" and rec_by["Syn Alpha"]["routes"] == "UNKNOWN"
assert rec_by["Examplovir"]["category_tags"] and rec_by["Examplovir"]["toxicity"].startswith("The oral LD50")
assert rec_by["Creamex"]["toxicity"] == "UNKNOWN" and all(not isinstance(v, float) for v in rec_by["Creamex"].values())
# table round trip + readable view without long text
tbl.save_table(table, tmp / "tbl")
back = tbl.load_table(tmp / "tbl" / "drug_table.pkl")
assert back.shape == table.shape and list(back["drug_name"]) == list(table["drug_name"])
light = pd.read_csv(tmp / "tbl" / "drug_table_light.tsv", sep="\t")
assert "db_toxicity" not in light.columns and "fda_boxed_warning" not in light.columns and "ModelAvg4" in light.columns
assert "fda_has_boxed_warning" in light.columns and "chembl_potency_summary" in light.columns
# matched candidates without a DrugBank entry get no record, but stay in the table
assert len(table) == 8 and int((~table["in_drugbank"]).sum()) == 1
print("build_inputs ok:", {k: int(feats[k].sum()) for k in FLAG_NAMES})

# ---------------------------------------------------------------- prompts
system, user_t = ann_mod.load_prompts(HERE / "prompts", True)
assert "$" not in system and "Examplovir" in system.replace("Examplovir", "Examplovir")
by_name = {r["drug_name"]: r for r in records}
p = ann_mod.user_prompt(user_t, by_name["Examplovir"])
assert "oral LD50 in rats" in p and "$" not in p
print("prompts ok")


# ---------------------------------------------------------------- annotate (stub LLM)
def stub(rec, seed):
    flags = {k: {"known": False, "evidence": ""} for k in ann_mod.SOFT_FLAGS}
    base = dict(drug_class="Test class", mechanism_of_action="Binds target X.", toxicity_summary="Not reported",
                potency_summary="Not reported", safety_tier="unknown", safety_evidence="",
                rna_viruses_inhibited=[], dna_viruses_inhibited=[], **flags)
    n = rec["drug_name"]
    if n == "Examplovir":
        base.update(safety_tier="moderate", safety_evidence="Elevated liver enzymes were reported in some patients.",
                    toxicity_summary="The oral LD50 in rats is 1500 mg/kg.", rna_viruses_inhibited=["Example virus"])
        base["endothelial"] = {"known": True, "evidence": "acts on endothelial cells of the moon"}   # fabricated quote
    elif n == "Fakemab":
        base.update(safety_tier="high", safety_evidence="The intravenous LD50 is > 300 mg/kg in mice.",
                    toxicity_summary="LD50 > 300 mg/kg (mice, IV).", potency_summary="IC50 5 nM")   # IC50 number invented
    elif n == "Creamex":
        base.update(safety_tier="low", safety_evidence="well tolerated")    # no toxicity input -> must be forced to unknown
    elif n == "Syn Alpha":       # no DrugBank toxicity, but an FDA boxed warning -> tier allowed, quote from the label
        base.update(safety_tier="high", safety_evidence="WARNING: Fatal hepatotoxicity has occurred.")
    elif n == "CODE-1":          # candidate name (alias of Aliasdrug) is what records carry
        base.update(safety_tier="high", safety_evidence="Severe cardiotoxicity has been reported.")
    obj = DrugAnnotation(**base)
    d, checks = ann_mod.postprocess(obj, ann_mod.user_prompt(user_t, rec), rec["tox_input_present"])
    return {"annotation": d, "checks": checks}


calls = {"Contrastol": 0}


def flaky(rec, seed):
    if rec["drug_name"] == "Contrastol":      # fails on the first attempt only
        calls["Contrastol"] += 1
        if calls["Contrastol"] == 1:
            raise TimeoutError("simulated server timeout")
    return stub(rec, seed)


ann_dir = tmp / "annotations"
logs = []
s = ann_mod.run_annotation(records, ann_dir / "run_0", flaky, run_idx=0, workers=1, retry_passes=1, log=logs.append)
assert s["n_failed"] == 0 and calls["Contrastol"] == 2, (s, calls)                  # retried and recovered
assert len((ann_dir / "run_0" / "failures.jsonl").read_text().splitlines()) == 1
s = ann_mod.run_annotation(records, ann_dir / "run_0", flaky, run_idx=0, log=logs.append)   # resume: nothing to do
assert len((ann_dir / "run_0" / "annotations.jsonl").read_text().splitlines()) == 7
s = ann_mod.run_annotation(records, ann_dir / "run_1", stub, run_idx=1, workers=2, retry_passes=0, log=logs.append)
assert s["n_failed"] == 0
print("annotate runner ok (retry, resume, parallel)")

loaded = rep.load_annotations(ann_dir)
chk = {r["drug_name"]: r for r in (v[0] for v in loaded.values())}
assert chk["Creamex"]["annotation"]["safety_tier"] == "unknown" and "tier_forced_unknown_no_input" in chk["Creamex"]["checks"]
assert chk["Fakemab"]["annotation"]["potency_summary"] == NOT_REPORTED
assert "potency_summary_numbers_removed" in chk["Fakemab"]["checks"]
assert chk["Fakemab"]["annotation"]["toxicity_summary"].startswith("LD50")            # supported numbers kept
assert not chk["Examplovir"]["annotation"]["endothelial"]["known"]                    # fabricated quote rejected
assert chk["Examplovir"]["annotation"]["safety_tier"] == "moderate"
print("output checks ok")

# ---------------------------------------------------------------- scoring + report
pts = load_points(HERE / "scoring_points.csv")
r = score_row(75.0, {**{k: False for k in FLAG_NAMES}, "oral_or_inhalation": True, "chemotherapy": True,
                     "safety_low": False, "safety_moderate": True, "safety_high": False, "safety_unknown": False}, pts)
assert r["breakdown"] == "ModelAvg(75.00)+Chemo(-6)+SafetyModerate(-3)+Oral/Inh(+3)" and r["score_before"] == 69.0, r
pts.cap = 5
r = score_row(75.0, {**{k: False for k in FLAG_NAMES}, "vaccine": True, "chemotherapy": True,
                     "safety_low": False, "safety_moderate": False, "safety_high": False, "safety_unknown": True}, pts)
assert r["score_before"] == 70.0 and r["breakdown"].endswith("Cap(+15)"), r        # penalties -20 capped at -5
pts.cap = None
assert list(rescale_0_100(pd.Series([1.0, 2.0, 3.0]))) == [0.0, 50.0, 100.0]
assert list(rescale_0_100(pd.Series([2.0, 2.0]))) == [50.0, 50.0]

df = rep.build_report(table, loaded, pts)
d = df.set_index("Drug")
assert df["Final score (0-100)"].max() == 100 and df["Final score (0-100)"].min() == 0
assert d.loc["Examplovir", "Score breakdown"] == "ModelAvg(80.00)+SafetyModerate(-3)+Oral/Inh(+3)"
assert d.loc["Examplovir", "Score before rescaling"] == 80.0
assert d.loc["Testvax Vaccine", "Score breakdown"].startswith("ModelAvg(90.00)+Vaccine(-12)")
assert d.loc["CODE-1", "safety_tier"] == "high" and d.loc["CODE-1", "Score breakdown"] == \
    "ModelAvg(75.00)+SafetyHigh(-6)+Oral/Inh(+3)"
assert d.loc["Syn Alpha", "safety_tier"] == "high" and d.loc["Syn Alpha", "Score breakdown"] == "ModelAvg(65.00)+SafetyHigh(-6)"
assert bool(d.loc["Syn Alpha", "FDA boxed warning"]) and not bool(d.loc["Examplovir", "FDA boxed warning"])
assert d.loc["Fakemab", "ED50/IC50"] == "[ChEMBL] IC50 5 nM vs EGFR (n=3, CHEMBL1234)"
assert d.loc["Examplovir", "ED50/IC50"] == NOT_REPORTED
assert d.loc["Ghost", "Class of drug"] == "Not in DrugBank" and d.loc["Ghost", "safety_tier"] == "unknown"
assert d.loc["Examplovir", "RNA virus inhibited"] == "Example virus" and d.loc["Examplovir", "n_runs"] == 2
assert list(df.columns[:12]) == ["Rank", "Drug", "Final score (0-100)", "Score before rescaling", "ModelAvg",
                                 "Score breakdown", "Class of drug", "Mechanism of action", "Toxicity (LD50)",
                                 "ED50/IC50", "Route of administration", "RNA virus inhibited"]
out = tmp / "report.tsv"
df.to_csv(out, sep="\t", index=False)
assert len(pd.read_csv(out, sep="\t")) == 8
print("report ok")
print("\nall smoke tests passed")
print(df[["Rank", "Drug", "Final score (0-100)", "Score breakdown", "safety_tier"]].to_string(index=False))
