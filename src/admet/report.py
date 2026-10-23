"""Step 3: unified table + annotations (+ scoring_points.csv) -> ranked report with a transparent score breakdown.

Run from src/:  python admet/report.py
Several runs (annotate.py --runs N) are combined: majority vote for the safety tier and the soft flags.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from features import FLAG_NAMES  # noqa: E402
from schemas import NOT_REPORTED  # noqa: E402
from scoring import load_points, rescale_0_100, safety_flags, score_row  # noqa: E402
from table import load_table  # noqa: E402

SOFT = ("endothelial", "neuronal", "respiratory")
TIER_PRIORITY = ["high", "moderate", "low", "unknown"]     # tie-break: the more severe tier wins


def load_annotations(ann_dir) -> dict:
    """drugbank_id -> list of annotation records (one per run)."""
    out = {}
    for f in sorted(Path(ann_dir).glob("run_*/annotations.jsonl")):
        last = {}
        for line in f.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                last[rec["drugbank_id"]] = rec              # a drug re-annotated in the same run: last one wins
        for dbid, rec in last.items():
            out.setdefault(dbid, []).append(rec)
    return out


def aggregate(recs: list) -> dict:
    """Combine the annotations of one drug over runs."""
    n = len(recs)
    tiers = [r["annotation"]["safety_tier"] for r in recs]
    counts = Counter(tiers)
    best = max(counts.values())
    tier = next(t for t in TIER_PRIORITY if counts.get(t, 0) == best)
    rep = next(r for r in recs if r["annotation"]["safety_tier"] == tier)    # representative run for the text fields
    ann = rep["annotation"]

    def virus_list(key):
        c = Counter()
        first = {}
        for r in recs:
            for v in r["annotation"][key]:
                c[v.lower()] += 1
                first.setdefault(v.lower(), v)
        return [first[k] for k, cnt in c.most_common() if cnt >= n / 2]

    soft = {}
    for k in SOFT:
        votes = [r["annotation"][k]["known"] for r in recs]
        known = sum(votes) > n / 2
        ev = next((r["annotation"][k]["evidence"] for r in recs if r["annotation"][k]["known"]), "") if known else ""
        soft[k] = (known, ev)
    return {
        "n_runs": n, "tier": tier, "tier_agreement": round(counts[tier] / n, 2), "ann": ann, "soft": soft,
        "rna": virus_list("rna_viruses_inhibited"), "dna": virus_list("dna_viruses_inhibited"),
        "checks": sorted(rep.get("checks", {}).keys()),
    }


def build_report(features: pd.DataFrame, ann_by_id: dict, pts) -> pd.DataFrame:
    rows = []
    for _, f in features.iterrows():
        agg = aggregate(ann_by_id[f["drugbank_id"]]) if f["drugbank_id"] in ann_by_id else None
        tier = agg["tier"] if agg else "unknown"
        flags = {name: bool(f[name]) for name in FLAG_NAMES}
        flags.update(safety_flags(tier))
        sc = score_row(f["ModelAvg4"], flags, pts)
        ann = agg["ann"] if agg else None
        filler = "Not annotated" if f["in_drugbank"] else "Not in DrugBank"
        chembl = f.get("chembl_potency_summary")
        potency = (f"[ChEMBL] {chembl}" if isinstance(chembl, str) and chembl.strip()
                   else (ann["potency_summary"] if ann else filler))
        rows.append({
            "Drug": f["drug_name"],
            "Score before rescaling": round(sc["score_before"], 2) if pd.notna(sc["score_before"]) else np.nan,
            "ModelAvg": f["ModelAvg4"],
            "Score breakdown": sc["breakdown"],
            "Class of drug": ann["drug_class"] if ann else filler,
            "Mechanism of action": ann["mechanism_of_action"] if ann else filler,
            "Toxicity (LD50)": ann["toxicity_summary"] if ann else filler,
            "ED50/IC50": potency,
            "Route of administration": f["routes"] if isinstance(f["routes"], str) and f["routes"] else NOT_REPORTED,
            "RNA virus inhibited": "; ".join(agg["rna"]) if agg and agg["rna"] else "",
            "drugbank_id": f["drugbank_id"],
            "type": f.get("db_type"),
            "safety_tier": tier,
            "safety_evidence": ann["safety_evidence"] if ann else "",
            "DNA virus inhibited": "; ".join(agg["dna"]) if agg and agg["dna"] else "",
            **{f"{k}_known": (agg["soft"][k][0] if agg else False) for k in SOFT},
            **{f"{k}_evidence": (agg["soft"][k][1] if agg else "") for k in SOFT},
            **({"FDA boxed warning": bool(f["fda_has_boxed_warning"]) if pd.notna(f["fda_has_boxed_warning"]) else np.nan}
               if "fda_has_boxed_warning" in f.index else {}),
            "n_runs": agg["n_runs"] if agg else 0,
            "tier_agreement": agg["tier_agreement"] if agg else np.nan,
            "output_checks": "; ".join(agg["checks"]) if agg else "",
        })
    df = pd.DataFrame(rows)
    df.insert(1, "Final score (0-100)", rescale_0_100(df["Score before rescaling"]).round(1))
    df = df.sort_values(["Final score (0-100)", "Drug"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default="../data/DB/admet/drug_table.pkl", help="unified table from build_inputs.py")
    p.add_argument("--annotations", default="../data/DB/admet/annotations")
    p.add_argument("--points", default=str(HERE / "scoring_points.csv"))
    p.add_argument("--out", default="../data/DB/admet/report.tsv")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    features = load_table(a.table)
    pts = load_points(a.points)
    ann = load_annotations(a.annotations)
    df = build_report(features, ann, pts)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, sep="\t", index=False)

    annotated = int((df["n_runs"] > 0).sum())
    print(f"{len(df)} drugs | annotated: {annotated} | not in DrugBank: {int((~features['in_drugbank']).sum())} "
          f"| missing annotation (in DrugBank): {int(((df['n_runs'] == 0) & df['type'].notna()).sum())}")
    print("safety tiers:", df["safety_tier"].value_counts().to_dict())
    print("drugs with output checks triggered:", int((df["output_checks"] != "").sum()))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
