"""Score = ModelAvg + points from scoring_points.csv (+ optional cap on summed penalties). Pure Python, no LLM."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

SAFETY_TIERS = ["low", "moderate", "high", "unknown"]


@dataclass
class Points:
    scored: Dict[str, dict] = field(default_factory=dict)   # flag -> {points, label, kind}
    cap: Optional[float] = None                              # max summed penalty (positive number) or None


def load_points(path) -> Points:
    df = pd.read_csv(Path(path))
    need = {"flag", "points", "kind", "label"}
    if not need <= set(df.columns):
        raise ValueError(f"{path}: missing columns {sorted(need - set(df.columns))}")
    pts = Points()
    for _, r in df.iterrows():
        kind = r["kind"]
        if kind in ("penalty", "bonus"):
            p = float(r["points"])
            if kind == "penalty" and p > 0:
                raise ValueError(f"{r['flag']}: penalties must be <= 0 (got {p})")
            if kind == "bonus" and p < 0:
                raise ValueError(f"{r['flag']}: bonuses must be >= 0 (got {p})")
            label = r["label"] if isinstance(r["label"], str) and r["label"] else r["flag"]
            pts.scored[r["flag"]] = {"points": p, "label": label, "kind": kind}
        elif kind == "setting" and r["flag"] == "max_total_penalty" and pd.notna(r["points"]):
            pts.cap = abs(float(r["points"]))
    return pts


def safety_flags(tier: str) -> Dict[str, bool]:
    return {f"safety_{t}": tier == t for t in SAFETY_TIERS}


def _fmt(p: float) -> str:
    return f"{p:+g}"


def score_row(model_avg: float, flags: Dict[str, bool], pts: Points) -> dict:
    """Return score_before, the readable breakdown string and the applied points."""
    missing = [f for f in pts.scored if f not in flags]
    if missing:
        raise KeyError(f"flags listed in scoring_points.csv but not provided: {missing}")
    applied = {f: spec["points"] for f, spec in pts.scored.items() if flags[f] and spec["points"] != 0}
    penalties = sum(p for p in applied.values() if p < 0)
    cap_added = 0.0
    if pts.cap is not None and -penalties > pts.cap:
        cap_added = -penalties - pts.cap
    total = sum(applied.values()) + cap_added
    parts = [f"ModelAvg({model_avg:.2f})"] if pd.notna(model_avg) else ["ModelAvg(NA)"]
    parts += [f"{pts.scored[f]['label']}({_fmt(p)})" for f, p in applied.items()]
    if cap_added:
        parts.append(f"Cap({_fmt(cap_added)})")
    return {
        "score_before": model_avg + total if pd.notna(model_avg) else np.nan,
        "breakdown": "+".join(parts),
        "applied": applied,
    }


def rescale_0_100(s: pd.Series) -> pd.Series:
    """Min-max to 0-100 over the whole set (cosmetic: does not change the ranking). All equal -> 50."""
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or hi == lo:
        return pd.Series(np.where(s.notna(), 50.0, np.nan), index=s.index)
    return (s - lo) / (hi - lo) * 100
