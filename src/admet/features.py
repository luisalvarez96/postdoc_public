"""Deterministic drug features and flags from DrugBank structured fields (no LLM).

The flag names must match the `flag` column of scoring_points.csv. The definitions below are first proposals and
should be reviewed (patterns marked VERIFY have not been checked against the candidate list).
"""
import re
from typing import Dict, Iterable, Set

from drugbank import is_missing

FLAG_NAMES = [
    "vaccine", "antibody", "other_biotech", "diagnostic_agent", "chemotherapy", "immunosuppressant",
    "antiinflammatory", "iv_only", "topical_only", "oral_or_inhalation",
]

TOX_TAG = re.compile(r"toxic|teratogen|carcinogen|mutagen|hepatotox|nephrotox|cardiotox|neurotox|ototox|myelosuppress", re.I)
VACCINE_NAME = re.compile(r"vaccin|antigen|toxoid", re.I)          # VERIFY: 'antigen'/'toxoid' in drug names
MAB_NAME = re.compile(r"mab\b", re.I)
DIAG_CATEGORY = re.compile(r"contrast media|diagnostic|radiopharmaceutical|radiolog|imaging", re.I)  # VERIFY

ORAL_TOKENS = {"oral"}
INHALATION_HINTS = ("inhal", "respiratory")
IV_COMPATIBLE = {"intravenous", "parenteral"}       # 'parenteral' alone is not enough to call a drug IV-only
TOPICAL_TOKENS = {"topical", "cutaneous", "ophthalmic", "auricular (otic)", "dental", "intravitreal"}  # nasal/transdermal excluded


def route_tokens(routes: Iterable[str]) -> Set[str]:
    """DrugBank joins several routes with ';' (e.g. 'Intravenous; Parenteral'): split into atomic lowercase tokens."""
    toks = set()
    for r in routes or []:
        for part in str(r).split(";"):
            part = part.strip().lower()
            if part:
                toks.add(part)
    return toks


def deterministic_flags(name: str, dtype: str, atc: str, categories: str, routes: Iterable[str]) -> Dict[str, bool]:
    name, dtype = name or "", (dtype or "").lower()
    cats = categories or ""
    codes = [c for c in str(atc or "").split("|") if c]
    toks = route_tokens(routes)

    vaccine = (any(c.startswith("J07") for c in codes) or bool(VACCINE_NAME.search(name))
               or "vaccin" in cats.lower())
    biotech = dtype == "biotech"
    antibody = biotech and not vaccine and bool(MAB_NAME.search(name))
    return {
        "vaccine": vaccine,
        "antibody": antibody,
        "other_biotech": biotech and not vaccine and not antibody,
        "diagnostic_agent": any(c.startswith(("V08", "V09")) for c in codes) or bool(DIAG_CATEGORY.search(cats)),
        "chemotherapy": any(c.startswith("L01") for c in codes),
        "immunosuppressant": "immunosuppress" in cats.lower(),
        "antiinflammatory": any(c.startswith("M01A") for c in codes) or "anti-inflammatory" in cats.lower(),
        "iv_only": bool(toks) and "intravenous" in toks and toks <= IV_COMPATIBLE,
        "topical_only": bool(toks) and toks <= TOPICAL_TOKENS,
        "oral_or_inhalation": any(t in ORAL_TOKENS for t in toks) or any(h in t for t in toks for h in INHALATION_HINTS),
    }


def toxicity_info(toxicity, categories) -> Dict[str, bool]:
    """Whether the input carries any safety information (used to force safety_tier='unknown')."""
    return {
        "tox_text_present": not is_missing(toxicity, 30),
        "tox_tag_present": bool(TOX_TAG.search(categories or "")),
    }
