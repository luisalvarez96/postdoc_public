"""DrugBank parsing and text cleaning (no LLM, no scoring)."""
import re
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd

NS = {"db": "http://www.drugbank.ca"}

TEXT_COLS = [
    "description", "indication", "pharmacodynamics", "toxicity", "mechanism-of-action", "metabolism", "absorption",
    "half_life", "protein_binding", "clearance", "volume_of_distribution", "route_of_elimination",
    "affected-organisms", "targets", "categories",
]
MISSING_TOKENS = {"", "nan", "none", "unknown", "not available"}


def _find_text(drug, path):
    return drug.findtext(path, namespaces=NS)


def _join(drug, path, sep="|"):
    return sep.join(e.text.strip() for e in drug.findall(path, NS) if e.text and e.text.strip())


def _norm_route(r):
    return re.sub(r"\s+", " ", r).strip().title()


def parse_drugbank_xml(file_path) -> pd.DataFrame:
    """One row per DrugBank entry. Loads the whole XML in memory (several GB, a few minutes)."""
    import xml.etree.ElementTree as ET

    root = ET.parse(file_path).getroot()
    rows = []
    for drug in root.findall("db:drug", NS):
        row = {
            "drugbank_id": drug.findtext('db:drugbank-id[@primary="true"]', namespaces=NS),
            "name": _find_text(drug, "db:name"),
            "type": drug.get("type"),
        }
        for col, tag in [
            ("description", "description"), ("indication", "indication"), ("pharmacodynamics", "pharmacodynamics"),
            ("toxicity", "toxicity"), ("mechanism-of-action", "mechanism-of-action"), ("metabolism", "metabolism"),
            ("absorption", "absorption"), ("half_life", "half-life"), ("protein_binding", "protein-binding"),
            ("clearance", "clearance"), ("volume_of_distribution", "volume-of-distribution"),
            ("route_of_elimination", "route-of-elimination"),
        ]:
            row[col] = _find_text(drug, f"db:{tag}")
        row["groups"] = _join(drug, "db:groups/db:group")
        row["synonyms"] = _join(drug, "db:synonyms/db:synonym")
        row["affected-organisms"] = _join(drug, "db:affected-organisms/db:affected-organism")
        row["targets"] = _join(drug, "db:targets/db:target/db:name")
        row["categories"] = _join(drug, "db:categories/db:category/db:category", " || ")
        row["atc"] = "|".join(a.get("code") for a in drug.findall("db:atc-codes/db:atc-code", NS) if a.get("code"))

        dos = drug.findall("db:dosages/db:dosage", NS)
        row["n_dosages"] = len(dos)
        row["dosage_routes"] = sorted({
            _norm_route(d.findtext("db:route", namespaces=NS)) for d in dos
            if (d.findtext("db:route", namespaces=NS) or "").strip()})
        prods = drug.findall("db:products/db:product", NS)
        row["n_products"] = len(prods)
        row["product_routes"] = sorted({
            _norm_route(p.findtext("db:route", namespaces=NS)) for p in prods
            if (p.findtext("db:route", namespaces=NS) or "").strip()})

        # identifiers for the planned FDA-label / ChEMBL joins (caches built before this change lack them)
        ext = {}
        for e in drug.findall("db:external-identifiers/db:external-identifier", NS):
            res, ident = e.findtext("db:resource", namespaces=NS), e.findtext("db:identifier", namespaces=NS)
            if res and ident:
                ext.setdefault(res, ident)
        row["unii"] = _find_text(drug, "db:unii")
        row["cas_number"] = _find_text(drug, "db:cas-number")
        row["chembl_id"] = ext.get("ChEMBL")
        row["pubchem_cid"] = ext.get("PubChem Compound")
        rows.append(row)
    return pd.DataFrame(rows)


def load_drugbank(xml_path, cache_path, rebuild=False) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if cache_path.exists() and not rebuild:
        return pd.read_pickle(cache_path)
    db = parse_drugbank_xml(xml_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    db.to_pickle(cache_path)
    return db


def clean_citations(v):
    """Drop [A123, L45] style citations and collapse whitespace."""
    if isinstance(v, str):
        v = re.sub(r"\[[A-Za-z0-9,\s]+\]", "", v)
        return re.sub(r"\s+", " ", v).strip()
    return v


def is_missing(v, min_len=1) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return True
    s = str(v).strip()
    return s.lower() in MISSING_TOKENS or len(s) < min_len


# ---------------------------------------------------------------------------
# Category-tag cleaner (moved from admet_scoring.ipynb, unchanged)
# ---------------------------------------------------------------------------
TARGET_NORMALIZATIONS: Dict[str, str] = {
    r"^Cytochrome P-450\s+": "",
    r"^Cytochrome P-450 Enzyme": "CYP",
    r"^UDP-Glucuronosyltransferase\s*": "UGT",
    r"^Glutathione S-Transferase\s*": "GST",
    r"^Sulfotransferase\s*": "SULT",
    r"^N-acetyltransferase\s*": "NAT",
    r"^Thiopurine S-methyltransferase": "TPMT",
    r"^Carboxylesterase\s*": "CES",
    r"^Organic Anion Transporting Polypeptide\s*": "OATP",
    r"^Organic Anion Transporter\s*": "OAT",
    r"^Organic Cation Transporter\s*": "OCT",
    r"^Multidrug Resistance-Associated Protein\s*": "MRP",
    r"^Breast Cancer Resistance Protein": "BCRP",
    r"^Bile Salt Export Pump": "BSEP",
    r"^P-glycoprotein\s*": "P-gp ",
    r"^Monoamine Oxidase\s*": "MAO-",
}
ADME_ROLE_PATTERN = re.compile(
    r"^(.*?)\s+(Substrates|Inhibitors|Inducers|Modulators)(?:\s+with\s+a\s+Narrow\s+Therapeutic\s+Index)?$",
    re.IGNORECASE,
)
CHEMICAL_NOISE_PATTERNS = re.compile(
    r"^(amino acids|proteins|albumins|peptides|lipids|heterocyclic|benzene|"
    r"carboxylic|diterpenes|cycloparaffins|cyclodecanes|terpenes|small molecules|"
    r"organic compounds|macromolecular substances)",
    re.IGNORECASE,
)


def normalize_target_name(target_name: str) -> str:
    for pattern, replacement in TARGET_NORMALIZATIONS.items():
        target_name = re.sub(pattern, replacement, target_name, flags=re.IGNORECASE)
    return target_name.strip()


def clean_and_deduplicate_categories(raw_categories_str: str) -> str:
    """Collapse ADME-role tags (CYPs, transporters, ...) and drop chemical-ontology noise."""
    if not raw_categories_str or raw_categories_str in ("UNKNOWN", "None"):
        return "None specified"
    tags = [t.strip() for t in raw_categories_str.split("||") if t.strip()]

    adme_map: Dict[str, Set[str]] = {}
    filtered_clinical_tags: Set[str] = set()
    has_global_nti = False
    for tag in tags:
        tag_lower = tag.lower()
        if "narrow therapeutic index" in tag_lower:
            has_global_nti = True
        adme_match = ADME_ROLE_PATTERN.match(tag)
        if adme_match:
            target_name = normalize_target_name(adme_match.group(1).strip())
            adme_map.setdefault(target_name, set()).add(adme_match.group(2).capitalize())
            if "narrow therapeutic index" in tag_lower:
                adme_map[target_name].add("NTI")
            continue
        if CHEMICAL_NOISE_PATTERNS.search(tag_lower):
            continue
        filtered_clinical_tags.add(tag)

    sorted_clinical = sorted(filtered_clinical_tags, key=len, reverse=True)
    deduped = []
    for tag in sorted_clinical:
        if not any(tag != existing and tag in existing for existing in deduped):
            deduped.append(tag)

    consolidated = []
    if has_global_nti:
        consolidated.append("Narrow Therapeutic Index (NTI)")
    consolidated.extend(sorted(deduped))
    for target, roles in sorted(adme_map.items()):
        role_order = ["Substrate", "Inhibitor", "Inducer", "Modulator"]
        active = [r for r in role_order if r in roles]
        role_str = "/".join(active) if active else "Modulator"
        if "NTI" in roles:
            role_str += " - NTI"
        consolidated.append(f"{target} {role_str}")
    return ", ".join(consolidated) if consolidated else "None specified"
