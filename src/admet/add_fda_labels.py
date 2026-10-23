"""Optional step: add FDA label safety text (openFDA API) to the unified table, one request pair per drug.

Run from src/ on a machine with internet:  python admet/add_fda_labels.py --limit 3     (try 3 drugs first)
Adds columns fda_* to drug_table.pkl: boxed warning, warnings, adverse reactions, counts and provenance.
Only the UNII (or, as a fallback, the generic drug name) is sent to api.fda.gov. Responses are cached on disk, so the
step is resumable and cheap to re-run.

Rules (conservative on purpose):
- Only labels of the single-ingredient product are used (a combination product's boxed warning may belong to its partner
  drug). If only combination labels exist, no text is added and fda_boxed_combination_only records it.
- The most recent matching label (by effective_time) supplies the text.
- fda_has_boxed_warning is True only if a single-ingredient label has a boxed warning.
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table import load_table, merge_columns, save_table  # noqa: E402
from web import JsonCache, get_json  # noqa: E402

FDA_URL = "https://api.fda.gov/drug/label.json"
SCHEMA_VERSION = 1                      # bump when the extraction logic changes (invalidates cached results)
WARNING_FIELDS = ["warnings_and_cautions", "warnings"]
PAGES = (8, 50)                         # first look at 8 labels, widen to 50 if none is single-ingredient
FDA_COLS = ["fda_match", "fda_unii_used", "fda_n_labels", "fda_n_boxed", "fda_label_date", "fda_set_id",
            "fda_single_ingredient_found", "fda_boxed_combination_only", "fda_has_boxed_warning", "fda_boxed_warning",
            "fda_warnings", "fda_adverse_reactions", "fda_error"]


def _text(v) -> str:
    if isinstance(v, list):
        v = "\n\n".join(str(x) for x in v)
    v = re.sub(r"[ \t]+", " ", str(v or ""))
    return re.sub(r"\n{3,}", "\n\n", v).strip()


def _date(label) -> str:
    t = str(label.get("effective_time", ""))
    return f"{t[:4]}-{t[4:6]}-{t[6:8]}" if len(t) == 8 else t


def _is_single(label, kind, unii, name) -> bool:
    of = label.get("openfda", {})
    if kind == "unii":
        return {u.upper() for u in of.get("unii", [])} == {unii.upper()}
    gn = {g.lower() for g in of.get("generic_name", [])}
    return bool(gn) and gn == {name.lower()}


def search_labels(get, search, limit, api_key):
    params = {"search": search, "limit": limit, "sort": "effective_time:desc"}
    if api_key:
        params["api_key"] = api_key
    r = get(FDA_URL, params, not_found_ok=True)
    if r is None:
        return 0, []
    return int(r.get("meta", {}).get("results", {}).get("total", len(r.get("results", [])))), r.get("results", [])


def first_single(get, search, kind, unii, name, api_key):
    """(total hits, first single-ingredient label or None, any labels returned)."""
    total, labels = 0, []
    for limit in PAGES:
        total, labels = search_labels(get, search, limit, api_key)
        single = next((lab for lab in labels if _is_single(lab, kind, unii, name)), None)
        if single is not None or total <= limit:
            return total, single, bool(labels)
    return total, None, bool(labels)


def _empty() -> dict:
    out = {c: None for c in FDA_COLS}
    out.update(fda_n_labels=0, fda_n_boxed=0, fda_single_ingredient_found=False, fda_boxed_combination_only=False,
               fda_has_boxed_warning=False)
    return out


def _attempt(get, kind, search, unii, name, api_key):
    """Result for one search (by UNII or by name), or None if it has no hits at all."""
    total_a, label_a, _ = first_single(get, search, kind, unii, name, api_key)
    if total_a == 0:
        return None
    out = _empty()
    out.update(fda_match=kind, fda_unii_used=unii if kind == "unii" else None, fda_n_labels=total_a)
    total_b, label_b, any_b = first_single(get, search + " AND _exists_:boxed_warning", kind, unii, name, api_key)
    out["fda_n_boxed"] = total_b
    if label_a is not None:
        out.update(fda_single_ingredient_found=True, fda_label_date=_date(label_a), fda_set_id=label_a.get("set_id"),
                   fda_adverse_reactions=_text(label_a.get("adverse_reactions")) or None)
        out["fda_warnings"] = next((_text(label_a[f]) for f in WARNING_FIELDS if _text(label_a.get(f))), None)
    if label_b is not None:
        out.update(fda_has_boxed_warning=True, fda_boxed_warning=_text(label_b.get("boxed_warning")))
    elif any_b:
        out["fda_boxed_combination_only"] = True
    return out


def fetch_for_drug(get, unii, name, api_key=None, name_fallback=True) -> dict:
    """Extract the fda_* fields for one drug. `get(url, params, not_found_ok=True)` returns parsed JSON or None.
    A search with a single-ingredient label wins; otherwise the first search that had any hit is reported."""
    attempts = []
    if unii:
        attempts.append(("unii", f'openfda.unii:"{unii}"'))
    if name and name_fallback:
        attempts.append(("name", f'openfda.generic_name:"{name}"'))
    found = []
    for kind, search in attempts:
        res = _attempt(get, kind, search, unii or "", name or "", api_key)
        if res is None:
            continue
        if res["fda_single_ingredient_found"]:
            return res
        found.append(res)
    return found[0] if found else _empty()


def run(table: pd.DataFrame, get, *, cache: JsonCache, api_key=None, name_fallback=True, limit=None, refresh=False,
        max_failures=5, log=print) -> pd.DataFrame:
    rows = table[table["in_drugbank"].astype(bool)]
    if limit:
        rows = rows.head(limit)
    results, live, failed, consecutive = {}, 0, 0, 0
    for i, (_, r) in enumerate(rows.iterrows(), 1):
        unii = r.get("db_unii") if isinstance(r.get("db_unii"), str) and r.get("db_unii").strip() else None
        name = r.get("db_name") if isinstance(r.get("db_name"), str) else r["drug_name"]
        key = f"v{SCHEMA_VERSION}_{unii or 'noUNII_' + str(name).lower()}"
        res = None if refresh else cache.get(key)
        if res is None:
            try:
                res = fetch_for_drug(get, unii, name, api_key, name_fallback)
                cache.put(key, res)
                live += 1
                consecutive = 0
            except Exception as e:  # noqa: BLE001 - recorded per drug, not cached, retried on the next run
                res = {**_empty(), "fda_error": f"{type(e).__name__}: {e}"[:300]}
                failed += 1
                consecutive += 1
        results[r["drugbank_id"]] = res
        if consecutive >= max_failures:
            log(f"stopping: {consecutive} consecutive failures (is api.fda.gov reachable?). Progress is cached; "
                "re-run later to continue.")
            break
        if i % 25 == 0:
            log(f"  {i}/{len(rows)} drugs ({live} fetched live, {failed} failed)")
    new = pd.DataFrame.from_dict(results, orient="index").reindex(columns=FDA_COLS)
    new["fda_fetched_at"] = time.strftime("%Y-%m-%d")
    out = merge_columns(table, new)
    for c in ("fda_has_boxed_warning", "fda_single_ingredient_found", "fda_boxed_combination_only"):
        out[c] = out[c].map(lambda v: bool(v) if pd.notna(v) else False)
    out["fda_n_labels"] = out["fda_n_labels"].fillna(0).astype(int)
    out["fda_n_boxed"] = out["fda_n_boxed"].fillna(0).astype(int)
    return out


def summary(table: pd.DataFrame) -> str:
    k = table["in_drugbank"].astype(bool)
    return (f"{int(k.sum())} drugs | matched by UNII: {int((table['fda_match'] == 'unii').sum())}, by name: "
            f"{int((table['fda_match'] == 'name').sum())}, no label: {int((k & table['fda_match'].isna()).sum())} | "
            f"single-ingredient label: {int(table['fda_single_ingredient_found'].sum())} | "
            f"boxed warning: {int(table['fda_has_boxed_warning'].sum())} | "
            f"only combination labels have one: {int(table['fda_boxed_combination_only'].sum())} | "
            f"failed requests: {int(table['fda_error'].notna().sum())}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default="../data/DB/admet/drug_table.pkl")
    p.add_argument("--cache_dir", default="../data/DB/admet/cache/openfda")
    p.add_argument("--api_key", default=os.environ.get("OPENFDA_API_KEY"),
                   help="free openFDA key (or env OPENFDA_API_KEY); without one the daily request limit is low")
    p.add_argument("--sleep", type=float, default=0.3, help="pause after each live request, in seconds")
    p.add_argument("--retries", type=int, default=3, help="retries per request on 429/5xx/network errors")
    p.add_argument("--max_failures", type=int, default=5, help="stop after this many consecutive failed drugs")
    p.add_argument("--no_name_fallback", action="store_true", help="do not search by generic name when the UNII fails")
    p.add_argument("--limit", type=int, default=None, help="only the first N DrugBank-matched drugs (test run)")
    p.add_argument("--refresh", action="store_true", help="ignore the cache")
    return p.parse_args(argv)


def main(argv=None, get=None):
    a = parse_args(argv)
    table = load_table(a.table)
    if "db_unii" not in table.columns:
        print("warning: no db_unii column in the table; re-run build_inputs.py --rebuild_cache (name fallback only)")
    if get is None:
        get = lambda url, params, **kw: get_json(url, params, sleep=a.sleep, retries=a.retries, **kw)  # noqa: E731
    out = run(table, get, cache=JsonCache(a.cache_dir), api_key=a.api_key, name_fallback=not a.no_name_fallback,
              limit=a.limit, refresh=a.refresh, max_failures=a.max_failures)
    if a.limit:
        print("NOTE: --limit was used; the output table only has fda_* data for those drugs")
    save_table(out, Path(a.table).parent)
    print(summary(out))


if __name__ == "__main__":
    main()
