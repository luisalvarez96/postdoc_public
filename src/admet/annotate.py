"""Step 2: one LLM annotation call per drug -> <out_dir>/run_<k>/annotations.jsonl.

Run from src/:  python admet/annotate.py --model qwen2.5:7b [--limit 5] [--runs 1] [--workers 1]

- Resumable: drugs already in annotations.jsonl are skipped. Failed drugs are retried in extra passes.
- Every quote and every number in the model output is checked against the text it was given (verify.py).
- The model never sees ModelAvg. The score is computed later by report.py.
"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Template

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from schemas import NOT_REPORTED, DrugAnnotation  # noqa: E402
from table import load_table, records_from_table  # noqa: E402
from verify import numbers_supported, quote_supported  # noqa: E402

PROMPT_FIELDS = [
    "drug_name", "drugbank_id", "drug_type", "atc", "category_tags", "description", "indication",
    "mechanism_of_action", "pharmacodynamics", "targets", "affected_organisms", "absorption", "protein_binding",
    "volume_of_distribution", "metabolism", "half_life", "clearance", "route_of_elimination", "toxicity", "fda_label",
    "routes",
]
SOFT_FLAGS = ("endothelial", "neuronal", "respiratory")


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
def load_prompts(prompts_dir, use_fewshot=True):
    d = Path(prompts_dir)
    few = ""
    if use_fewshot:
        few = Template((d / "annotate_fewshot.txt").read_text()).substitute(not_reported=NOT_REPORTED)
    system = Template((d / "annotate_system.txt").read_text()).substitute(not_reported=NOT_REPORTED, fewshot=few)
    return system, Template((d / "annotate_user.txt").read_text())


def user_prompt(user_template: Template, rec: dict) -> str:
    return user_template.substitute({k: rec[k] for k in PROMPT_FIELDS})


# ---------------------------------------------------------------------------
# post-processing: nothing unsupported by the input survives
# ---------------------------------------------------------------------------
def postprocess(ann: DrugAnnotation, source: str, tox_input_present: bool):
    d, checks = ann.model_dump(), {}
    tier = d["safety_tier"]
    if tier != "unknown":
        if not tox_input_present:
            checks["tier_forced_unknown_no_input"] = tier
            tier = "unknown"
        elif not quote_supported(d["safety_evidence"], source):
            checks["safety_evidence_unsupported"] = tier
            tier = "unknown"
    d["safety_tier"] = tier
    if tier == "unknown":
        d["safety_evidence"] = ""
    for k in SOFT_FLAGS:
        if d[k]["known"] and not quote_supported(d[k]["evidence"], source):
            checks[f"{k}_evidence_unsupported"] = True
            d[k] = {"known": False, "evidence": ""}
    for k in ("toxicity_summary", "potency_summary"):
        ok, bad = numbers_supported(d[k], source)
        if not ok:
            checks[f"{k}_numbers_removed"] = bad
            d[k] = NOT_REPORTED
    return d, checks


# ---------------------------------------------------------------------------
# LLM client (imported lazily so the rest of the module works without instructor/openai)
# ---------------------------------------------------------------------------
def make_annotator(args, system_prompt, user_template):
    import instructor
    from openai import OpenAI

    client = instructor.from_openai(
        OpenAI(base_url=args.base_url, api_key=args.api_key),
        mode=getattr(instructor.Mode, args.instructor_mode),
    )

    def annotate(rec, seed):
        prompt = user_prompt(user_template, rec)
        kwargs = dict(
            model=args.model, response_model=DrugAnnotation, temperature=args.temperature,
            max_retries=args.max_retries, timeout=args.timeout,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        )
        if args.backend == "ollama":
            kwargs["extra_body"] = {"options": {"num_ctx": args.num_ctx, "num_predict": args.max_tokens, "seed": seed}}
        else:
            kwargs.update(max_tokens=args.max_tokens, seed=seed)
        ann = client.chat.completions.create(**kwargs)
        d, checks = postprocess(ann, prompt, rec["tox_input_present"])
        return {"annotation": d, "checks": checks}

    return annotate


# ---------------------------------------------------------------------------
# resumable runner with a retry pass for failed drugs
# ---------------------------------------------------------------------------
def _load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["drugbank_id"])
    return done


def _execute(todo, fn, workers):
    """Yield (record, result, error) as work completes."""
    if workers <= 1:
        for rec in todo:
            try:
                yield rec, fn(rec), None
            except Exception as e:  # noqa: BLE001 - any failure is logged and retried
                yield rec, None, e
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, rec): rec for rec in todo}
        for fut in as_completed(futs):
            try:
                yield futs[fut], fut.result(), None
            except Exception as e:  # noqa: BLE001
                yield futs[fut], None, e


def run_annotation(records, run_dir, annotate_fn, *, run_idx=0, seed=0, workers=1, retry_passes=1,
                   model="", log=print):
    """Annotate `records`, appending one JSON line per drug to run_dir/annotations.jsonl. Returns a summary dict."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ann_path, fail_path = run_dir / "annotations.jsonl", run_dir / "failures.jsonl"
    todo = [r for r in records if r["drugbank_id"] not in _load_done(ann_path)]
    log(f"[run {run_idx}] {len(records) - len(todo)} already done, {len(todo)} to annotate")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    fn = lambda rec: annotate_fn(rec, seed + run_idx)  # noqa: E731
    for attempt in range(retry_passes + 1):
        if not todo:
            break
        label = "pass" if attempt == 0 else f"retry {attempt}"
        failed = []
        it = _execute(todo, fn, workers)
        if tqdm:
            it = tqdm(it, total=len(todo), desc=f"run {run_idx} {label}")
        with open(ann_path, "a") as out, open(fail_path, "a") as fl:
            for rec, res, err in it:
                if err is None:
                    out.write(json.dumps({"drugbank_id": rec["drugbank_id"], "drug_name": rec["drug_name"],
                                          "run": run_idx, "model": model, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                          **res}, ensure_ascii=False) + "\n")
                    out.flush()
                else:
                    fl.write(json.dumps({"drugbank_id": rec["drugbank_id"], "drug_name": rec["drug_name"],
                                         "attempt": attempt, "error": f"{type(err).__name__}: {err}"[:500]}) + "\n")
                    fl.flush()
                    failed.append(rec)
        log(f"[run {run_idx}] {label}: {len(todo) - len(failed)} ok, {len(failed)} failed")
        todo = failed
    return {"run": run_idx, "failed_ids": [r["drugbank_id"] for r in todo], "n_failed": len(todo)}


def _hash(*parts) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


def write_meta(run_dir: Path, meta: dict, force=False):
    p = Path(run_dir) / "run_meta.json"
    if p.exists() and not force:
        old = json.loads(p.read_text())
        for k in ("prompt_hash", "schema_hash", "model"):
            if old.get(k) != meta.get(k):
                raise SystemExit(f"{p}: {k} differs from the earlier run in this directory ({old.get(k)} vs "
                                 f"{meta.get(k)}). Use a new --out_dir or pass --force to mix them.")
    p.write_text(json.dumps(meta, indent=2))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default="../data/DB/admet/drug_table.pkl", help="unified table from build_inputs.py")
    p.add_argument("--max_field_chars", type=int, default=8000, help="cap per DrugBank text field in the prompt")
    p.add_argument("--fda_max_chars", type=int, default=4000, help="cap per FDA label section in the prompt")
    p.add_argument("--out_dir", default="../data/DB/admet/annotations")
    p.add_argument("--prompts_dir", default=str(HERE / "prompts"))
    p.add_argument("--no_fewshot", action="store_true", help="omit the synthetic few-shot example")
    p.add_argument("--backend", choices=["ollama", "openai"], default="ollama",
                   help="ollama: pass num_ctx/num_predict/seed as Ollama options; openai: standard max_tokens/seed")
    p.add_argument("--base_url", default="http://localhost:11434/v1")
    p.add_argument("--api_key", default="ollama")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--instructor_mode", default="JSON", choices=["JSON", "TOOLS", "JSON_SCHEMA", "MD_JSON"])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0, help="base seed; run k uses seed + k")
    p.add_argument("--num_ctx", type=int, default=8192, help="Ollama context window (prompt + output)")
    p.add_argument("--max_tokens", type=int, default=1500, help="max output tokens")
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--max_retries", type=int, default=3, help="instructor retries on validation errors")
    p.add_argument("--runs", type=int, default=1, help="number of independent runs (for noise / majority vote)")
    p.add_argument("--run_offset", type=int, default=0, help="first run index (to add runs to an existing out_dir)")
    p.add_argument("--workers", type=int, default=1, help="parallel requests (keep 1 if the server struggles)")
    p.add_argument("--retry_passes", type=int, default=1, help="extra passes over drugs that failed")
    p.add_argument("--limit", type=int, default=None, help="only the first N drugs (test runs)")
    p.add_argument("--only", nargs="+", default=None, help="only these drug names (case-insensitive)")
    p.add_argument("--print_prompt", action="store_true", help="print the filled prompt for the first drug and exit")
    p.add_argument("--force", action="store_true", help="allow mixing runs made with different prompts/model")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    records = records_from_table(load_table(a.table), a.max_field_chars, a.fda_max_chars)
    if a.only:
        want = {x.lower() for x in a.only}
        records = [r for r in records if r["drug_name"].lower() in want]
    if a.limit:
        records = records[: a.limit]
    system_prompt, user_t = load_prompts(a.prompts_dir, not a.no_fewshot)

    if a.print_prompt:
        print("=== SYSTEM ===\n" + system_prompt + "\n=== USER ===\n" + user_prompt(user_t, records[0]))
        return 0

    meta_base = {
        "model": a.model, "backend": a.backend, "temperature": a.temperature, "num_ctx": a.num_ctx,
        "max_tokens": a.max_tokens, "fewshot": not a.no_fewshot,
        "prompt_hash": _hash(system_prompt, user_t.template),
        "schema_hash": _hash(json.dumps(DrugAnnotation.model_json_schema(), sort_keys=True)),
    }
    annotate_fn = make_annotator(a, system_prompt, user_t)
    n_failed = 0
    for k in range(a.run_offset, a.run_offset + a.runs):
        run_dir = Path(a.out_dir) / f"run_{k}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_meta(run_dir, {**meta_base, "seed": a.seed + k, "run": k}, a.force)
        s = run_annotation(records, run_dir, annotate_fn, run_idx=k, seed=a.seed, workers=a.workers,
                           retry_passes=a.retry_passes, model=a.model)
        n_failed += s["n_failed"]
        if s["n_failed"]:
            (run_dir / "failed_ids.txt").write_text("\n".join(s["failed_ids"]) + "\n")
            print(f"[run {k}] {s['n_failed']} drugs still failing; see {run_dir / 'failures.jsonl'}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
