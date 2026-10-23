"""Small HTTP + disk-cache helpers for the external-source fetchers (standard library only).

Only public identifiers (UNII, ChEMBL ID, generic drug names) are ever sent. No DrugBank text leaves the machine.
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

USER_AGENT = "hgtdr-admet-fetch/0.1 (academic research; contact: see repository)"


class HttpError(RuntimeError):
    pass


def get_json(url: str, params: Optional[dict] = None, *, retries: int = 5, backoff: float = 2.0, timeout: float = 60,
             not_found_ok: bool = False, sleep: float = 0.0):
    """GET a JSON document. Retries on 429/5xx and network errors with exponential backoff.

    Returns None for HTTP 404 when `not_found_ok` (openFDA answers 404 when a search has no hits).
    `sleep` is a polite pause after every successful live request.
    """
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params, doseq=True)
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if sleep:
                time.sleep(sleep)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404 and not_found_ok:
                return None
            last = e
            if e.code not in (429, 500, 502, 503, 504):
                raise HttpError(f"HTTP {e.code} for {url}") from e
            wait = float(e.headers.get("Retry-After", 0) or 0) or backoff ** attempt
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            last, wait = e, backoff ** attempt
        if attempt < retries:
            time.sleep(wait)
    raise HttpError(f"giving up after {retries + 1} attempts: {url} ({last})")


class JsonCache:
    """One JSON file per key. Lets a fetch be interrupted and resumed, and re-runs cost no requests."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".json")

    def get(self, key: str):
        p = self._path(key)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, key: str, obj) -> None:
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False))
        tmp.replace(self._path(key))
