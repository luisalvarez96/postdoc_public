"""Checks that LLM output is supported by the text it was given (no fabricated quotes or numbers)."""
import re
from typing import List, Tuple

_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _norm(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s or "")            # DrugBank uses <sub>, <sup>
    s = re.sub(r"[^0-9a-zA-Z.%/ ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def quote_supported(quote: str, source: str, min_overlap: float = 0.9) -> bool:
    """True if the (normalised) quote is a substring of the source, or >= min_overlap of its words occur in it.
    The word-overlap fallback only catches clearly fabricated quotes; it does not prove a faithful quote."""
    q, s = _norm(quote), _norm(source)
    if not q:
        return False
    if q in s:
        return True
    words = [w for w in q.split() if len(w) > 2]
    if not words:
        return False
    vocab = set(s.split())
    return sum(w in vocab for w in words) / len(words) >= min_overlap


def _numbers(text: str) -> List[str]:
    out = []
    for m in _NUM.findall(re.sub(r"<[^>]+>", "", text or "")):
        m = m.replace(",", "") if re.fullmatch(r"\d{1,3}(,\d{3})+", m) else m   # 1,500 == 1500
        out.append(m.rstrip(".,"))
    return out


def numbers_supported(text: str, source: str) -> Tuple[bool, List[str]]:
    """Every number in `text` must also appear in `source`. Returns (ok, unsupported_numbers)."""
    have = set(_numbers(source))
    bad = [n for n in _numbers(text) if n not in have]
    return (not bad, bad)
