# src/acme_trust_cli/parsers/readme_parser.py
from __future__ import annotations
import re
from typing import Iterable, List, Optional, Tuple

# ---------- Markdown cleanup helpers ----------

FENCED_BLOCK = re.compile(r"(^|\n)```.*?\n.*?\n```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`]+`")
HTML_COMMENTS = re.compile(r"<!--.*?-->", re.DOTALL)
LINK_REF_DEF = re.compile(r"^\s*\[[^\]]+\]:\s+\S+.*$", re.MULTILINE)

def _strip_markdown_noise(md: str) -> str:
    """
    Remove content that commonly confuses heading scans and regex matching:
      - fenced code blocks ```...```
      - inline code `...`
      - HTML comments <!-- ... -->
      - link reference definitions [id]: url
    """
    text = FENCED_BLOCK.sub("\n", md)
    text = INLINE_CODE.sub(" ", text)
    text = HTML_COMMENTS.sub(" ", text)
    text = LINK_REF_DEF.sub("", text)
    return text
# example usage:
# cleaned_md = _strip_markdown_noise(readme_md)
# ---------- Section extraction ----------

LICENSE_HX = re.compile(r"^(#{1,6})\s*license\b.*$", re.IGNORECASE | re.MULTILINE)

def extract_section(md: str, title_regex: re.Pattern) -> Optional[str]:
    """
    Generic section extractor: find a heading match and return text until the next
    heading of the same or higher level.
    """
    if not md:
        return None
    md2 = _strip_markdown_noise(md)
    m = title_regex.search(md2)
    if not m:
        return None

    level = len(m.group(1)) if m.lastindex and isinstance(m.group(1), str) else 1
    start = m.end()

    # Next heading of same-or-higher level
    nxt = re.search(rf"^#{{1,{level}}}\s+\S+", md2[start:], flags=re.MULTILINE)
    end = (start + nxt.start()) if nxt else len(md2)
    section = md2[start:end].strip()
    return section or None

def extract_license_block(md: str) -> Optional[str]:
    """Convenience wrapper to slice the README's License section text."""
    return extract_section(md, LICENSE_HX)
# example usage:
# readme_md = "... contents of README.md ..."
# license_section = extract_license_block(readme_md)
# if license_section:
#     spdx_ids = find_spdx_ids(license_section)
#     spdx_exprs = find_spdx_expressions(license_section)
#     hints = find_license_hints(license_section)
# ---------- SPDX detection ----------

# SPDX header form (often in LICENSE files)
SPDX_LINE = re.compile(
    r"spdx[- ]license[- ]identifier\s*:\s*([A-Za-z0-9\.\-\+]+)",
    re.IGNORECASE,
)
# Also detect bare SPDX-like tokens, e.g., MIT, Apache-2.0, GPL-3.0-only
SPDX_TOKEN = re.compile(
    r"\b([A-Za-z][A-Za-z0-9\.\-\+]{1,40})\b"
)

def find_spdx_ids(text: str) -> List[str]:
    """
    Return explicit SPDX identifiers found either in a header-style line or as bare tokens.
    You should still post-process with your canonical whitelist/blacklist.
    Dedupes while preserving first-seen order.
    """
    if not text:
        return []
    ids: List[str] = []

    # Header style
    for m in SPDX_LINE.finditer(text):
        ids.append(m.group(1))

    # Bare tokens (filter to things that look like SPDX IDs)
    # We'll keep a small allowlist of common prefixes to reduce false positives.
    ALLOWED_PREFIXES = (
        "MIT", "Apache-", "BSD-", "GPL-", "LGPL-", "AGPL-",
        "MPL-", "EPL-", "CDDL-", "CC", "Unlicense", "Zlib", "Artistic-", "WTFPL"
    )
    for m in SPDX_TOKEN.finditer(text):
        token = m.group(1)
        if any(token.startswith(pref) for pref in ALLOWED_PREFIXES) or token in ("ISC",):
            ids.append(token)

    # Deduplicate, keep order
    seen = set()
    out = []
    for t in ids:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
# example usage:
# readme_md = "... contents of README.md ..."
# license_section = extract_license_block(readme_md)
# if license_section:
#     spdx_ids = find_spdx_ids(license_section)
# ---------- SPDX expression harvesting ----------

# Very light-weight pull of expressions like "MIT OR GPL-3.0-only", "(Apache-2.0 OR MIT)"
SPDX_EXPR = re.compile(
    r"(?P<expr>(?:\(|\s|^)(?:[A-Za-z0-9\.\-\+]+\s+(?:OR|AND)\s+)+[A-Za-z0-9\.\-\+]+(?:\)|\s|$))",
    re.IGNORECASE,
)

# def find_spdx_expressions(text: str) -> List[str]:
#     """
#     Return raw potential SPDX expressions to feed into your expression parser.
#     """
#     if not text:
#         return []
#     hits = []
#     for m in SPDX_EXPR.finditer(text):
#         expr = m.group("expr").strip(" ()\n\t")
#         hits.append(expr)
#     # Dedup while preserving order
#     seen = set()
#     out = []
#     for e in hits:
#         if e not in seen:
#             seen.add(e)
#             out.append(e)
#     return out
# example usage:
# readme_md = "... contents of README.md ..."
# license_section = extract_license_block(readme_md)
# if license_section:
#     spdx_ids = find_spdx_ids(license_section)
#     spdx_exprs = find_spdx_expressions(license_section)
# ---------- Named-license / keyword hints ----------

# Looser matches to feed your alias map (not canonical by themselves)
NAMED_LICENSE = re.compile(
    r"\b("
    r"mit(?:\s+license)?|"
    r"apache(?:\s+license)?(?:\s*2(?:\.0)?)?|"
    r"bsd(?:\s*(?:2|3)[ -]?clause)?|"
    r"isc(?:\s+license)?|"
    r"mozilla\s+public\s+license|mpl|"
    r"eclipse\s+public\s+license|epl|"
    r"gnu\s+lesser\s+general\s+public\s+license|lgpl(?:\s*v?\d(?:\.\d)?)?|"
    r"gnu\s+general\s+public\s+license|gpl(?:\s*v?\d(?:\.\d)?)?|"
    r"agpl(?:\s*v?\d(?:\.\d)?)?|"
    r"cc[-\s]?by[-\s]?nc(?:[-\s]?sa)?(?:\s*\d(?:\.\d)?)?|"
    r"creative\s+commons\s+zero|cc0|"
    r"unlicense|zlib|artistic\s*2(?:\.0)?|wtfpl|"
    r"proprietary|non[-\s]?commercial|shareware|freeware|"
    r"see\s+license|custom\s+license|open\s+source"
    r")\b",
    re.IGNORECASE,
)

def find_license_hints(text: str) -> List[str]:
    """
    Return fuzzy license keywords (lowercased) you can pass to your alias map or
    ambiguous buckets. Use this *after* explicit SPDX detection.
    """
    if not text:
        return []
    matches = [m.group(1).lower() for m in NAMED_LICENSE.finditer(text)]
    # Normalize trivial variants
    normed = [m.replace("  ", " ").strip() for m in matches]
    # Dedup keep order
    seen = set()
    out = []
    for t in normed:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
# Example usage:
# readme_md = "... contents of README.md ..."
# license_section = extract_license_block(readme_md)
# if license_section:
#     spdx_ids = find_spdx_ids(license_section)
#     spdx_exprs = find_spdx_expressions(license_section)
#     hints = find_license_hints(license_section)

# ---------- Convenience pipeline ----------

def extract_license_evidence(readme_md: Optional[str], license_file_text: Optional[str]) -> Tuple[str, List[str], List[str], List[str]]:
    """
    High-level helper:
      1) Prefer LICENSE* file text; else use README's License section.
      2) From chosen text, return:
         - source: "LICENSE" | "README" | "NONE"
         - spdx_ids: explicit IDs found
         - spdx_exprs: expressions to feed to parser (e.g., 'MIT OR GPL-3.0-only')
         - hints: fuzzy keywords to resolve via aliases / ambiguous rules
    """
    source = "NONE"
    chosen = None

    if license_file_text and license_file_text.strip():
        source = "LICENSE"
        chosen = license_file_text
    elif readme_md:
        block = extract_license_block(readme_md)
        if block:
            source = "README"
            chosen = block

    if not chosen:
        return source, [], [], []

    ids = find_spdx_ids(chosen)
    # exprs = find_spdx_expressions(chosen)
    exprs = []
    hints = find_license_hints(chosen)
    return source, ids, exprs, hints
# Example usage:
# readme_md = "... contents of README.md ..."
# license_file_text = "... contents of LICENSE file ..."
# source, spdx_ids, spdx_exprs, hints = extract_license_evidence(readme_md, license_file_text)



