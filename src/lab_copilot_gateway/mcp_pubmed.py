"""PubMed MCP result normalizer (Slice 4).

Pure functions that convert the raw ``structuredContent`` from the
PubMed MCP server (v2.9.8) into the bounded article-record shape the
Gateway contract defines.

Remote tool names (actual server): ``pubmed_search_articles``,
``pubmed_fetch_articles``.  Local Gateway names: ``literature.search_pubmed``,
``literature.fetch_pubmed_articles``.

---- Deployment runbook (schema-hash pinning) ----

Schema hashes were computed *authoritatively* from the Zod v4 source
definitions in the pubmed-mcp-server v2.9.8 npm package (commit
ef0be2c0).  The ``z4mini.toJSONSchema({target: "draft-7", io: "input"})``
output (the exact path the ``@modelcontextprotocol/sdk`` uses) was
canonicalized with ``json.dumps(sort_keys=True, separators=(",", ":"))``
and hashed with SHA-256.

If the server ever upgrades, re-confirm with::

    python -c "
    import json, hashlib
    from lab_copilot_gateway.mcp_adapter import McpRemoteTool
    # paste tools/list output for pubmed_search_articles / pubmed_fetch_articles
    rt = McpRemoteTool(name='...', input_schema={...})
    print(rt.input_schema_hash())
    "

.. and replace the hash in ``mcp_bindings.py``.
"""

from __future__ import annotations

from typing import Any

MAX_ABSTRACT_CHARS: int = 500
"""Maximum abstract length in characters before truncation."""


def normalize_search_result(raw_structured_content: Any) -> dict[str, Any]:
    """Normalize a ``pubmed_search_articles`` result into the contract shape.

    The raw result has ``pmids`` + ``summaries`` lists with optional
    ``enriched`` metadata.  We extract only the article records.
    """
    if not isinstance(raw_structured_content, dict):
        return {"articles": [], "total": 0}

    summaries = raw_structured_content.get("summaries")
    if not isinstance(summaries, list):
        return {"articles": [], "total": 0}

    articles = [normalize_article(s) for s in summaries if isinstance(s, dict)]
    return {"articles": articles, "total": len(articles)}


def normalize_fetch_result(raw_structured_content: Any) -> dict[str, Any]:
    """Normalize a ``pubmed_fetch_articles`` result into the contract shape.

    The raw result has ``articles`` (detailed records) and
    ``totalReturned``.  Each article may have a fuller abstract.
    """
    if not isinstance(raw_structured_content, dict):
        return {"articles": [], "total": 0}

    raw_articles = raw_structured_content.get("articles")
    if not isinstance(raw_articles, list):
        return {"articles": [], "total": 0}

    articles = [normalize_article(a) for a in raw_articles if isinstance(a, dict)]
    return {"articles": articles, "total": len(articles)}


def normalize_article(raw_article: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single article record into the bounded Gateway shape.

    Returns a dict with every contract key present (null / ``[]`` defaults).
    Abstract is truncated to ``MAX_ABSTRACT_CHARS``.
    The PubMed URL is synthesized from the PMID.
    """
    pmid = _as_str(raw_article.get("pmid"))
    return {
        "pmid": pmid,
        "title": _as_str(raw_article.get("title")),
        "authors": _extract_authors(raw_article.get("authors")),
        "journal": _extract_journal(raw_article),
        "year": _extract_year(raw_article),
        "doi": _as_str(raw_article.get("doi")),
        "abstract": _extract_abstract(raw_article.get("abstractText")),
        "url": _extract_url(raw_article.get("pubmedUrl"), pmid),
    }


# ---------------------------------------------------------------------------
# Internal helpers — each single-purpose to keep complexity low
# ---------------------------------------------------------------------------


def _as_str(value: Any) -> str | None:
    """Cast to str, returning None for empty/falsy/missing values."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _extract_authors(raw_authors: Any) -> list[str]:
    """Normalize authors from various shapes to a list of formatted strings.

    The fetch tool returns structured authors [{firstName, lastName, ...}].
    The search tool returns a flat "authors" string.
    """
    if isinstance(raw_authors, list):
        result: list[str] = []
        for a in raw_authors:
            if isinstance(a, dict):
                name = _format_author(a)
                if name:
                    result.append(name)
            elif isinstance(a, str) and a.strip():
                result.append(a.strip())
        return result
    if isinstance(raw_authors, str) and raw_authors.strip():
        return [raw_authors.strip()]
    return []


def _extract_journal(raw_article: dict[str, Any]) -> str | None:
    """Extract journal: prefer journalInfo.title then fall back to source."""
    journal_info = raw_article.get("journalInfo")
    if isinstance(journal_info, dict):
        journal = _as_str(journal_info.get("title"))
        if journal:
            return journal
    return _as_str(raw_article.get("source"))


def _extract_year(raw_article: dict[str, Any]) -> int | None:
    """Extract publication year from pubDate (search) or journalInfo (fetch)."""
    pub_date = raw_article.get("pubDate")
    if isinstance(pub_date, str) and pub_date.strip():
        year = _parse_year_from_date_string(pub_date)
        if year is not None:
            return year
    ji = raw_article.get("journalInfo")
    if isinstance(ji, dict):
        pd = ji.get("publicationDate")
        if isinstance(pd, dict):
            year_str = _as_str(pd.get("year"))
            if year_str:
                try:
                    return int(year_str)
                except (ValueError, TypeError):
                    pass
    return None


def _extract_abstract(raw_abstract: Any) -> str | None:
    """Extract and optionally truncate abstract text to MAX_ABSTRACT_CHARS."""
    text = _as_str(raw_abstract)
    if text and len(text) > MAX_ABSTRACT_CHARS:
        return text[:MAX_ABSTRACT_CHARS]
    return text if text else None


def _extract_url(raw_url: Any, pmid: str | None) -> str | None:
    """Extract PubMed URL or synthesize from PMID."""
    url = _as_str(raw_url)
    if url:
        return url
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return None


def _format_author(author: dict[str, Any]) -> str:
    """Format a structured author dict to a readable string."""
    collective = _as_str(author.get("collectiveName"))
    if collective:
        return collective

    last = _as_str(author.get("lastName")) or ""
    first = _as_str(author.get("firstName")) or ""
    initials = _as_str(author.get("initials")) or ""

    # Reason: compose name from the strongest available fields.
    if last and first:
        name = f"{first} {last}"
    elif last:
        name = last
    elif first:
        name = first
    elif initials:
        name = initials
    else:
        return ""

    return name.strip()


def _parse_year_from_date_string(pub_date: str) -> int | None:
    """Extract a 4-digit year from a PubMed date string like '2024 Feb 01'."""
    import re

    m = re.search(r"(\d{4})", pub_date)
    if m:
        try:
            return int(m.group(1))
        except (ValueError, TypeError):
            pass
    return None
