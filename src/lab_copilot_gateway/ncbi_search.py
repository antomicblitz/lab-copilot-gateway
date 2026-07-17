"""NCBI ESearch + ESummary client for the ``opencloning.search_ncbi`` tool.

This module gives the LLM a *generalizable* way to discover NCBI
nucleotide accessions by gene name — instead of hardcoding a static
lookup table of accessions that goes stale every RefSeq release.

It follows the same pattern as ``_invoke_opencloning_search_parts``
(SynVectorDB) and ``_invoke_opencloning_fetch_igem_part`` (iGEM):
call an external public API from within the gateway, return the
results in the same ``ok/result`` envelope the rest of the dispatch
uses.

API reference: https://www.ncbi.nlm.nih.gov/books/NBK25501/
Canonical implementation: this file is the single source of truth.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# NCBI EUtils base. Public, read-only, credential-free.
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI rate limit is 3 req/s without an API key, 10 req/s with one.
# Operators can set NCBI_API_KEY in the gateway environment to raise
# the rate limit. The key is forwarded as a query parameter (NCBI
# documents this in the ESearch/EUsage docs).
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")


def _esearch_url(term: str, retmax: int, db: str) -> str:
    params: dict[str, str] = {
        "db": db,
        "term": term,
        "retmax": str(retmax),
        "retmode": "json",
        "sort": "relevance",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return f"{EUTILS_BASE}/esearch.fcgi?{urlencode(params)}"


def _esummary_url(uids: list[str], db: str) -> str:
    params: dict[str, str] = {
        "db": db,
        "id": ",".join(uids),
        "retmode": "json",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return f"{EUTILS_BASE}/esummary.fcgi?{urlencode(params)}"


def _fetch_json(url: str, timeout: float = 15.0) -> dict[str, Any]:
    """Fetch JSON from an HTTPS URL. Raises HTTPError/URLError on failure."""
    req = Request(
        url, headers={"User-Agent": "LabCopilot/1.0", "Accept": "application/json"}
    )
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — public NCBI API
        return json.loads(resp.read().decode("utf-8"))


def _parse_esummary_results(
    esummary_resp: dict[str, Any], uid_list: list[str]
) -> list[dict[str, Any]]:
    """Parse ESummary response into a list of accession records."""
    result_obj = esummary_resp.get("result") or {}
    uid_arr: list[str] = list(result_obj.get("uids") or [])
    results: list[dict[str, Any]] = []
    for uid in uid_arr:
        rec = result_obj.get(uid) or {}
        results.append(
            {
                "accession": rec.get("caption") or rec.get("accessionversion") or "",
                "title": rec.get("title") or "",
                "length": int(rec.get("slen") or 0),
                "type": rec.get("moleculetype") or "nuccore",
                "uid": uid,
            }
        )
    return results


def search_ncbi(args: dict[str, Any]) -> dict[str, object]:
    """Search NCBI for nucleotide sequences by gene name or description.

    Two-step: ESearch (term → UIDs) → ESummary (UIDs → accessions,
    titles, lengths). The returned accessions are usable directly as
    ``repository_id`` values for ``/repository_id/genbank``.

    Args:
        args: ``query`` (free text or NCBI field-qualified term,
              e.g. ``INS[gene] AND Homo sapiens[Organism]``),
              ``retmax`` (default 5), ``db`` (default ``nuccore``).

    Returns:
        ``{ok, tool_name, result: {results, count, total_available}}``
        on success, or
        ``{ok: false, tool_name, reason, message, status_code?}``
        on failure.
    """
    query = (args.get("query") or "").strip()
    retmax = int(args.get("retmax") or 5)
    db = args.get("db") or "nuccore"

    if not query:
        return {
            "ok": False,
            "tool_name": "opencloning.search_ncbi",
            "reason": "missing_arg",
            "message": "query is required (e.g. 'INS[gene] AND Homo sapiens[Organism]')",
        }

    try:
        # Step 1: ESearch — free text or field-qualified term → UIDs
        esearch_resp = _fetch_json(_esearch_url(query, retmax, db))
        esearch_result = esearch_resp.get("esearchresult") or {}
        uid_list: list[str] = list(esearch_result.get("idlist") or [])
        total_count = int(esearch_result.get("count") or 0)

        if not uid_list:
            return {
                "ok": True,
                "tool_name": "opencloning.search_ncbi",
                "result": {
                    "results": [],
                    "count": 0,
                    "total_available": total_count,
                    "message": (
                        "No sequences found. Try a different query (e.g. use "
                        "[gene] or [Organism] field qualifiers)."
                    ),
                },
            }

        # Step 2: ESummary — UIDs → accessions, titles, lengths
        esummary_resp = _fetch_json(_esummary_url(uid_list, db))
        results = _parse_esummary_results(esummary_resp, uid_list)

        return {
            "ok": True,
            "tool_name": "opencloning.search_ncbi",
            "result": {
                "results": results,
                "count": len(results),
                "total_available": total_count,
            },
        }
    except HTTPError as exc:
        return {
            "ok": False,
            "tool_name": "opencloning.search_ncbi",
            "reason": "ncbi_http_error",
            "message": f"NCBI returned HTTP {exc.code} for {exc.url}: {exc.reason}",
            "status_code": exc.code,
        }
    except URLError as exc:
        return {
            "ok": False,
            "tool_name": "opencloning.search_ncbi",
            "reason": "ncbi_unreachable",
            "message": f"NCBI unreachable: {exc.reason}",
        }
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        return {
            "ok": False,
            "tool_name": "opencloning.search_ncbi",
            "reason": "ncbi_parse_error",
            "message": f"NCBI response parse error: {exc}",
        }
