"""Unit tests for the opencloning.search_ncbi tool.

Tests the NCBI ESearch + ESummary client without hitting the network.
URL building and response parsing are tested in isolation; the
end-to-end integration is exercised by the orchestrator's
playwright-driven smoke tests against a live NCBI EUtils.
"""

from __future__ import annotations

from unittest.mock import patch

from lab_copilot_gateway.ncbi_search import (
    _esearch_url,
    _esummary_url,
    search_ncbi,
)


# -------- URL building ---------------------------------------------------


def test_esearch_url_basic():
    url = _esearch_url("INS[gene] AND Homo sapiens[Organism]", 5, "nuccore")
    assert url.startswith("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?")
    assert "db=nuccore" in url
    assert "retmax=5" in url
    assert "retmode=json" in url
    assert "term=INS%5Bgene%5D" in url


def test_esummary_url_basic():
    url = _esummary_url(["1234", "5678"], "nuccore")
    assert url.startswith(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
    )
    assert "db=nuccore" in url
    assert "id=1234%2C5678" in url or "id=1234,5678" in url


# -------- search_ncbi(): missing arg -------------------------------------


def test_search_ncbi_rejects_empty_query():
    result = search_ncbi({"query": ""})
    assert result["ok"] is False
    assert result["tool_name"] == "opencloning.search_ncbi"
    assert result["reason"] == "missing_arg"


def test_search_ncbi_rejects_missing_query():
    result = search_ncbi({})
    assert result["ok"] is False
    assert result["reason"] == "missing_arg"


# -------- search_ncbi(): no hits -----------------------------------------


def test_search_ncbi_returns_empty_when_esearch_finds_nothing():
    esearch_response = {
        "header": {"type": "esearch", "version": "0.3"},
        "esearchresult": {"count": "0", "retmax": "5", "retstart": "0", "idlist": []},
    }
    with patch(
        "lab_copilot_gateway.ncbi_search._fetch_json", return_value=esearch_response
    ):
        result = search_ncbi({"query": "totally_made_up_gene_xyz123"})
    assert result["ok"] is True
    assert result["result"]["results"] == []
    assert result["result"]["count"] == 0
    assert result["result"]["total_available"] == 0
    assert "No sequences found" in result["result"]["message"]


# -------- search_ncbi(): happy path --------------------------------------


def test_search_ncbi_returns_accession_title_length_for_each_hit():
    esearch_response = {
        "esearchresult": {
            "count": "2",
            "retmax": "5",
            "retstart": "0",
            "idlist": ["111", "222"],
        }
    }
    esummary_response = {
        "result": {
            "uids": ["111", "222"],
            "111": {
                "caption": "NM_000207",
                "accessionversion": "NM_000207.3",
                "title": "Homo sapiens insulin (INS), mRNA",
                "slen": "465",
                "moleculetype": "mRNA",
            },
            "222": {
                "caption": "NM_001185098",
                "accessionversion": "NM_001185098.1",
                "title": "Homo sapiens insulin (INS), transcript variant 3",
                "slen": "644",
                "moleculetype": "mRNA",
            },
        }
    }

    def fake_fetch(url, timeout=15.0):
        if "esearch" in url:
            return esearch_response
        return esummary_response

    with patch("lab_copilot_gateway.ncbi_search._fetch_json", side_effect=fake_fetch):
        result = search_ncbi(
            {
                "query": "INS[gene] AND Homo sapiens[Organism] AND biomol_mRNA[prop]",
                "retmax": 5,
            }
        )

    assert result["ok"] is True
    assert result["result"]["count"] == 2
    assert result["result"]["total_available"] == 2
    assert result["result"]["results"][0]["accession"] == "NM_000207"
    assert result["result"]["results"][0]["title"].startswith("Homo sapiens insulin")
    assert result["result"]["results"][0]["length"] == 465
    assert result["result"]["results"][0]["type"] == "mRNA"
    assert result["result"]["results"][0]["uid"] == "111"


def test_search_ncbi_surfaces_status_code_on_ncbi_http_error():
    from urllib.error import HTTPError

    with patch(
        "lab_copilot_gateway.ncbi_search._fetch_json",
        side_effect=HTTPError(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?...",
            429,
            "Too Many Requests",
            {},
            None,
        ),
    ):
        result = search_ncbi({"query": "INS[gene]"})

    assert result["ok"] is False
    assert result["reason"] == "ncbi_http_error"
    assert result["status_code"] == 429
    assert "429" in result["message"]


def test_search_ncbi_handles_unexpected_response_shape():
    with patch(
        "lab_copilot_gateway.ncbi_search._fetch_json",
        side_effect=ValueError("NCBI returned non-JSON: <html>...rate limit..."),
    ):
        result = search_ncbi({"query": "INS[gene]"})

    assert result["ok"] is False
    assert result["reason"] == "ncbi_parse_error"
    assert "rate limit" in result["message"]


# -------- status_code propagation (OpenCloningAdapterError) --------------


def test_opencloning_adapter_error_carries_status_code():
    from lab_copilot_gateway.opencloning import OpenCloningAdapterError

    err = OpenCloningAdapterError("client_error", "404 Not Found", status_code=404)
    assert err.status_code == 404

    payload = err.to_dict()
    assert payload["reason"] == "client_error"
    assert payload["message"] == "404 Not Found"
    assert payload["status_code"] == 404


def test_opencloning_adapter_error_status_code_optional():
    from lab_copilot_gateway.opencloning import OpenCloningAdapterError

    err = OpenCloningAdapterError("client_error", "no upstream status")
    assert err.status_code is None

    payload = err.to_dict()
    assert "status_code" not in payload
    assert payload == {"reason": "client_error", "message": "no upstream status"}


def test_http_error_has_status_code_attribute():
    """_http_error must attach status_code to the exception so _execute
    can extract it via getattr(exc, 'status_code', None)."""
    from lab_copilot_gateway.opencloning import HttpOpenCloningClient

    # Build a minimal fake response object
    class FakeResp:
        status_code = 404
        reason = "Not Found"

        def json(self):
            return {"detail": "Not Found"}

        @property
        def text(self):
            return '{"detail": "Not Found"}'

    err = HttpOpenCloningClient._http_error(FakeResp(), "/repository_id/snapgene")
    assert err.status_code == 404
