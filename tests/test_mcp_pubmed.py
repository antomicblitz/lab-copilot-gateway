"""Unit tests for the PubMed MCP normalizer (Slice 4).

Covers:
    * normalize_article — all fields present
    * normalize_article — missing fields (null / [] defaults)
    * normalize_article — abstract truncation to 500 chars
    * normalize_article — PubMed URL synthesis from PMID
    * normalize_search_result — typical ESearch+ESummary shape
    * normalize_search_result — unrecognized shape → empty result
    * normalize_fetch_result — fuller abstracts, still capped
"""

from __future__ import annotations

from lab_copilot_gateway.mcp_pubmed import (
    MAX_ABSTRACT_CHARS,
    normalize_article,
    normalize_fetch_result,
    normalize_search_result,
)


# ---------------------------------------------------------------------------
# normalize_article
# ---------------------------------------------------------------------------


def test_normalize_article_all_fields_present() -> None:
    raw = {
        "pmid": "12345678",
        "title": "Test Title",
        "authors": [
            {"lastName": "Smith", "firstName": "John"},
            {"lastName": "Doe", "firstName": "Jane", "initials": "JD"},
        ],
        "source": "Nature",
        "pubDate": "2024 Feb 01",
        "doi": "10.1038/test",
        "pubmedUrl": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
    }
    result = normalize_article(raw)
    assert result["pmid"] == "12345678"
    assert result["title"] == "Test Title"
    assert result["authors"] == ["John Smith", "Jane Doe"]
    assert result["journal"] == "Nature"
    assert result["year"] == 2024
    assert result["doi"] == "10.1038/test"
    assert result["url"] == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    assert result["abstract"] is None  # search results don't carry abstracts


def test_normalize_article_missing_fields_default_to_null() -> None:
    """Every contract key must be present, even when the raw data is sparse."""
    result = normalize_article({"pmid": "1"})
    assert result["pmid"] == "1"
    assert result["title"] is None
    assert result["authors"] == []
    assert result["journal"] is None
    assert result["year"] is None
    assert result["doi"] is None
    assert result["abstract"] is None
    assert result["url"] == "https://pubmed.ncbi.nlm.nih.gov/1/"


def test_normalize_article_abstract_truncation() -> None:
    long_abstract = "X" * 600
    result = normalize_article({"pmid": "1", "abstractText": long_abstract})
    assert result["abstract"] == "X" * MAX_ABSTRACT_CHARS
    assert len(result["abstract"]) == MAX_ABSTRACT_CHARS


def test_normalize_article_abstract_within_limit_unchanged() -> None:
    short_abstract = "Short abstract."
    result = normalize_article({"pmid": "1", "abstractText": short_abstract})
    assert result["abstract"] == short_abstract


def test_normalize_article_url_synthesized_from_pmid() -> None:
    """When pubmedUrl is missing, synthesize from PMID."""
    result = normalize_article({"pmid": "99999999"})
    assert result["url"] == "https://pubmed.ncbi.nlm.nih.gov/99999999/"


def test_normalize_article_url_from_raw_preferred() -> None:
    """When pubmedUrl is present, use it directly."""
    result = normalize_article(
        {"pmid": "1", "pubmedUrl": "https://pubmed.ncbi.nlm.nih.gov/1/"}
    )
    assert result["url"] == "https://pubmed.ncbi.nlm.nih.gov/1/"


def test_normalize_article_authors_flat_string() -> None:
    """Search tool returns authors as a flat string."""
    result = normalize_article({"pmid": "1", "authors": "Smith J, Doe J"})
    assert result["authors"] == ["Smith J, Doe J"]


def test_normalize_article_structured_authors() -> None:
    """Fetch tool returns structured author records."""
    result = normalize_article(
        {
            "pmid": "1",
            "authors": [
                {"lastName": "Smith", "firstName": "John"},
                {"collectiveName": "The Consortium"},
                {"initials": "XY"},
            ],
        }
    )
    assert result["authors"] == ["John Smith", "The Consortium", "XY"]


def test_normalize_article_year_from_pub_date() -> None:
    """Extract year from search tool's pubDate field."""
    result = normalize_article({"pmid": "1", "pubDate": "2023 Nov 15"})
    assert result["year"] == 2023


def test_normalize_article_year_from_journal_info() -> None:
    """Extract year from fetch tool's journalInfo.publicationDate."""
    result = normalize_article(
        {
            "pmid": "1",
            "journalInfo": {
                "title": "Nature",
                "publicationDate": {"year": "2022", "month": "01"},
            },
        }
    )
    assert result["year"] == 2022


def test_normalize_article_journal_from_source() -> None:
    """Search tool returns journal as 'source'."""
    result = normalize_article({"pmid": "1", "source": "Nature"})
    assert result["journal"] == "Nature"


def test_normalize_article_journal_from_journal_info() -> None:
    """Fetch tool returns journal via journalInfo.title."""
    result = normalize_article(
        {"pmid": "1", "journalInfo": {"title": "Science"}, "source": "Nature"}
    )
    assert result["journal"] == "Science"


# ---------------------------------------------------------------------------
# normalize_search_result
# ---------------------------------------------------------------------------


def test_normalize_search_result_typical_shape() -> None:
    """ESearch+ESummary shaped structuredContent from the PubMed MCP server."""
    raw = {
        "query": "CRISPR",
        "offset": 0,
        "pmids": ["12345678", "87654321"],
        "summaries": [
            {
                "pmid": "12345678",
                "title": "Article One",
                "authors": "Smith J",
                "source": "Nature",
                "pubDate": "2024 Jan",
                "doi": "10.1038/one",
                "pubmedUrl": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            },
            {
                "pmid": "87654321",
                "title": "Article Two",
                "authors": "Doe J",
                "source": "Science",
                "pubDate": "2023",
                "pubmedUrl": "https://pubmed.ncbi.nlm.nih.gov/87654321/",
            },
        ],
        "searchUrl": "https://pubmed.ncbi.nlm.nih.gov/?term=CRISPR",
        "enriched": {"effectiveQuery": "CRISPR", "totalCount": 1000},
    }
    result = normalize_search_result(raw)
    assert result["total"] == 2
    articles = result["articles"]
    assert len(articles) == 2
    assert articles[0]["pmid"] == "12345678"
    assert articles[0]["title"] == "Article One"
    assert articles[1]["pmid"] == "87654321"


def test_normalize_search_result_unrecognized_shape() -> None:
    """Defensive: non-dict or missing summaries → empty result, no crash."""
    assert normalize_search_result(None) == {"articles": [], "total": 0}
    assert normalize_search_result([]) == {"articles": [], "total": 0}
    assert normalize_search_result("garbage") == {"articles": [], "total": 0}
    assert normalize_search_result({}) == {"articles": [], "total": 0}
    assert normalize_search_result({"summaries": None}) == {"articles": [], "total": 0}
    assert normalize_search_result({"summaries": "not-a-list"}) == {
        "articles": [],
        "total": 0,
    }


def test_normalize_search_result_empty_summaries() -> None:
    """Zero results is valid — returns empty articles."""
    result = normalize_search_result({"query": "nothing", "pmids": [], "summaries": []})
    assert result == {"articles": [], "total": 0}


def test_normalize_search_result_skips_non_dict_entries() -> None:
    """Non-dict entries in summaries are silently skipped."""
    result = normalize_search_result(
        {
            "summaries": [
                {"pmid": "1", "title": "Good"},
                "bad-entry",
                None,
                42,
            ]
        }
    )
    assert result["total"] == 1
    assert result["articles"][0]["pmid"] == "1"


# ---------------------------------------------------------------------------
# normalize_fetch_result
# ---------------------------------------------------------------------------


def test_normalize_fetch_result_typical_shape() -> None:
    """Fetch tool returns detailed articles with abstracts."""
    raw = {
        "articles": [
            {
                "pmid": "12345678",
                "title": "Full Article",
                "abstractText": "This is the full abstract of the article.",
                "authors": [
                    {"firstName": "John", "lastName": "Smith"},
                    {"firstName": "Jane", "lastName": "Doe"},
                ],
                "journalInfo": {
                    "title": "Nature",
                    "publicationDate": {"year": "2024"},
                },
                "doi": "10.1038/full",
                "pubmedUrl": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            },
        ],
        "totalReturned": 1,
    }
    result = normalize_fetch_result(raw)
    assert result["total"] == 1
    article = result["articles"][0]
    assert article["pmid"] == "12345678"
    assert article["abstract"] == "This is the full abstract of the article."
    assert article["authors"] == ["John Smith", "Jane Doe"]
    assert article["journal"] == "Nature"
    assert article["year"] == 2024


def test_normalize_fetch_result_fuller_abstract_still_capped() -> None:
    """Fetch abstracts are still capped at MAX_ABSTRACT_CHARS."""
    long_abstract = "A" * 600
    raw = {
        "articles": [
            {
                "pmid": "1",
                "abstractText": long_abstract,
            }
        ],
        "totalReturned": 1,
    }
    result = normalize_fetch_result(raw)
    assert len(result["articles"][0]["abstract"]) == MAX_ABSTRACT_CHARS


def test_normalize_fetch_result_unrecognized_shape() -> None:
    """Defensive: non-dict or missing articles → empty result."""
    assert normalize_fetch_result(None) == {"articles": [], "total": 0}
    assert normalize_fetch_result([]) == {"articles": [], "total": 0}
    assert normalize_fetch_result({}) == {"articles": [], "total": 0}
    assert normalize_fetch_result({"articles": None}) == {"articles": [], "total": 0}


def test_normalize_fetch_result_empty_articles() -> None:
    result = normalize_fetch_result({"articles": [], "totalReturned": 0})
    assert result == {"articles": [], "total": 0}
