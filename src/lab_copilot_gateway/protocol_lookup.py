"""Protocol lookup service for OpenCloning workflows.

Searches approved eLabFTW protocol/resource entries to retrieve
PCR/assembly conditions backed by lab-approved reagents and manuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lab_copilot_gateway.elabftw import ElabftwClient, _normalize_metadata

# Metadata field keys used in Protocol resource extra_fields.
_METHOD_TYPE = "method_type"
_REAGENT = "reagent_product_name"
_ALIASES = "aliases"
_STATUS = "status"
_APPROVED = "approved_for_use"
_TM_RULE = "annealing_temperature_rule"
_EXT_RATE = "extension_rate"
_CYCLE_CT = "cycle_count_guidance"
_ADDITIVE = "additive_guidance"
_INCUBATION = "assembly_incubation_guidance"
_MANUFACTURER = "manufacturer"
_CATALOG = "catalog_number"

_ALL_REQUIRED = {_METHOD_TYPE, _STATUS}


@dataclass
class ProtocolMatch:
    """A single protocol entry matched by the lookup."""

    item_id: int
    title: str
    method_type: str
    reagent_product_name: str
    status: str
    approved_for_use: bool
    annealing_temperature_rule: str | None = None
    extension_rate: str | None = None
    cycle_count_guidance: str | None = None
    additive_guidance: str | None = None
    assembly_incubation_guidance: str | None = None
    manufacturer: str | None = None
    catalog_number: str | None = None
    aliases: list[str] = field(default_factory=list)
    match_confidence: str = "exact"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict for API responses."""
        return {
            "item_id": self.item_id,
            "title": self.title,
            "method_type": self.method_type,
            "reagent_product_name": self.reagent_product_name,
            "status": self.status,
            "approved_for_use": self.approved_for_use,
            "annealing_temperature_rule": self.annealing_temperature_rule,
            "extension_rate": self.extension_rate,
            "cycle_count_guidance": self.cycle_count_guidance,
            "additive_guidance": self.additive_guidance,
            "assembly_incubation_guidance": self.assembly_incubation_guidance,
            "manufacturer": self.manufacturer,
            "catalog_number": self.catalog_number,
            "aliases": self.aliases,
            "match_confidence": self.match_confidence,
            "warnings": self.warnings,
        }


@dataclass
class ProtocolLookupResult:
    """Result of a protocol lookup."""

    query_method: str
    query_reagent: str | None
    matches: list[ProtocolMatch]
    best_match: ProtocolMatch | None
    fallback_used: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict for API responses."""
        return {
            "query_method": self.query_method,
            "query_reagent": self.query_reagent,
            "matches": [m.to_dict() for m in self.matches],
            "best_match": self.best_match.to_dict() if self.best_match else None,
            "fallback_used": self.fallback_used,
            "warnings": self.warnings,
        }


class ProtocolLookupService:
    """Search approved eLabFTW protocol entries for PCR/assembly conditions.

    Uses the eLabFTW client to search resource items, parse their structured
    metadata, and match against the requested method/reagent.  Only items with
    ``status = "approved"`` AND ``approved_for_use = True`` are selectable.
    """

    def __init__(
        self,
        elabftw_client: ElabftwClient,
        category_name: str = "Protocol",
    ) -> None:
        self._client = elabftw_client
        self._category_name = category_name

    def _build_no_approved_result(
        self,
        method_type: str,
        reagent_name: str | None,
        all_entries: list[dict[str, Any]],
    ) -> ProtocolLookupResult:
        """Build a fallback result when no approved entries are found."""
        msgs: list[str] = []
        if not all_entries:
            msgs.append(
                f"No approved protocol entries found for method_type={method_type!r}"
            )
        else:
            msgs.append(
                f"No approved entries found — all {len(all_entries)} "
                "entries have non-approved status"
            )
        result = ProtocolLookupResult(
            query_method=method_type,
            query_reagent=reagent_name,
            matches=[],
            best_match=None,
            fallback_used=True,
            warnings=msgs,
        )
        # Include the unapproved entries as debug info (with match_confidence
        # set to "method_only" so the caller can see them).
        for e in all_entries:
            result.matches.append(self._entry_to_match(e, "method_only", []))
        return result

    def _select_best_match(
        self,
        approved: list[dict[str, Any]],
        reagent_name: str | None,
        user_aliases: list[str],
    ) -> tuple[list[ProtocolMatch], ProtocolMatch | None]:
        """Score and rank approved entries, returning matches and best match."""
        matches: list[ProtocolMatch] = []
        best: ProtocolMatch | None = None

        for entry in approved:
            matched_conf, match_warnings = self._score_match(
                entry, reagent_name, user_aliases
            )
            pm = self._entry_to_match(entry, matched_conf, match_warnings)
            matches.append(pm)
            if best is None:
                best = pm

        return matches, best

    def lookup(
        self,
        method_type: str,
        reagent_name: str | None = None,
        aliases: list[str] | None = None,
    ) -> ProtocolLookupResult:
        """Search for approved protocol entries.

        Hierarchy:
        1. Exact reagent name match (status=approved)
        2. Alias match (status=approved)
        3. Method-type-only match (status=approved)
        4. Fallback: no match, return generic rules with warning
        """
        if not method_type:
            return ProtocolLookupResult(
                query_method=method_type or "",
                query_reagent=reagent_name,
                matches=[],
                best_match=None,
                fallback_used=True,
                warnings=["method_type is required"],
            )

        # Search eLabFTW items matching the method type.
        try:
            items = self._client.search_items(method_type, limit=50)
        except Exception as exc:
            return ProtocolLookupResult(
                query_method=method_type,
                query_reagent=reagent_name,
                matches=[],
                best_match=None,
                fallback_used=True,
                warnings=[f"eLabFTW search failed: {exc}"],
            )

        # Parse all returned items into protocol entries.
        all_entries: list[dict[str, Any]] = []
        for item in items:
            entry = self._parse_item(item)
            if entry is not None:
                # Post-filter by exact method_type — text search can return
                # items whose body mentions the term without being the same
                # method (e.g. "ligation" matches "restriction_ligation").
                entry_method = str(entry.get(_METHOD_TYPE, "")).strip().lower()
                if entry_method == method_type.strip().lower():
                    all_entries.append(entry)

        # Filter to approved-only.
        approved = [e for e in all_entries if self._is_approved(e)]

        if not approved:
            return self._build_no_approved_result(
                method_type, reagent_name, all_entries
            )

        # Score and rank approved entries.
        user_aliases = aliases or []
        matches, best = self._select_best_match(approved, reagent_name, user_aliases)

        fallback_used = not matches or not any(
            m.match_confidence in ("exact", "alias") for m in matches
        )

        return ProtocolLookupResult(
            query_method=method_type,
            query_reagent=reagent_name,
            matches=matches,
            best_match=best,
            fallback_used=fallback_used,
            warnings=[],
        )

    def validate_corpus(self) -> list[dict[str, str]]:
        """Validate all protocol entries in the corpus.

        Returns a list of validation issues:
        - missing required fields
        - duplicate reagent aliases
        - deprecated entries matching common queries
        - entries not marked approved
        """
        issues: list[dict[str, str]] = []

        # Gather items by searching for common method types.
        all_items = self._gather_protocol_candidates()
        if all_items is None:
            issues.append(
                {
                    "severity": "error",
                    "message": "eLabFTW search failed during validation",
                }
            )
            return issues

        seen_aliases: dict[str, int] = {}
        seen_titles: dict[str, int] = {}

        for item in all_items:
            self._validate_corpus_item(item, seen_aliases, seen_titles, issues)

        return issues

    def _validate_corpus_item(
        self,
        item: dict[str, Any],
        seen_aliases: dict[str, int],
        seen_titles: dict[str, int],
        issues: list[dict[str, str]],
    ) -> None:
        """Validate a single corpus item, appending issues to the list."""
        pid = int(item.get("id", 0))
        title = str(item.get("title", ""))
        entry = self._parse_item(item)

        # Even if entry is None (no method_type), check whether the item
        # has extra_fields at all — that signals a protocol-like item
        # that is missing required fields.
        if entry is None:
            self._check_unparsed_item_fields(item, pid, title, issues)
            return

        # Check required fields.
        self._check_required_fields(entry, pid, title, issues)

        # Check for non-approved statuses and deprecated entries.
        self._check_entry_status(entry, pid, title, issues)

        # Check for duplicate aliases.
        self._check_duplicate_aliases(entry, pid, title, seen_aliases, issues)

        # Check for duplicate titles.
        self._check_duplicate_title(title, pid, seen_titles, issues)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_unparsed_item_fields(
        item: dict[str, Any],
        pid: int,
        title: str,
        issues: list[dict[str, str]],
    ) -> None:
        """Check for protocol-like items that are missing required fields."""
        metadata = _normalize_metadata(item.get("metadata"))
        if metadata and isinstance(metadata.get("extra_fields"), dict):
            ef = metadata["extra_fields"]
            missing = sorted(k for k in _ALL_REQUIRED if k not in ef)
            if missing:
                issues.append(
                    {
                        "item_id": str(pid),
                        "severity": "error",
                        "message": (
                            f"Missing required field(s): {missing} (title: {title})"
                        ),
                    }
                )

    @staticmethod
    def _check_required_fields(
        entry: dict[str, Any],
        pid: int,
        title: str,
        issues: list[dict[str, str]],
    ) -> None:
        """Check that all required fields are present in the entry."""
        missing = sorted(k for k in _ALL_REQUIRED if not entry.get(k))
        if missing:
            issues.append(
                {
                    "item_id": str(pid),
                    "severity": "error",
                    "message": (
                        f"Missing required field(s): {missing} (title: {title})"
                    ),
                }
            )

    @staticmethod
    def _check_entry_status(
        entry: dict[str, Any],
        pid: int,
        title: str,
        issues: list[dict[str, str]],
    ) -> None:
        """Check for non-approved and deprecated entry statuses."""
        status = str(entry.get(_STATUS, "")).strip().lower()
        if status and status != "approved":
            issues.append(
                {
                    "item_id": str(pid),
                    "severity": "warning",
                    "message": (
                        f"Entry has status={status!r} — "
                        f"not selectable by copilot (title: {title})"
                    ),
                }
            )

        # Flag deprecated entries that still match common queries.
        if status == "deprecated":
            reagent = entry.get(_REAGENT, "")
            aliases_str = entry.get(_ALIASES, "")
            if reagent or aliases_str:
                issues.append(
                    {
                        "item_id": str(pid),
                        "severity": "warning",
                        "message": (
                            f"Deprecated entry may match common queries "
                            f"(reagent: {reagent!r}, aliases: {aliases_str!r}) "
                            f"(title: {title})"
                        ),
                    }
                )

    @staticmethod
    def _check_duplicate_aliases(
        entry: dict[str, Any],
        pid: int,
        title: str,
        seen_aliases: dict[str, int],
        issues: list[dict[str, str]],
    ) -> None:
        """Check for duplicate reagent aliases across entries."""
        aliases_raw = entry.get(_ALIASES, "")
        alias_list = [a.strip() for a in aliases_raw.split(",") if a.strip()]
        for alias in alias_list:
            alias_lower = alias.lower()
            if alias_lower in seen_aliases:
                other_pid = seen_aliases[alias_lower]
                issues.append(
                    {
                        "item_id": str(pid),
                        "severity": "warning",
                        "message": (
                            f"Duplicate alias {alias!r} also used in "
                            f"item #{other_pid} (title: {title})"
                        ),
                    }
                )
            else:
                seen_aliases[alias_lower] = pid

    @staticmethod
    def _check_duplicate_title(
        title: str,
        pid: int,
        seen_titles: dict[str, int],
        issues: list[dict[str, str]],
    ) -> None:
        """Check for duplicate entry titles."""
        title_lower = title.lower().strip()
        if title_lower:
            if title_lower in seen_titles:
                other_pid = seen_titles[title_lower]
                issues.append(
                    {
                        "item_id": str(pid),
                        "severity": "warning",
                        "message": (
                            f"Duplicate title matches item #{other_pid} "
                            f"(title: {title})"
                        ),
                    }
                )
            else:
                seen_titles[title_lower] = pid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gather_protocol_candidates(self) -> list[dict[str, Any]] | None:
        """Search eLabFTW for items that may be protocol entries.

        Returns None if ALL searches fail (network error, auth, etc.).
        Returns an item-level deduplicated list otherwise (possibly empty).
        """
        seen_ids: set[int] = set()
        all_items: list[dict[str, Any]] = []
        # Reason: search several common method types to maximise coverage
        # across different protocol categories.
        search_terms = ("pcr", "gibson", "ligation", "assembly", "restriction")
        any_succeeded = False
        for term in search_terms:
            try:
                results = self._client.search_items(term, limit=100)
                any_succeeded = True
                for r in results:
                    rid = r.get("id")
                    if rid is not None and rid not in seen_ids:
                        seen_ids.add(rid)
                        all_items.append(r)
            except Exception:
                pass
        return all_items if any_succeeded else None

    @staticmethod
    def _parse_item(item: dict[str, Any]) -> dict[str, Any] | None:
        """Parse an eLabFTW item dict into a structured protocol entry.

        Returns None if the item has no recognizable protocol metadata
        (no ``method_type`` in extra_fields).
        """
        metadata = _normalize_metadata(item.get("metadata"))
        if metadata is None:
            return None
        extra_fields = metadata.get("extra_fields")
        if not isinstance(extra_fields, dict):
            return None
        if _METHOD_TYPE not in extra_fields:
            return None

        entry: dict[str, Any] = {}
        for key in (
            _METHOD_TYPE,
            _REAGENT,
            _ALIASES,
            _STATUS,
            _APPROVED,
            _TM_RULE,
            _EXT_RATE,
            _CYCLE_CT,
            _ADDITIVE,
            _INCUBATION,
            _MANUFACTURER,
            _CATALOG,
        ):
            ef = extra_fields.get(key)
            if isinstance(ef, dict):
                entry[key] = ef.get("value", "")
            else:
                entry[key] = ef if ef is not None else ""

        entry["_item_id"] = int(item.get("id", 0))
        entry["_title"] = str(item.get("title", ""))
        return entry

    @staticmethod
    def _is_approved(entry: dict[str, Any]) -> bool:
        """Check if a parsed entry is approved for use."""
        status = str(entry.get(_STATUS, "")).strip().lower()
        approved_flag = entry.get(_APPROVED, False)
        if isinstance(approved_flag, str):
            approved_flag = approved_flag.strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        return status == "approved" and bool(approved_flag)

    @staticmethod
    def _match_reagent_or_aliases(
        reagent_name: str,
        user_aliases: list[str],
        entry_reagent: str,
        entry_aliases: list[str],
    ) -> str | None:
        """Check if reagent_name or user aliases match the entry.

        Returns ``"exact"``, ``"alias"``, or ``None`` (no match).
        """
        cleaned_reagent = reagent_name.strip().lower()

        # 1. Exact match against reagent_product_name.
        if entry_reagent == cleaned_reagent:
            return "exact"

        # 2. Reagent name matches an alias in the entry.
        if cleaned_reagent in entry_aliases:
            return "alias"

        # 3. One of the user-supplied aliases matches.
        for ua in user_aliases:
            ua_lower = ua.strip().lower()
            if ua_lower == entry_reagent:
                return "alias"
            if ua_lower in entry_aliases:
                return "alias"

        return None

    @staticmethod
    def _score_match(
        entry: dict[str, Any],
        reagent_name: str | None,
        user_aliases: list[str],
    ) -> tuple[str, list[str]]:
        """Score how well an entry matches the query.

        Returns ``(match_confidence, warnings)``.
        """
        match_warnings: list[str] = []

        if reagent_name:
            entry_reagent = entry.get(_REAGENT, "").strip().lower()
            entry_aliases_str = entry.get(_ALIASES, "")
            entry_aliases = [
                a.strip().lower() for a in entry_aliases_str.split(",") if a.strip()
            ]

            matched = ProtocolLookupService._match_reagent_or_aliases(
                reagent_name, user_aliases, entry_reagent, entry_aliases
            )
            if matched is not None:
                return matched, match_warnings

            match_warnings.append(
                f"No exact or alias match for {reagent_name!r} — "
                "using first approved entry for this method type"
            )

        return "method_only", match_warnings

    @staticmethod
    def _entry_to_match(
        entry: dict[str, Any],
        confidence: str,
        match_warnings: list[str],
    ) -> ProtocolMatch:
        """Convert a parsed entry dict to a ProtocolMatch."""
        aliases_str = entry.get(_ALIASES, "")
        aliases_list = [a.strip() for a in aliases_str.split(",") if a.strip()]

        def _str(val: Any) -> str | None:
            if val is None or val == "":
                return None
            return str(val)

        return ProtocolMatch(
            item_id=int(entry.get("_item_id", 0)),
            title=str(entry.get("_title", "")),
            method_type=str(entry.get(_METHOD_TYPE, "")),
            reagent_product_name=str(entry.get(_REAGENT, "")),
            status=str(entry.get(_STATUS, "")),
            approved_for_use=ProtocolLookupService._is_approved(entry),
            annealing_temperature_rule=_str(entry.get(_TM_RULE)),
            extension_rate=_str(entry.get(_EXT_RATE)),
            cycle_count_guidance=_str(entry.get(_CYCLE_CT)),
            additive_guidance=_str(entry.get(_ADDITIVE)),
            assembly_incubation_guidance=_str(entry.get(_INCUBATION)),
            manufacturer=_str(entry.get(_MANUFACTURER)),
            catalog_number=_str(entry.get(_CATALOG)),
            aliases=aliases_list,
            match_confidence=confidence,
            warnings=match_warnings,
        )
