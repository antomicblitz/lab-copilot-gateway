"""Deterministic validation bundle for OpenCloning writeback approval.

Built after simulation completes and before formal approval.  Structural
blockers prevent approval.  Warnings are visible but non-blocking.

The validation bundle is the deterministic gate between "simulation done" and
"user can approve writeback."  It ensures the LLM cannot approve an incomplete
cloning package — the bundle is built from captured facts (sequences, sources,
primers, artifacts, step manifests), not from LLM output, so the LLM cannot
influence validation results.

Blockers (``is_approved = False`` if any fail):

    * final_construct_exists  — a final construct sequence with file_content
    * genbank_artifact_exists — GenBank artifact content available for writeback

Warnings (visible, non-blocking):

    * post_assembly_annotation — annotation attempted/completed
    * history_json_attempted   — cloning history data accumulated
    * primer_details           — Tm/sequence available for each primer
    * primer_heterodimer       — heterodimer checked for primer pairs
    * protocol_lookup          — approved protocol found (no fallback)
    * preview_refs             — OVE preview refs generated
    * notebook_summary_draft   — notebook summary draft generated (T10)
    * audit_report_draft       — Markdown audit report draft generated (T10)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationLevel(Enum):
    """Severity level of a validation check."""

    BLOCKER = "blocker"  # Prevents approval
    WARNING = "warning"  # Visible but does not block


@dataclass
class ValidationCheck:
    """A single validation check result."""

    check_id: str  # e.g. "final_construct_exists"
    label: str  # Human-readable: "Final construct exists"
    level: ValidationLevel
    status: str  # "pass" | "fail" | "not_checked" | "skipped"
    detail: str  # Explanation or error message
    data: dict[str, Any] | None = None  # Optional structured data


@dataclass
class ValidationBundle:
    """Complete validation result for a cloning run."""

    run_id: str
    checks: list[ValidationCheck] = field(default_factory=list)
    blockers: list[ValidationCheck] = field(default_factory=list)
    warnings: list[ValidationCheck] = field(default_factory=list)
    is_approved: bool = False  # True if no blockers

    def add_check(self, check: ValidationCheck) -> None:
        """Add a check result and route it to blockers/warnings.

        A BLOCKER with status "fail" prevents approval.  A WARNING with any
        non-"pass" status is surfaced as a visible warning.  "skipped" checks
        (not applicable) are neither blockers nor warnings.
        """
        self.checks.append(check)
        if check.level == ValidationLevel.BLOCKER and check.status == "fail":
            self.blockers.append(check)
        elif check.level == ValidationLevel.WARNING and check.status not in (
            "pass",
            "skipped",
        ):
            self.warnings.append(check)

    def finalize(self) -> None:
        """Compute final approval status — approved iff no blockers.

        This is the absolute gate: if any blocker failed, ``is_approved`` is
        ``False`` and no downstream override can flip it.
        """
        self.is_approved = len(self.blockers) == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict matching the bundle schema."""
        return {
            "schema": "opencloning_validation_bundle_v1",
            "run_id": self.run_id,
            "is_approved": self.is_approved,
            "blocker_count": len(self.blockers),
            "warning_count": len(self.warnings),
            "checks": [
                {
                    "check_id": c.check_id,
                    "label": c.label,
                    "level": c.level.value,
                    "status": c.status,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
            "blockers": [
                {"check_id": c.check_id, "label": c.label, "detail": c.detail}
                for c in self.blockers
            ],
            "warnings": [
                {"check_id": c.check_id, "label": c.label, "detail": c.detail}
                for c in self.warnings
            ],
        }


class ValidationBundleBuilder:
    """Builds a validation bundle from cloning run state.

    Each check method records a single validation result.  After all checks
    are recorded, ``build()`` finalizes the bundle (computes ``is_approved``).
    """

    def __init__(self, run_id: str) -> None:
        self.bundle = ValidationBundle(run_id=run_id)

    def check_final_construct(
        self,
        has_final_sequence: bool,
        sequence_id: int | None = None,
        sequence_length: int | None = None,
    ) -> None:
        """Check: final construct sequence exists (BLOCKER)."""
        if has_final_sequence:
            if sequence_id is not None:
                length_str = f", {sequence_length} bp" if sequence_length else ""
                detail = f"Final construct sequence id={sequence_id}{length_str}"
            else:
                detail = "Final construct sequence present"
            self.bundle.add_check(
                ValidationCheck(
                    check_id="final_construct_exists",
                    label="Final construct exists",
                    level=ValidationLevel.BLOCKER,
                    status="pass",
                    detail=detail,
                )
            )
        else:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="final_construct_exists",
                    label="Final construct exists",
                    level=ValidationLevel.BLOCKER,
                    status="fail",
                    detail="No final construct sequence found in the cloning run",
                )
            )

    def check_genbank_artifact(self, has_genbank: bool) -> None:
        """Check: final GenBank artifact exists (BLOCKER)."""
        if has_genbank:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="genbank_artifact_exists",
                    label="GenBank artifact exists",
                    level=ValidationLevel.BLOCKER,
                    status="pass",
                    detail="GenBank artifact content is available for writeback",
                )
            )
        else:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="genbank_artifact_exists",
                    label="GenBank artifact exists",
                    level=ValidationLevel.BLOCKER,
                    status="fail",
                    detail="No GenBank artifact content available for writeback",
                )
            )

    def check_annotation(
        self, annotation_attempted: bool, annotation_completed: bool
    ) -> None:
        """Check: post-assembly annotation attempted/completed (WARNING).

        - pass if annotation was attempted and completed
        - warning (fail) if attempted but not completed
        - warning (not_checked) if not attempted
        """
        if annotation_attempted and annotation_completed:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="post_assembly_annotation",
                    label="Post-assembly annotation",
                    level=ValidationLevel.WARNING,
                    status="pass",
                    detail="Post-assembly annotation completed",
                )
            )
        elif annotation_attempted and not annotation_completed:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="post_assembly_annotation",
                    label="Post-assembly annotation",
                    level=ValidationLevel.WARNING,
                    status="fail",
                    detail="Annotation was attempted but did not complete successfully",
                )
            )
        else:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="post_assembly_annotation",
                    label="Post-assembly annotation",
                    level=ValidationLevel.WARNING,
                    status="not_checked",
                    detail="Post-assembly annotation was not attempted",
                )
            )

    def check_history_json(self, history_generated: bool) -> None:
        """Check: _history.json generation attempted (WARNING).

        The history JSON is built from accumulated strategy sequences and
        sources during writeback.  If no strategy data was accumulated, the
        history JSON cannot be generated.
        """
        if history_generated:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="history_json_attempted",
                    label="Cloning history JSON",
                    level=ValidationLevel.WARNING,
                    status="pass",
                    detail="Cloning history data accumulated for _history.json generation",
                )
            )
        else:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="history_json_attempted",
                    label="Cloning history JSON",
                    level=ValidationLevel.WARNING,
                    status="not_checked",
                    detail=(
                        "No cloning history data accumulated — "
                        "_history.json cannot be generated"
                    ),
                )
            )

    def check_primer_details(self, primer_count: int, details_available: int) -> None:
        """Check: primer details attempted for each primer (WARNING)."""
        if primer_count == 0:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="primer_details",
                    label="Primer details",
                    level=ValidationLevel.WARNING,
                    status="skipped",
                    detail="No primers in the cloning run",
                )
            )
        elif details_available >= primer_count:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="primer_details",
                    label="Primer details",
                    level=ValidationLevel.WARNING,
                    status="pass",
                    detail=f"Primer details available for all {primer_count} primer(s)",
                )
            )
        else:
            missing = primer_count - details_available
            self.bundle.add_check(
                ValidationCheck(
                    check_id="primer_details",
                    label="Primer details",
                    level=ValidationLevel.WARNING,
                    status="fail",
                    detail=(
                        f"Primer details missing for {missing} of {primer_count} "
                        f"primer(s) — Tm/sequence not available"
                    ),
                )
            )

    def check_primer_heterodimer(self, primer_pairs: int, checked_pairs: int) -> None:
        """Check: primer heterodimer attempted for primer pairs (WARNING)."""
        if primer_pairs == 0:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="primer_heterodimer",
                    label="Primer heterodimer",
                    level=ValidationLevel.WARNING,
                    status="skipped",
                    detail="Fewer than 2 primers — heterodimer check not applicable",
                )
            )
        elif checked_pairs >= primer_pairs:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="primer_heterodimer",
                    label="Primer heterodimer",
                    level=ValidationLevel.WARNING,
                    status="pass",
                    detail=f"Heterodimer checked for all {primer_pairs} primer pair(s)",
                )
            )
        else:
            unchecked = primer_pairs - checked_pairs
            self.bundle.add_check(
                ValidationCheck(
                    check_id="primer_heterodimer",
                    label="Primer heterodimer",
                    level=ValidationLevel.WARNING,
                    status="fail",
                    detail=(
                        f"Heterodimer not checked for {unchecked} of {primer_pairs} "
                        f"primer pair(s)"
                    ),
                )
            )

    def check_protocol_lookup(self, protocol_found: bool, fallback_used: bool) -> None:
        """Check: protocol lookup status (WARNING)."""
        if protocol_found and not fallback_used:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="protocol_lookup",
                    label="Protocol lookup",
                    level=ValidationLevel.WARNING,
                    status="pass",
                    detail="Approved protocol found for the cloning method",
                )
            )
        elif fallback_used:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="protocol_lookup",
                    label="Protocol lookup",
                    level=ValidationLevel.WARNING,
                    status="fail",
                    detail="No approved protocol found — fallback conditions used",
                )
            )
        else:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="protocol_lookup",
                    label="Protocol lookup",
                    level=ValidationLevel.WARNING,
                    status="not_checked",
                    detail="Protocol lookup was not performed",
                )
            )

    def check_preview_refs(self, steps_with_previews: int, total_steps: int) -> None:
        """Check: OVE preview refs generated where possible (non-blocking)."""
        if steps_with_previews > 0:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="preview_refs",
                    label="OVE preview refs",
                    level=ValidationLevel.WARNING,
                    status="pass",
                    detail=(
                        f"Preview refs generated for {steps_with_previews} "
                        f"of {total_steps} step(s)"
                    ),
                )
            )
        else:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="preview_refs",
                    label="OVE preview refs",
                    level=ValidationLevel.WARNING,
                    status="not_checked",
                    detail="No OVE preview refs generated",
                )
            )

    def check_notebook_summary(self, draft_generated: bool) -> None:
        """Check: notebook summary draft generated (WARNING).

        The notebook summary draft is produced by a downstream step (T10).
        Until then, this check reports a warning that the draft is missing.
        """
        if draft_generated:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="notebook_summary_draft",
                    label="Notebook summary draft",
                    level=ValidationLevel.WARNING,
                    status="pass",
                    detail="Notebook summary draft generated",
                )
            )
        else:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="notebook_summary_draft",
                    label="Notebook summary draft",
                    level=ValidationLevel.WARNING,
                    status="not_checked",
                    detail="Notebook summary draft not yet generated",
                )
            )

    def check_audit_report(self, draft_generated: bool) -> None:
        """Check: Markdown audit report draft generated (WARNING).

        The audit report draft is produced by a downstream step (T10).
        Until then, this check reports a warning that the draft is missing.
        """
        if draft_generated:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="audit_report_draft",
                    label="Audit report draft",
                    level=ValidationLevel.WARNING,
                    status="pass",
                    detail="Audit report draft generated",
                )
            )
        else:
            self.bundle.add_check(
                ValidationCheck(
                    check_id="audit_report_draft",
                    label="Audit report draft",
                    level=ValidationLevel.WARNING,
                    status="not_checked",
                    detail="Audit report draft not yet generated",
                )
            )

    def build(self) -> ValidationBundle:
        """Finalize and return the validation bundle."""
        self.bundle.finalize()
        return self.bundle


# --- standalone builder ----------------------------------------------------


def build_validation_bundle(
    run_id: str,
    sequences: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    primers: list[dict[str, Any]],
    artifacts: dict[str, Any] | None,
    step_manifests: list[dict[str, Any]],
    protocol_result: dict[str, Any] | None,
    preview_refs_count: int,
    notebook_draft_generated: bool = False,
    audit_draft_generated: bool = False,
) -> ValidationBundle:
    """Build a validation bundle from the cloning run state.

    Examines captured facts (sequences, sources, primers, artifacts, step
    manifests, protocol result, preview refs) and produces a deterministic
    validation bundle.  The LLM cannot influence the result — only captured
    state is examined.

    Args:
        run_id: Identifier for the cloning run.
        sequences: Accumulated strategy sequences (each may have
            ``file_content``, ``id``, ``length``, ``feature_count``).
        sources: Accumulated strategy sources (provenance chain).
        primers: Accumulated strategy primers (each may have ``sequence``,
            ``tm``, ``melting_temperature``, etc.).
        artifacts: Artifact payload dict — either the normalized artifacts
            payload (with an ``artifacts`` list) or approval_args (with
            ``artifact_content`` / ``artifact_id``).  ``None`` means no
            artifact.
        step_manifests: List of step manifest dicts (from
            ``StepManifest.to_dict()``).  May be empty if not wired.
        protocol_result: Protocol lookup result dict (from
            ``ProtocolLookupResult.to_dict()``).  ``None`` if no lookup.
        preview_refs_count: Number of OVE preview refs generated.
        notebook_draft_generated: Whether the notebook summary draft has
            been generated (default ``False`` — T10 sets ``True``).
        audit_draft_generated: Whether the Markdown audit report draft has
            been generated (default ``False`` — T10 sets ``True``).
    """
    builder = ValidationBundleBuilder(run_id=run_id)

    # 1. Final construct exists (BLOCKER).
    final_seq = _find_final_construct(sequences, sources)
    has_final = final_seq is not None and bool(
        isinstance(final_seq, dict) and final_seq.get("file_content")
    )
    seq_id = _int_or_none(final_seq.get("id")) if final_seq else None
    seq_len = _int_or_none(final_seq.get("length")) if final_seq else None
    builder.check_final_construct(has_final, seq_id, seq_len)

    # 2. GenBank artifact exists (BLOCKER).
    has_genbank = _has_genbank_artifact(artifacts)
    builder.check_genbank_artifact(has_genbank)

    # 3. Post-assembly annotation (WARNING).
    annotation_attempted, annotation_completed = _detect_annotation(
        step_manifests, final_seq
    )
    builder.check_annotation(annotation_attempted, annotation_completed)

    # 4. History JSON (WARNING).
    history_generated = bool(sequences or sources)
    builder.check_history_json(history_generated)

    # 5. Primer details (WARNING).
    primer_count = len(primers)
    details_available = _count_primers_with_details(primers)
    builder.check_primer_details(primer_count, details_available)

    # 6. Primer heterodimer (WARNING).
    primer_pairs = _count_primer_pairs(primer_count)
    checked_pairs = _count_checked_heterodimer_pairs(
        step_manifests, primers, primer_pairs
    )
    builder.check_primer_heterodimer(primer_pairs, checked_pairs)

    # 7. Protocol lookup (WARNING).
    protocol_found, fallback_used = _detect_protocol_status(protocol_result)
    builder.check_protocol_lookup(protocol_found, fallback_used)

    # 8. Preview refs (non-blocking).
    total_steps = max(len(step_manifests), len(sequences), 1)
    builder.check_preview_refs(preview_refs_count, total_steps)

    # 9. Notebook summary draft (WARNING — produced by T10).
    builder.check_notebook_summary(draft_generated=notebook_draft_generated)

    # 10. Audit report draft (WARNING — produced by T10).
    builder.check_audit_report(draft_generated=audit_draft_generated)

    return builder.build()


# --- detection helpers ------------------------------------------------------


def _collect_input_seq_refs(inputs: Any) -> set[Any]:
    """Collect sequence references from a source's input list."""
    if not isinstance(inputs, list):
        return set()
    refs: set[Any] = set()
    for inp in inputs:
        if isinstance(inp, dict):
            seq_ref = inp.get("sequence")
            if seq_ref is not None:
                refs.add(seq_ref)
    return refs


def _collect_source_input_ids(sources: list[dict[str, Any]]) -> set[Any]:
    """Collect all sequence IDs referenced as source inputs."""
    input_ids: set[Any] = set()
    for src in sources:
        if not isinstance(src, dict):
            continue
        input_ids.update(_collect_input_seq_refs(src.get("input")))
    return input_ids


def _find_last_seq_with_content(
    sequences: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the last sequence in the list that has file_content."""
    for seq in reversed(sequences):
        if isinstance(seq, dict) and seq.get("file_content"):
            return seq
    return sequences[-1] if sequences and isinstance(sequences[-1], dict) else None


def _find_final_construct(
    sequences: list[dict[str, Any]], sources: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Find the final construct sequence.

    The final construct is a leaf sequence — one whose id is not referenced
    as an input by any source.  If there are multiple leaves, the last one
    (most recently produced) is the final construct.  If there are no leaves
    (e.g. all sequences are inputs), fall back to the last sequence with
    ``file_content``.
    """
    if not sequences:
        return None

    # Collect all ids referenced as source inputs.
    input_ids = _collect_source_input_ids(sources)

    # Leaf sequences: not referenced as an input by any source.
    leaves = [
        seq
        for seq in sequences
        if isinstance(seq, dict) and seq.get("id") not in input_ids
    ]

    if leaves:
        # Prefer the last leaf with file_content.
        for seq in reversed(leaves):
            if seq.get("file_content"):
                return seq
        return leaves[-1]

    # Fallback: last sequence with file_content.
    return _find_last_seq_with_content(sequences)


def _has_genbank_artifact(artifacts: dict[str, Any] | None) -> bool:
    """Check if a GenBank artifact is available for writeback.

    Accepts either the normalized artifacts payload (with an ``artifacts``
    list) or approval_args (with ``artifact_content`` / ``artifact_id``).
    """
    if not artifacts or not isinstance(artifacts, dict):
        return False
    # Normalized artifacts payload (from normalize_opencloning_artifacts).
    artifact_list = artifacts.get("artifacts")
    if isinstance(artifact_list, list) and artifact_list:
        return True
    # Inline artifact content (from approval_args).
    if artifacts.get("artifact_content"):
        return True
    # Bundle reference (from approval_args).
    if artifacts.get("artifact_id"):
        return True
    return False


def _detect_annotation(
    step_manifests: list[dict[str, Any]],
    final_seq: dict[str, Any] | None,
) -> tuple[bool, bool]:
    """Detect whether post-assembly annotation was attempted and completed.

    Returns ``(annotation_attempted, annotation_completed)``.

    Primary signal: step manifests with ``operation_type == "annotation"`` or
    endpoint containing ``/annotate/``.
    Fallback: the final construct has annotated features (``feature_count > 0``
    or a GenBank FEATURES section with feature lines).
    """
    # Primary: check step manifests.
    annotation_manifests = [
        m
        for m in step_manifests
        if isinstance(m, dict)
        and (
            m.get("operation_type") == "annotation"
            or "/annotate/" in str(m.get("endpoint", ""))
        )
    ]
    if annotation_manifests:
        attempted = True
        completed = any(m.get("status") == "succeeded" for m in annotation_manifests)
        return attempted, completed

    # Fallback: check if the final construct has features.
    if final_seq is not None and _sequence_has_features(final_seq):
        return True, True

    return False, False


def _genbank_has_feature_lines(file_content: str) -> bool:
    """Check if a GenBank file_content string has at least one feature line."""
    in_features = False
    for line in file_content.splitlines():
        if line.startswith("FEATURES"):
            in_features = True
            continue
        if in_features:
            if line.startswith("ORIGIN") or line.startswith("//"):
                break
            # Feature lines start with 5 spaces followed by a non-space char.
            if len(line) > 5 and line[:5] == " " * 5 and line[5] != " ":
                return True
    return False


def _sequence_has_features(seq: dict[str, Any]) -> bool:
    """Check if a sequence has annotated features.

    Checks the ``feature_count`` field first, then falls back to parsing the
    GenBank ``file_content`` for a non-empty FEATURES section.
    """
    if not isinstance(seq, dict):
        return False
    feature_count = _int_or_none(seq.get("feature_count"))
    if feature_count is not None and feature_count > 0:
        return True
    file_content = seq.get("file_content")
    if not isinstance(file_content, str):
        return False
    return _genbank_has_feature_lines(file_content)


def _count_primers_with_details(primers: list[dict[str, Any]]) -> int:
    """Count primers that have analysis detail fields (Tm, GC, etc.).

    "Primer details" refers to the analysis results from ``/primer_details``
    (Tm, GC content, heterodimer, etc.), not the primer sequence itself —
    the sequence is the primer, not a detail about it.
    """
    count = 0
    for primer in primers:
        if not isinstance(primer, dict):
            continue
        has_tm = (
            primer.get("tm") is not None
            or primer.get("melting_temperature") is not None
        )
        has_gc = (
            primer.get("gc_content") is not None or primer.get("gc_percent") is not None
        )
        has_analysis = (
            primer.get("heterodimer") is not None
            or primer.get("self_dimer") is not None
            or primer.get("hairpin") is not None
        )
        if has_tm or has_gc or has_analysis:
            count += 1
    return count


def _count_primer_pairs(primer_count: int) -> int:
    """Number of unique primer pairs for heterodimer checking."""
    if primer_count < 2:
        return 0
    return primer_count * (primer_count - 1) // 2


def _count_checked_heterodimer_pairs(
    step_manifests: list[dict[str, Any]],
    primers: list[dict[str, Any]],
    primer_pairs: int,
) -> int:
    """Detect how many primer pairs had heterodimer checked.

    Primary signal: step manifests with endpoints containing ``heterodimer``
    or ``primer_details``, or ``operation_type == "primer_design"``.
    Fallback: primers with heterodimer/self_dimer/hairpin data in their fields.
    """
    # Primary: check step manifests for primer analysis steps.
    has_analysis = any(
        isinstance(m, dict)
        and (
            "heterodimer" in str(m.get("endpoint", "")).lower()
            or "primer_details" in str(m.get("endpoint", "")).lower()
            or m.get("operation_type") == "primer_design"
        )
        for m in step_manifests
    )
    if has_analysis:
        return primer_pairs

    # Fallback: check if any primer has heterodimer data.
    has_heterodimer_data = any(
        isinstance(p, dict)
        and (
            p.get("heterodimer") is not None
            or p.get("self_dimer") is not None
            or p.get("hairpin") is not None
        )
        for p in primers
    )
    if has_heterodimer_data:
        return primer_pairs

    return 0


def _detect_protocol_status(
    protocol_result: dict[str, Any] | None,
) -> tuple[bool, bool]:
    """Detect protocol lookup status.

    Returns ``(protocol_found, fallback_used)``.
    """
    if not protocol_result or not isinstance(protocol_result, dict):
        return False, False
    fallback_used = bool(protocol_result.get("fallback_used"))
    # Protocol found if there's a best_match or non-empty matches list.
    has_match = bool(
        protocol_result.get("best_match")
        or (
            isinstance(protocol_result.get("matches"), list)
            and protocol_result["matches"]
        )
    )
    return has_match, fallback_used


def _int_or_none(value: Any) -> int | None:
    """Return an int value or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
