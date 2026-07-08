"""Short-lived preview store for intermediate OpenCloning sequences.

Stores GenBank/Fasta payloads generated during cloning runs so the
widget can fetch them on-demand for OVE preview rendering. Payloads
are bound to context tokens and expire after a TTL.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

DEFAULT_TTL_SECONDS = 1800  # 30 minutes — enough for a cloning session
MAX_PREVIEWS_PER_RUN = 50  # prevent unbounded growth


@dataclass
class PreviewPayload:
    """A single previewable sequence payload."""

    preview_ref: str  # unique ref ID
    run_id: str
    step_id: str
    context_token_hash: str  # for auth binding (hash, not raw token)
    sequence_id: int | None  # OpenCloning sequence ID
    file_format: str  # "genbank" | "fasta"
    file_content: str  # the actual GenBank/Fasta text
    sequence_length: int | None
    is_circular: bool | None
    created_at: float  # timestamp
    expires_at: float  # expiry timestamp


class OpenCloningPreviewStore:
    """In-memory preview store with TTL expiry and context binding."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._store: dict[str, PreviewPayload] = {}
        self._ttl = ttl_seconds

    def store(
        self,
        run_id: str,
        step_id: str,
        context_token_hash: str,
        sequence_id: int | None,
        file_format: str,
        file_content: str,
        is_circular: bool | None = None,
    ) -> str:
        """Store a previewable payload. Returns the preview_ref."""
        # Generate unique ref
        preview_ref = f"preview-{uuid.uuid4().hex[:12]}"
        now = time.time()
        # Evict expired entries
        self._evict_expired(now)
        # Enforce per-run limit
        self._enforce_run_limit(run_id)
        # Store
        self._store[preview_ref] = PreviewPayload(
            preview_ref=preview_ref,
            run_id=run_id,
            step_id=step_id,
            context_token_hash=context_token_hash,
            sequence_id=sequence_id,
            file_format=file_format,
            file_content=file_content,
            sequence_length=len(file_content) if file_content else 0,
            is_circular=is_circular,
            created_at=now,
            expires_at=now + self._ttl,
        )
        return preview_ref

    def fetch(self, preview_ref: str, context_token_hash: str) -> PreviewPayload | None:
        """Fetch a preview payload.

        Returns None if not found, expired, or token mismatch.
        """
        payload = self._store.get(preview_ref)
        if payload is None:
            return None
        if time.time() > payload.expires_at:
            del self._store[preview_ref]
            return None
        if payload.context_token_hash != context_token_hash:
            return None
        return payload

    def _evict_expired(self, now: float) -> None:
        """Remove all expired entries."""
        expired = [ref for ref, p in self._store.items() if now > p.expires_at]
        for ref in expired:
            del self._store[ref]

    def _enforce_run_limit(self, run_id: str) -> None:
        """If a run has too many previews, remove the oldest."""
        run_previews = [
            (ref, p) for ref, p in self._store.items() if p.run_id == run_id
        ]
        if len(run_previews) >= MAX_PREVIEWS_PER_RUN:
            run_previews.sort(key=lambda x: x[1].created_at)
            ref_to_remove = run_previews[0][0]
            del self._store[ref_to_remove]


# Module-level singleton (like ArtifactBundleStore)
_preview_store: OpenCloningPreviewStore | None = None


def get_preview_store() -> OpenCloningPreviewStore:
    """Return the process-wide preview store."""
    global _preview_store
    if _preview_store is None:
        _preview_store = OpenCloningPreviewStore()
    return _preview_store


def reset_preview_store() -> None:
    """Reset the preview store singleton (for testing)."""
    global _preview_store
    _preview_store = None
