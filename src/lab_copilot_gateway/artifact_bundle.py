"""Short-lived OpenCloning artifact bundle custody (C51).

The chat transcript and execution plans carry compact manifests only.  The
actual GenBank/FASTA/history bytes stay in this process-local store until the
user downloads them or approves writeback.  This module deliberately has no
durable backend: eLabFTW remains the durable artifact store after approval.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable


DEFAULT_BUNDLE_TTL_SECONDS = 15 * 60
DEFAULT_MAX_ARTIFACT_BYTES = 1_048_576  # 1 MiB, aligned with OpenCloning writeback
_RESERVED_MANIFEST_KEYS: frozenset[str] = frozenset(
    {
        "artifact_id",
        "plan_id",
        "plan_hash",
        "filename",
        "mime_type",
        "size_bytes",
        "sha256",
        "created_at",
        "expires_at",
    }
)


class ArtifactBundleError(Exception):
    """Base class for artifact bundle custody failures."""


class ArtifactMissing(ArtifactBundleError):
    """Raised when a bundle/artifact reference does not exist."""


class ArtifactExpired(ArtifactBundleError):
    """Raised when an artifact exists but its bundle has expired."""


class ArtifactTooLarge(ArtifactBundleError):
    """Raised when artifact bytes exceed the configured writeback limit."""

    def __init__(self, size: int, limit: int) -> None:
        self.size = size
        self.limit = limit
        super().__init__(f"artifact is {size} bytes; limit is {limit} bytes")


class ArtifactHashMismatch(ArtifactBundleError):
    """Raised when expected manifest identity does not match stored bytes."""


class ArtifactSizeMismatch(ArtifactBundleError):
    """Raised when expected size metadata does not match stored bytes."""


class ArtifactContextMismatch(ArtifactBundleError):
    """Raised when request context does not match bundle binding metadata."""


class ArtifactManifestCollision(ArtifactBundleError):
    """Raised when caller-supplied manifest fields shadow reserved identity fields."""


@dataclass(frozen=True)
class StoredArtifact:
    """One server-held artifact plus its public manifest."""

    artifact_id: str
    plan_id: str
    plan_hash: str
    filename: str
    mime_type: str
    bytes_data: bytes
    sha256: str
    size_bytes: int
    created_at: float
    expires_at: float
    binding: dict[str, Any] = field(default_factory=dict)
    manifest_extra: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        """Return the user-facing manifest without raw bytes."""
        manifest: dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }
        manifest.update(self.manifest_extra)
        return manifest


class ArtifactBundleStore:
    """Process-local, expiry-bound artifact byte store.

    Keys are intentionally scoped by ``plan_id`` + ``plan_hash`` +
    ``artifact_id`` so stale approvals or stale chat cards cannot resolve a
    newer artifact with the same visible filename.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = DEFAULT_BUNDLE_TTL_SECONDS,
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_artifact_bytes = max_artifact_bytes
        self._clock = clock or time.time
        self._artifacts: dict[tuple[str, str, str], StoredArtifact] = {}

    def put(
        self,
        *,
        plan_id: str,
        plan_hash: str,
        artifact_id: str,
        filename: str,
        mime_type: str,
        data: bytes | str,
        binding: dict[str, Any] | None = None,
        manifest_extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store bytes and return a manifest with hash/size metadata."""
        raw = data.encode("utf-8") if isinstance(data, str) else data
        if len(raw) > self.max_artifact_bytes:
            raise ArtifactTooLarge(len(raw), self.max_artifact_bytes)
        extra = dict(manifest_extra or {})
        overlap = _RESERVED_MANIFEST_KEYS.intersection(extra)
        if overlap:
            raise ArtifactManifestCollision(
                f"manifest_extra cannot override reserved keys: {sorted(overlap)}"
            )

        now = self._clock()
        artifact = StoredArtifact(
            artifact_id=artifact_id,
            plan_id=plan_id,
            plan_hash=plan_hash,
            filename=filename,
            mime_type=mime_type,
            bytes_data=raw,
            sha256=hashlib.sha256(raw).hexdigest(),
            size_bytes=len(raw),
            created_at=now,
            expires_at=now + self.ttl_seconds,
            binding=dict(binding or {}),
            manifest_extra=extra,
        )
        self._artifacts[self._key(plan_id, plan_hash, artifact_id)] = artifact
        return artifact.manifest()

    def get(
        self,
        *,
        plan_id: str,
        plan_hash: str,
        artifact_id: str,
        expected_sha256: str | None = None,
        expected_size_bytes: int | None = None,
        binding: dict[str, Any] | None = None,
    ) -> StoredArtifact:
        """Resolve an artifact and fail closed on stale or mismatched data."""
        key = self._key(plan_id, plan_hash, artifact_id)
        artifact = self._artifacts.get(key)
        if artifact is None:
            raise ArtifactMissing(f"artifact {artifact_id!r} not found")

        if self._clock() >= artifact.expires_at:
            self._artifacts.pop(key, None)
            raise ArtifactExpired(f"artifact {artifact_id!r} expired")

        actual_sha256 = hashlib.sha256(artifact.bytes_data).hexdigest()
        if actual_sha256 != artifact.sha256:
            raise ArtifactHashMismatch("stored bytes no longer match manifest hash")
        if expected_sha256 is not None and expected_sha256 != artifact.sha256:
            raise ArtifactHashMismatch("expected hash does not match artifact hash")
        if (
            expected_size_bytes is not None
            and expected_size_bytes != artifact.size_bytes
        ):
            raise ArtifactSizeMismatch("expected size does not match artifact size")

        self._validate_binding(artifact, binding or {})
        return artifact

    def purge_expired(self) -> int:
        """Remove expired artifacts and return the number purged."""
        now = self._clock()
        expired = [
            key
            for key, artifact in self._artifacts.items()
            if now >= artifact.expires_at
        ]
        for key in expired:
            self._artifacts.pop(key, None)
        return len(expired)

    @staticmethod
    def _key(plan_id: str, plan_hash: str, artifact_id: str) -> tuple[str, str, str]:
        return (plan_id, plan_hash, artifact_id)

    @staticmethod
    def _validate_binding(
        artifact: StoredArtifact, request_binding: dict[str, Any]
    ) -> None:
        for key, expected in artifact.binding.items():
            if request_binding.get(key) != expected:
                raise ArtifactContextMismatch(f"artifact binding mismatch for {key!r}")


_default_store: ArtifactBundleStore | None = None


def get_artifact_bundle_store() -> ArtifactBundleStore:
    """Return the process-wide artifact bundle store."""
    global _default_store
    if _default_store is None:
        _default_store = ArtifactBundleStore()
    return _default_store
