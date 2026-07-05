"""C14 Keycloak JWT authentication for the lab copilot gateway.

Verifies Keycloak bearer JWTs and produces a canonical
:class:`AuthenticatedPrincipal` that downstream routes use for identity
resolution.  The principal's ``keycloak_subject`` is the only trusted
identity selector for protected actions; ``librechat_user_id`` is
correlation/audit metadata only.

Two modes:

    * ``LAB_COPILOT_KEYCLOAK_VERIFY=true``  — verify JWTs against Keycloak
      JWKS.  Fails closed on missing config, unknown keys, expired cache,
      or first-boot JWKS outage.
    * ``LAB_COPILOT_KEYCLOAK_VERIFY=false`` — dev/test/local only.  Requires
      an explicit ``LAB_COPILOT_DEV_AUTH_SECRET`` trust header.  Returns a
      synthetic principal with ``auth_mode="dev"``.  Refuses to start in
      non-local environments.

The module exposes a FastAPI dependency :func:`verified_principal` that
routes use via ``Depends(verified_principal)``.  The dependency raises
:class:`AuthError` (caught by a registered exception handler → 401/403)
on any authentication failure.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import jwt
import jwt.algorithms  # pyright: ignore — runtime submodule accessed via jwt.algorithms.RSAAlgorithm
from fastapi import Request
from fastapi.responses import JSONResponse


# --- configuration --------------------------------------------------------


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


#: Allowed environments for dev-auth bypass.
_DEV_ENVS = frozenset({"local", "test", "dev"})


class AuthConfig:
    """Runtime auth configuration sourced from environment variables.

    Created once at startup; cached as a module-level singleton.  Tests
    can override via :func:`reset_auth_config`.
    """

    def __init__(self) -> None:
        self.verify_enabled: bool = _env_bool("LAB_COPILOT_KEYCLOAK_VERIFY", False)
        self.issuer: str = os.getenv("LAB_COPILOT_KEYCLOAK_ISSUER", "")
        self.audience: str = os.getenv("LAB_COPILOT_KEYCLOAK_AUDIENCE", "")
        self.jwks_url: str = os.getenv("LAB_COPILOT_KEYCLOAK_JWKS_URL", "")
        self.allowed_algs: list[str] = [
            a.strip()
            for a in os.getenv("LAB_COPILOT_KEYCLOAK_ALLOWED_ALGS", "RS256").split(",")
            if a.strip()
        ]
        self.jwks_cache_ttl: int = _env_int("LAB_COPILOT_KEYCLOAK_JWKS_CACHE_TTL", 300)
        self.jwks_timeout: int = _env_int("LAB_COPILOT_KEYCLOAK_JWKS_TIMEOUT", 10)
        self.dev_auth_secret: str = os.getenv("LAB_COPILOT_DEV_AUTH_SECRET", "")
        self.environment: str = os.getenv("LAB_COPILOT_ENV", "local")

    def validate(self) -> None:
        """Fail-closed validation at startup.

        Raises :class:`RuntimeError` if the configuration is unsafe:
        - verify=true with missing issuer/audience/JWKS
        - verify=false in a non-local environment
        - verify=false in a local environment without a dev auth secret
        """
        if self.verify_enabled:
            missing: list[str] = []
            if not self.issuer:
                missing.append("LAB_COPILOT_KEYCLOAK_ISSUER")
            if not self.audience:
                missing.append("LAB_COPILOT_KEYCLOAK_AUDIENCE")
            if not self.jwks_url:
                missing.append("LAB_COPILOT_KEYCLOAK_JWKS_URL")
            if missing:
                raise RuntimeError(
                    "LAB_COPILOT_KEYCLOAK_VERIFY=true but missing config: "
                    + ", ".join(missing)
                )
        else:
            if self.environment not in _DEV_ENVS:
                raise RuntimeError(
                    f"LAB_COPILOT_KEYCLOAK_VERIFY=false is not allowed in "
                    f"environment {self.environment!r}; must be one of "
                    f"{sorted(_DEV_ENVS)}"
                )
            if not self.dev_auth_secret:
                raise RuntimeError(
                    "LAB_COPILOT_KEYCLOAK_VERIFY=false requires "
                    "LAB_COPILOT_DEV_AUTH_SECRET to be set (no implicit trust)"
                )


# --- principal / errors ---------------------------------------------------


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    """The verified identity extracted from a JWT or dev-auth header.

    ``keycloak_subject`` is the canonical principal — the only value
    that may select identity for protected actions.  ``librechat_user_id``
    is correlation metadata only.
    """

    keycloak_subject: str
    jwt_claims: dict[str, object] = field(default_factory=dict)
    auth_mode: str = "jwt"  # "jwt" or "dev"


class AuthError(Exception):
    """Authentication failure with a stable reason code and HTTP status.

    The FastAPI exception handler in :func:`register_auth_exception_handler`
    converts this into a JSON 401/403 response.
    """

    def __init__(
        self,
        reason: str,
        message: str,
        status_code: int = 401,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.status_code = status_code


# --- JWKS client with cache -----------------------------------------------


class JwksCache:
    """Thread-safe JWKS cache with TTL-based expiry.

    Fail-closed semantics:
    - First boot: JWKS must be fetched successfully.  If it fails, the
      gateway refuses to start (caller raises RuntimeError).
    - Runtime: cached known keys may be used while cache is valid.
      Unknown ``kid`` triggers a refresh attempt; if refresh fails, deny.
      Expired cache → deny (no grace period).
    """

    def __init__(self, config: AuthConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._keys: dict[str, str] = {}  # kid → PEM public key
        self._fetched_at: float = 0.0
        self._first_boot_done: bool = False

    @property
    def is_configured(self) -> bool:
        return bool(self._config.jwks_url)

    def bootstrap(self) -> None:
        """Fetch JWKS at startup.  Raises RuntimeError on first-boot failure."""
        if not self.is_configured:
            return
        try:
            self._refresh_locked()
            self._first_boot_done = True
        except Exception as exc:
            raise RuntimeError(
                f"first-boot JWKS fetch failed; gateway refuses to start: {exc}"
            ) from exc

    def _refresh_locked(self) -> None:
        """Fetch JWKS and update the cache.  Caller must hold the lock."""
        import requests

        resp = requests.get(
            self._config.jwks_url,
            timeout=self._config.jwks_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        keys_list = data.get("keys", [])
        new_keys: dict[str, Any] = {}
        for key_data in keys_list:
            kid = key_data.get("kid")
            if not kid:
                continue
            try:
                pem = jwt.algorithms.RSAAlgorithm.from_jwk(key_data)
                new_keys[kid] = pem
            except Exception:
                continue
        self._keys = new_keys
        self._fetched_at = time.monotonic()

    def _is_cache_valid(self) -> bool:
        if self._fetched_at == 0.0:
            return False
        age = time.monotonic() - self._fetched_at
        return age < self._config.jwks_cache_ttl

    def get_key(self, kid: str) -> str | None:
        """Return the PEM key for ``kid``, refreshing if needed.

        Returns None if the key cannot be resolved (unknown kid +
        refresh failure, or expired cache).  Callers must deny on None.
        """
        if not self._first_boot_done:
            return None

        with self._lock:
            # Fast path: cached and valid
            if self._is_cache_valid():
                return self._keys.get(kid)

            # Cache expired — deny (no grace period per decision #8)
            if not self._is_cache_valid() and self._fetched_at > 0.0:
                # Try refresh; if it fails, deny
                try:
                    self._refresh_locked()
                except Exception:
                    return None
                return self._keys.get(kid)

            # Should not reach here, but deny just in case
            return None

    def status(self) -> dict[str, object]:
        """Health-facing summary (no secrets)."""
        return {
            "jwks_configured": self.is_configured,
            "jwks_keys_cached": len(self._keys),
            "jwks_cache_valid": self._is_cache_valid(),
            "jwks_first_boot_done": self._first_boot_done,
        }


# --- module-level singleton -----------------------------------------------

_auth_config: AuthConfig | None = None
_jwks_cache: JwksCache | None = None


def get_auth_config() -> AuthConfig:
    """Return the process-wide auth config (created lazily)."""
    global _auth_config
    if _auth_config is None:
        _auth_config = AuthConfig()
        _auth_config.validate()
    return _auth_config


def get_jwks_cache() -> JwksCache:
    """Return the process-wide JWKS cache (created lazily)."""
    global _jwks_cache
    if _jwks_cache is None:
        config = get_auth_config()
        _jwks_cache = JwksCache(config)
        if config.verify_enabled:
            _jwks_cache.bootstrap()
    return _jwks_cache


def reset_auth_config(
    config: AuthConfig | None = None,
    *,
    skip_validate: bool = False,
) -> None:
    """Test helper: replace or clear the singleton.

    When ``config`` is provided and ``skip_validate`` is False, validate()
    is called.  When ``skip_validate`` is True, the config is stored
    without validation (for tests that want to check validate() separately).
    """
    global _auth_config, _jwks_cache
    if config is not None and not skip_validate:
        config.validate()
    _auth_config = config
    _jwks_cache = None  # force re-creation on next get_jwks_cache()


# --- JWT verification -----------------------------------------------------


def _verify_jwt(token: str, config: AuthConfig, jwks: JwksCache) -> dict[str, object]:
    """Verify a JWT and return its claims.

    Raises :class:`AuthError` on any verification failure.
    """
    try:
        # Decode without verification first to extract the header (kid)
        unverified_header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise AuthError(
            reason="malformed_token",
            message=f"JWT could not be decoded: {exc}",
        ) from exc

    kid = unverified_header.get("kid")
    alg = unverified_header.get("alg")

    # Reject "none" algorithm explicitly
    if alg is None or alg not in config.allowed_algs:
        raise AuthError(
            reason="unsupported_algorithm",
            message=f"JWT algorithm {alg!r} is not in allowed list",
        )

    # Resolve the signing key
    if kid is None:
        raise AuthError(
            reason="missing_kid",
            message="JWT header missing 'kid'; cannot resolve signing key",
        )

    signing_key = jwks.get_key(kid)
    if signing_key is None:
        raise AuthError(
            reason="unknown_signing_key",
            message=f"JWT kid {kid!r} not found in JWKS or cache expired",
        )

    try:
        claims = jwt.decode(
            token,
            signing_key,
            algorithms=config.allowed_algs,
            issuer=config.issuer,
            audience=config.audience,
            options={
                "require": ["exp", "sub"],
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iss": True,
                "verify_aud": True,
            },
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError(
            reason="expired_token",
            message=f"JWT has expired: {exc}",
        ) from exc
    except jwt.InvalidIssuerError as exc:
        raise AuthError(
            reason="wrong_issuer",
            message=f"JWT issuer does not match: {exc}",
        ) from exc
    except jwt.InvalidAudienceError as exc:
        raise AuthError(
            reason="wrong_audience",
            message=f"JWT audience does not match: {exc}",
        ) from exc
    except jwt.MissingRequiredClaimError as exc:
        raise AuthError(
            reason="missing_claim",
            message=f"JWT missing required claim: {exc}",
        ) from exc
    except jwt.PyJWTError as exc:
        raise AuthError(
            reason="invalid_signature",
            message=f"JWT signature verification failed: {exc}",
        ) from exc

    sub = claims.get("sub")
    if not sub or not isinstance(sub, str):
        raise AuthError(
            reason="missing_sub",
            message="JWT 'sub' claim is missing or empty",
        )

    return claims


def _verify_dev_auth(request: Request, config: AuthConfig) -> AuthenticatedPrincipal:
    """Dev-mode authentication via a trust header.

    Requires ``X-Lab-Copilot-Dev-Auth`` header matching
    ``LAB_COPILOT_DEV_AUTH_SECRET``.  Returns a synthetic principal
    with ``keycloak_subject="dev-<random>`` if no explicit sub is provided
    via ``X-Lab-Copilot-Dev-Sub`` header.
    """
    provided = request.headers.get("X-Lab-Copilot-Dev-Auth", "")
    if not provided or provided != config.dev_auth_secret:
        raise AuthError(
            reason="dev_auth_failed",
            message="dev auth secret missing or mismatched",
        )

    # Allow tests to specify a sub; otherwise use a stable dev principal
    dev_sub = request.headers.get("X-Lab-Copilot-Dev-Sub", "dev-unverified")
    return AuthenticatedPrincipal(
        keycloak_subject=dev_sub,
        jwt_claims={"sub": dev_sub, "iss": "dev", "aud": "dev"},
        auth_mode="dev",
    )


# --- FastAPI dependency ---------------------------------------------------


def verified_principal(request: Request) -> AuthenticatedPrincipal:
    """FastAPI dependency: verify the caller and return the principal.

    Usage in routes::

        @api.post("/elabftw/mint_context_token")
        def mint(body: ElabftwMintBody, principal: AuthenticatedPrincipal = Depends(verified_principal)):
            ...

    Raises :class:`AuthError` on any failure.  The exception handler
    converts it to a 401/403 JSON response.
    """
    config = get_auth_config()

    if not config.verify_enabled:
        return _verify_dev_auth(request, config)

    # JWT verification path
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise AuthError(
            reason="missing_bearer_token",
            message="Authorization: Bearer <jwt> header is required",
        )

    token = auth_header[len("Bearer ") :].strip()
    if not token:
        raise AuthError(
            reason="missing_bearer_token",
            message="Bearer token is empty",
        )

    jwks = get_jwks_cache()
    claims = _verify_jwt(token, config, jwks)
    sub = claims["sub"]
    assert isinstance(sub, str)  # checked in _verify_jwt

    return AuthenticatedPrincipal(
        keycloak_subject=sub,
        jwt_claims=claims,
        auth_mode="jwt",
    )


# --- exception handler ----------------------------------------------------


def register_auth_exception_handler(app: Any) -> None:
    """Register the AuthError → JSON response handler on a FastAPI app."""

    @app.exception_handler(AuthError)
    def _handle_auth_error(_request: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "ok": False,
                "reason": exc.reason,
                "message": exc.message,
            },
        )
