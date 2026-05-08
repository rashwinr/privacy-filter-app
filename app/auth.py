"""JWT bearer-token authentication for the Privacy Filter API.

Two accepted token types
------------------------
1. **Demo tokens** — HS256 JWT signed by this service's ``SECRET_KEY``.
   Issued by ``POST /api/demo-token`` (name + email → signed JWT).

2. **Keycloak tokens** (optional) — RS256 JWT from a Keycloak realm.
   Validated against the realm's published JWKS.
   Activated when ``KEYCLOAK_REALM_URL`` is set in the environment.

Auth bypass
-----------
Set ``KEYCLOAK_AUTH_ENABLED=false`` (or leave ``SECRET_KEY`` empty) to
disable enforcement entirely — useful for local development.
"""
from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError, JWTClaimsError

logger = logging.getLogger("privacy_filter.auth")

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _auth_enabled() -> bool:
    return os.getenv("KEYCLOAK_AUTH_ENABLED", "true").lower() not in ("false", "0", "no")


def _secret_key() -> str:
    return os.getenv("SECRET_KEY", "")


def _keycloak_realm_url() -> Optional[str]:
    url = os.getenv("KEYCLOAK_REALM_URL", "").rstrip("/")
    return url or None


def _keycloak_audience() -> Optional[str]:
    return os.getenv("KEYCLOAK_AUDIENCE") or os.getenv("KEYCLOAK_CLIENT_ID") or None


DEMO_TOKEN_ALGORITHM = "HS256"
KEYCLOAK_ALGORITHM = "RS256"

# ---------------------------------------------------------------------------
# JWKS cache (Keycloak)
# ---------------------------------------------------------------------------

# Simple module-level cache: {realm_url: {kid: public_key_pem}}
_jwks_cache: Dict[str, Dict[str, Any]] = {}
_jwks_fetched_at: Dict[str, float] = {}
_JWKS_TTL_SECONDS = 3600  # re-fetch keys after 1 hour


async def _fetch_jwks(realm_url: str) -> Dict[str, Any]:
    """Fetch and cache JWKS from Keycloak realm."""
    now = time.monotonic()
    if realm_url in _jwks_cache and (now - _jwks_fetched_at.get(realm_url, 0)) < _JWKS_TTL_SECONDS:
        return _jwks_cache[realm_url]

    oidc_url = f"{realm_url}/.well-known/openid-configuration"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            oidc = (await client.get(oidc_url)).raise_for_status().json()
            jwks_raw = (await client.get(oidc["jwks_uri"])).raise_for_status().json()
    except Exception as exc:
        logger.exception("Failed to fetch Keycloak JWKS from %s", oidc_url)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Auth service unreachable: {exc}",
        )

    keys_by_kid = {k["kid"]: k for k in jwks_raw.get("keys", [])}
    _jwks_cache[realm_url] = keys_by_kid
    _jwks_fetched_at[realm_url] = now
    return keys_by_kid


# ---------------------------------------------------------------------------
# Token validators
# ---------------------------------------------------------------------------

def _validate_demo_token(token: str) -> Dict[str, Any]:
    secret = _secret_key()
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Demo token validation is unavailable: SECRET_KEY is not configured.",
        )
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[DEMO_TOKEN_ALGORITHM],
            options={"require_exp": True},
        )
        if claims.get("type") != "demo":
            raise JWTClaimsError("Not a demo token")
        return claims
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Demo token has expired. Request a new one from the API Access tab.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (JWTError, JWTClaimsError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def _validate_keycloak_token(token: str, realm_url: str) -> Dict[str, Any]:
    """Validate a Keycloak-issued RS256 JWT against the realm JWKS."""
    # Peek at the header to get the kid without full verification
    try:
        unverified_header = jwt.get_unverified_header(token)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Malformed token header: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    kid = unverified_header.get("kid")
    jwks = await _fetch_jwks(realm_url)

    if kid not in jwks:
        # Unknown kid — force a JWKS refresh
        _jwks_fetched_at.pop(realm_url, None)
        jwks = await _fetch_jwks(realm_url)

    if kid not in jwks:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token signing key not found in Keycloak JWKS.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    audience = _keycloak_audience()
    decode_opts: Dict[str, Any] = {"require_exp": True}
    if not audience:
        decode_opts["verify_aud"] = False

    try:
        claims = jwt.decode(
            token,
            jwks[kid],
            algorithms=[KEYCLOAK_ALGORITHM],
            audience=audience,
            options=decode_opts,
        )
        return claims
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Keycloak token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (JWTError, JWTClaimsError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Keycloak token invalid: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)


async def require_bearer(request: Request) -> Dict[str, Any]:
    """FastAPI dependency — validates bearer token on protected routes.

    Usage::

        @app.post("/api/redact")
        async def redact(claims: dict = Depends(require_bearer)):
            ...

    Returns the decoded JWT claims dict on success.
    Raises HTTP 401 on any auth failure.
    Bypasses auth entirely when ``KEYCLOAK_AUTH_ENABLED=false``.
    """
    if not _auth_enabled():
        logger.debug("Auth bypassed (KEYCLOAK_AUTH_ENABLED=false)")
        return {"sub": "anonymous", "type": "bypass"}

    # Extract Authorization: Bearer <token>
    credentials: Optional[HTTPAuthorizationCredentials] = await _bearer_scheme(request)
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Authentication required. "
                "Obtain a demo token from the API Access tab and pass it as: "
                "Authorization: Bearer <token>"
            ),
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # Peek at the algorithm claim to route to the right validator
    try:
        alg = jwt.get_unverified_header(token).get("alg", "")
    except JWTError:
        alg = ""

    realm_url = _keycloak_realm_url()

    if alg == DEMO_TOKEN_ALGORITHM or not realm_url:
        # HS256 → demo token
        return _validate_demo_token(token)
    else:
        # RS256 (or any other alg) → try Keycloak
        return await _validate_keycloak_token(token, realm_url)
