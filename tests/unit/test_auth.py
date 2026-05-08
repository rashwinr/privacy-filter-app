"""Tests for JWT bearer authentication — demo tokens + protected routes.

Run with:
    pytest tests/unit/test_auth.py -v

All tests use the FastAPI TestClient with auth enforcement enabled via
a patched SECRET_KEY. The real model is never loaded (conftest autouse fixture).
"""
from __future__ import annotations

import time
import os
import pytest
from fastapi.testclient import TestClient
from jose import jwt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_SECRET = "test-secret-key-do-not-use-in-production"
ALGORITHM   = "HS256"


def _make_token(
    *,
    secret: str = TEST_SECRET,
    sub: str = "test@example.com",
    name: str = "Test User",
    email: str = "test@example.com",
    token_type: str = "demo",
    exp_offset: int = 86_400,          # seconds from now; negative → already expired
) -> str:
    now = int(time.time())
    payload = {
        "sub": sub,
        "name": name,
        "email": email,
        "type": token_type,
        "iat": now,
        "exp": now + exp_offset,
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_env(monkeypatch):
    """Ensure auth is enabled with a deterministic secret for all tests."""
    monkeypatch.setenv("SECRET_KEY", TEST_SECRET)
    monkeypatch.setenv("KEYCLOAK_AUTH_ENABLED", "true")
    # Disable Keycloak JWKS entirely (no realm URL) so only demo tokens work.
    monkeypatch.delenv("KEYCLOAK_REALM_URL", raising=False)
    # Use local storage so no GCS calls happen.
    monkeypatch.setenv("STORAGE_BACKEND", "local")


@pytest.fixture()
def client(tmp_data_dir):   # tmp_data_dir from conftest patches LOCAL_DATA_DIR
    # Import *inside* the fixture so env vars are already patched.
    from app.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# /api/demo-token (public)
# ---------------------------------------------------------------------------

class TestDemoTokenEndpoint:
    def test_returns_token_for_valid_request(self, client):
        r = client.post(
            "/api/demo-token",
            json={"name": "Alice Smith", "email": "alice@example.com"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in_days"] == 7

    def test_token_payload_contains_name_and_email(self, client):
        r = client.post(
            "/api/demo-token",
            json={"name": "Bob Jones", "email": "bob@example.com"},
        )
        token = r.json()["access_token"]
        claims = jwt.decode(token, TEST_SECRET, algorithms=[ALGORITHM])
        assert claims["name"] == "Bob Jones"
        assert claims["email"] == "bob@example.com"
        assert claims["type"] == "demo"

    def test_token_signature_uses_secret_key(self, client):
        r = client.post(
            "/api/demo-token",
            json={"name": "Carol", "email": "carol@example.com"},
        )
        token = r.json()["access_token"]
        # Should decode fine with the correct secret.
        claims = jwt.decode(token, TEST_SECRET, algorithms=[ALGORITHM])
        assert claims["sub"] == "carol@example.com"
        # Must fail with a wrong secret.
        with pytest.raises(Exception):
            jwt.decode(token, "wrong-secret", algorithms=[ALGORITHM])

    def test_rejects_invalid_email(self, client):
        r = client.post(
            "/api/demo-token",
            json={"name": "Dave", "email": "not-an-email"},
        )
        assert r.status_code == 422   # Pydantic validation error

    def test_rejects_missing_fields(self, client):
        r = client.post("/api/demo-token", json={"name": "Eve"})
        assert r.status_code == 422

    def test_endpoint_is_public_no_auth_required(self, client):
        """Should succeed with no Authorization header."""
        r = client.post(
            "/api/demo-token",
            json={"name": "Public User", "email": "pub@example.com"},
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# /api/health (always public)
# ---------------------------------------------------------------------------

class TestHealthEndpointPublic:
    def test_health_reachable_without_token(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_health_reachable_with_token(self, client):
        token = _make_token()
        r = client.get("/api/health", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# /api/redact — protected
# ---------------------------------------------------------------------------

class TestRedactAuth:
    def _txt_payload(self):
        return {"file": ("test.txt", b"Hello Alice Smith, your email is alice@example.com.", "text/plain")}

    def test_valid_token_allows_redact(self, client):
        token = _make_token()
        r = client.post(
            "/api/redact",
            files=self._txt_payload(),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200

    def test_missing_token_returns_401(self, client):
        r = client.post("/api/redact", files=self._txt_payload())
        assert r.status_code == 401

    def test_expired_token_returns_401(self, client):
        token = _make_token(exp_offset=-3600)   # expired 1 h ago
        r = client.post(
            "/api/redact",
            files=self._txt_payload(),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 401
        assert "expired" in r.json()["detail"].lower()

    def test_tampered_token_returns_401(self, client):
        token = _make_token()
        # Flip one character in the signature to corrupt it.
        parts = token.split(".")
        sig = parts[2]
        parts[2] = sig[:-1] + ("A" if sig[-1] != "A" else "B")
        bad_token = ".".join(parts)
        r = client.post(
            "/api/redact",
            files=self._txt_payload(),
            headers={"Authorization": f"Bearer {bad_token}"},
        )
        assert r.status_code == 401

    def test_wrong_secret_token_returns_401(self, client):
        token = _make_token(secret="completely-different-secret")
        r = client.post(
            "/api/redact",
            files=self._txt_payload(),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 401

    def test_wrong_token_type_returns_401(self, client):
        """A token signed correctly but missing type='demo' should be rejected."""
        token = _make_token(token_type="production")
        r = client.post(
            "/api/redact",
            files=self._txt_payload(),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# Auth bypass (KEYCLOAK_AUTH_ENABLED=false)
# ---------------------------------------------------------------------------

class TestAuthBypass:
    def test_redact_works_without_token_when_auth_disabled(self, monkeypatch, tmp_data_dir):
        monkeypatch.setenv("KEYCLOAK_AUTH_ENABLED", "false")
        from importlib import reload
        import app.auth as auth_mod
        reload(auth_mod)
        import app.main as main_mod
        reload(main_mod)

        with TestClient(main_mod.app) as c:
            r = c.post(
                "/api/redact",
                files={"file": ("t.txt", b"No token needed.", "text/plain")},
            )
            assert r.status_code == 200

        # Restore
        reload(auth_mod)
        reload(main_mod)
