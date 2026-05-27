"""
tests/test_auth.py

Unit tests for gateway/auth/jwt_handler.py.
Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Ensure a stable JWT secret for all tests
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-unit-tests-only")
os.environ.setdefault("DEMO_MODE", "true")
os.environ.setdefault("JWT_EXPIRY_SECONDS", "3600")

import pytest

from gateway.auth.jwt_handler import (
    create_token,
    verify_token,
    extract_user_id,
    issue_demo_token,
    _b64url_encode,
    _b64url_decode,
)


class TestBase64Utils:

    def test_encode_decode_roundtrip(self):
        original = b"hello world\x00\xff"
        encoded = _b64url_encode(original)
        assert isinstance(encoded, str)
        assert _b64url_decode(encoded) == original

    def test_no_padding_in_encoded(self):
        encoded = _b64url_encode(b"a")
        assert "=" not in encoded

    def test_empty_bytes(self):
        assert _b64url_encode(b"") == ""


class TestCreateToken:

    def test_creates_three_part_jwt(self):
        token = create_token("user-1")
        parts = token.split(".")
        assert len(parts) == 3

    def test_default_role_is_student(self):
        token = create_token("user-1")
        payload = verify_token(token)
        assert "student" in payload["roles"]

    def test_custom_roles(self):
        token = create_token("instructor-1", roles=["instructor", "admin"])
        payload = verify_token(token)
        assert "instructor" in payload["roles"]
        assert "admin" in payload["roles"]

    def test_sub_claim(self):
        token = create_token("alice")
        payload = verify_token(token)
        assert payload["sub"] == "alice"

    def test_exp_in_future(self):
        token = create_token("u", expires_in=300)
        payload = verify_token(token)
        assert payload["exp"] > int(time.time())

    def test_exp_approximately_correct(self):
        token = create_token("u", expires_in=300)
        payload = verify_token(token)
        diff = payload["exp"] - int(time.time())
        assert 295 <= diff <= 305

    def test_iat_claim_present(self):
        token = create_token("u")
        payload = verify_token(token)
        assert "iat" in payload
        assert abs(payload["iat"] - int(time.time())) < 5

    def test_jti_unique_per_token(self):
        t1 = create_token("u")
        t2 = create_token("u")
        assert verify_token(t1)["jti"] != verify_token(t2)["jti"]

    def test_different_users_different_tokens(self):
        assert create_token("alice") != create_token("bob")


class TestVerifyToken:

    def test_valid_token_returns_payload(self):
        token = create_token("u1")
        payload = verify_token(token)
        assert payload["sub"] == "u1"

    def test_expired_token_raises(self):
        token = create_token("u", expires_in=-1)
        with pytest.raises(ValueError, match="expired"):
            verify_token(token)

    def test_tampered_signature_raises(self):
        token = create_token("u")
        parts = token.split(".")
        tampered = parts[0] + "." + parts[1] + ".invalidsignature"
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            verify_token(tampered)

    def test_tampered_payload_raises(self):
        import base64, json
        token = create_token("u")
        h, p, s = token.split(".")
        # Decode, modify, re-encode payload
        padding = 4 - len(p) % 4
        if padding != 4:
            p += "=" * padding
        payload_data = json.loads(base64.urlsafe_b64decode(p))
        payload_data["sub"] = "hacker"
        new_p = base64.urlsafe_b64encode(
            json.dumps(payload_data, separators=(",", ":")).encode()
        ).rstrip(b"=").decode()
        tampered = f"{h}.{new_p}.{s}"
        with pytest.raises(ValueError):
            verify_token(tampered)

    def test_malformed_token_raises(self):
        with pytest.raises(ValueError, match="Malformed"):
            verify_token("notavalidtoken")

    def test_two_parts_raises(self):
        with pytest.raises(ValueError, match="Malformed"):
            verify_token("header.payload")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            verify_token("")

    def test_missing_sub_raises(self):
        # Construct a valid-signature token with no 'sub' claim
        import json, time as _time
        from gateway.auth.jwt_handler import _b64url_encode, _sign
        from gateway.config import get_gateway_settings
        settings = get_gateway_settings()

        import json as _json
        header  = _b64url_encode(_json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        payload_data = {"roles": ["student"], "iat": int(_time.time()), "exp": int(_time.time()) + 3600, "jti": "x"}
        payload = _b64url_encode(_json.dumps(payload_data).encode())
        sig     = _sign(header, payload, settings.jwt_secret)
        token   = f"{header}.{payload}.{sig}"

        with pytest.raises(ValueError, match="sub"):
            verify_token(token)


class TestExtractUserId:

    def test_extracts_user_id(self):
        token = create_token("extract-me")
        assert extract_user_id(token) == "extract-me"

    def test_invalid_token_raises(self):
        with pytest.raises(ValueError):
            extract_user_id("bad.token.here")


class TestIssueDemoToken:

    def test_issues_token_in_demo_mode(self):
        result = issue_demo_token("demo-user")
        assert "access_token" in result
        assert result["token_type"] == "bearer"
        assert result["user_id"] == "demo-user"
        assert result["expires_in"] > 0

    def test_issued_token_is_valid(self):
        result = issue_demo_token("demo-user-2")
        token = result["access_token"]
        payload = verify_token(token)
        assert payload["sub"] == "demo-user-2"

    def test_demo_mode_disabled_raises(self):
        import importlib
        # Temporarily set DEMO_MODE=false
        original = os.environ.get("DEMO_MODE", "true")
        os.environ["DEMO_MODE"] = "false"
        try:
            # Clear the lru_cache so the new env value is picked up
            from gateway.config import get_gateway_settings as _gs
            _gs.cache_clear()
            with pytest.raises(PermissionError, match="[Dd]emo"):
                issue_demo_token("u")
        finally:
            os.environ["DEMO_MODE"] = original
            from gateway.config import get_gateway_settings as _gs
            _gs.cache_clear()
