"""Authenticated measurement transport; plaintext is restricted to loopback."""

import ipaddress
import os
from pathlib import Path
from urllib.parse import urlsplit

import requests


def require_secure_url(value):
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.port == 0
        or any(c.isspace() for c in value)
    ):
        raise ValueError(
            "Measurement URL must be HTTP(S) without credentials, query or fragment"
        )
    try:
        local = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        local = parsed.hostname.lower() == "localhost"
    if parsed.scheme == "http" and not local:
        raise ValueError(
            "Remote measurement services require HTTPS; HTTP is loopback-only"
        )
    return value.rstrip("/")


class VerifiedSession(requests.Session):
    def __init__(self, ca_cert=None):
        super().__init__()
        ca_cert = ca_cert or os.environ.get("SMA_CA_CERT")
        if ca_cert is not None and (
            not isinstance(ca_cert, str) or not Path(ca_cert).is_file()
        ):
            raise ValueError(
                "Measurement ca_cert must identify a readable CA certificate file"
            )
        self.verify = ca_cert or True

    def request(self, method, url, **kwargs):
        require_secure_url(url)
        if kwargs.get("verify") is False or self.verify is False:
            raise ValueError("Measurement TLS verification cannot be disabled")
        kwargs["allow_redirects"] = False
        kwargs.setdefault("verify", self.verify)
        kwargs.setdefault("timeout", (10, 30))
        response = super().request(method, url, **kwargs)
        if 300 <= response.status_code < 400:
            raise requests.HTTPError(
                "Measurement endpoint redirect refused", response=response
            )
        return response
