"""Transport policy tests, independent of measurement queries."""

from unittest.mock import Mock, patch

import pytest
import requests
from sma.prometheus import Prometheus
from sma.tls import VerifiedSession, require_secure_url


@pytest.mark.parametrize(
    "url",
    ["http://metrics.example:9090", "http://10.0.0.1:9090", "http://127.0.0.1.evil"],
)
def test_remote_http_is_rejected(url):
    with pytest.raises(ValueError, match="HTTPS"):
        Prometheus({"address": url})


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:9090", "http://[::1]:9090", "https://metrics.example"]
)
def test_local_http_and_remote_https_are_supported(url):
    assert require_secure_url(url) == url


def test_custom_ca_is_used_and_redirects_are_refused(tmp_path):
    ca = tmp_path / "ca.crt"
    ca.write_text("fixture public CA; no handshake in this policy test")
    client = Prometheus({"address": "https://metrics.example", "ca_cert": str(ca)})
    response = Mock(status_code=302)
    with patch.object(requests.Session, "request", return_value=response) as send:
        with pytest.raises(requests.HTTPError, match="redirect refused"):
            client.session.get("https://metrics.example/api/v1/query")
    assert send.call_args.kwargs["verify"] == str(ca)
    assert send.call_args.kwargs["allow_redirects"] is False


def test_tls_validation_cannot_be_disabled():
    with VerifiedSession() as client:
        with pytest.raises(ValueError, match="cannot be disabled"):
            client.get("https://metrics.example", verify=False)


def test_missing_custom_ca_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="CA certificate file"):
        Prometheus(
            {
                "address": "https://metrics.example",
                "ca_cert": str(tmp_path / "missing.pem"),
            }
        )


def test_real_https_measurement_with_private_ca(tmp_path, monkeypatch):
    import http.server
    import ssl
    import subprocess
    import threading

    monkeypatch.setenv("NO_PROXY", "*")
    config = tmp_path / "openssl.cnf"
    config.write_text(
        "[req]\ndistinguished_name=dn\n[dn]\nCN=Measurement test\n[v3]\n"
        "basicConstraints=critical,CA:TRUE\nsubjectAltName=IP:127.0.0.1\n"
    )
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=Measurement test",
            "-config",
            str(config),
            "-extensions",
            "v3",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"success","data":{"activeTargets":[]}}')

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"https://127.0.0.1:{server.server_port}"
    try:
        with VerifiedSession() as untrusted:
            with pytest.raises(requests.exceptions.SSLError):
                untrusted.get(url)
        client = Prometheus({"address": url, "ca_cert": str(cert)})
        assert client.targets()["data"]["activeTargets"] == []
        client.session.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
