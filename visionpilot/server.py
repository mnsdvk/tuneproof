"""Zero-dependency web endpoint (stdlib only).

    python -m visionpilot.server --port 8080

GET  /                      demo UI
GET  /api/health            runtime + OpenCV build info (shows Arm/COOL flags)
GET  /api/sample?mode=dark  run the agent on a synthetic frame with ground truth
POST /api/inspect           raw image bytes in body -> JSON trace + overlay
POST /api/approve           human decision on a flagged item (appended to approvals.jsonl)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import service

STATIC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
MAX_BODY = 20 * 1024 * 1024
APPROVALS = os.environ.get("VP_APPROVALS", "approvals.jsonl")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                with open(os.path.join(STATIC, "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/api/health":
                return self._send(200, {"status": "ok", **service.build_info()})
            if u.path == "/api/sample":
                t0 = time.perf_counter()
                mode = q.get("mode", ["mixed"])[0]
                if mode not in ("clean", "blur", "dark", "noise", "mixed"):
                    return self._send(400, {"error": "bad mode"})
                r = service.inspect_sample(mode, int(q.get("seed", ["0"])[0]))
                r["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                return self._send(200, r)
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": str(e)})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > MAX_BODY:
            return self._send(413, {"error": "body missing or too large (max 20 MB)"})
        body = self.rfile.read(n)
        try:
            if self.path == "/api/inspect":
                t0 = time.perf_counter()
                r = service.inspect_bytes(body)
                r["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                return self._send(200, r)
            if self.path == "/api/approve":
                rec = json.loads(body)
                rec["ts"] = time.time()
                with open(APPROVALS, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                return self._send(200, {"ok": True})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": str(e)})
        self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):  # structured, single-line access log for CloudWatch
        print(json.dumps({"t": time.time(), "client": self.address_string(), "req": fmt % args}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    print(f"VisionPilot listening on {a.host}:{a.port} | {service.build_info()}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
