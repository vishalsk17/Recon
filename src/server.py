"""
The HTTP layer for the Recon.

A deliberately small stdlib server. Every route is a thin wrapper over
`src/service.py`, which is where the logic lives; this file's whole job is
turning requests into function calls and dicts into JSON, plus the handful of
refusals below.

**Why stdlib rather than FastAPI.** The single-file dashboard was specified to
be served by FastAPI. `pip install fastapi uvicorn` fails in this environment —
the network proxy returns 403 for PyPI — so this is built on `http.server`
instead. That is a substitution, and it is written down here and in the README
rather than passed off as a preference. The dashboard itself is unchanged: one
HTML file, no build step, no CDN. What is lost is FastAPI's request validation
and generated API docs; what is gained is a project with no dependency beyond
numpy, which for a security-sensitive demo is not a bad trade.

**This server binds loopback only, and cannot be made to do otherwise.**
There is no authentication in this build. A `--host` flag would therefore be a
flag that turns an unauthenticated money-moving control panel into a network
service, so the flag does not exist and `serve()` raises on any address that is
not loopback. The distinction matters: a configurable default that ships safe
still ships the capability, and the standing instruction on this project is
that dangerous capability should be *absent*, not defaulted off. Anyone who
genuinely needs remote access should put a reverse proxy with real
authentication in front of it, which is a deliberate act rather than a typo.

The other refusals, each for its own reason:

  * **Only GET and POST are routed.** No PUT or DELETE, because nothing in
    this system is edited or removed — the audit trail is append-only, and a
    route that could delete from it would defeat the point of hashing it.
  * **Request bodies are capped** before being read, so a declared
    `Content-Length` cannot make the server allocate to order.
  * **Static files come from a fixed dict**, not from joining a URL onto a
    directory. Path traversal is not filtered here; it is unrepresentable,
    because there is no user-controlled path to traverse.
  * **A Content-Security-Policy forbidding remote loads**, so even if a
    future edit pasted a CDN tag into the HTML, the browser would refuse it.
    Belt and braces on the no-external-assets property.
  * **`Cache-Control: no-store`**, because these responses contain decision
    records and customer identifiers and should not sit in a disk cache.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlparse

from . import config as C
from . import service as S

DEFAULT_PORT = 8737

# Bodies larger than this are refused before being read. The largest legitimate
# POST here is a narration request: two short identifiers.
MAX_BODY_BYTES = 64 * 1024

# Log lines are truncated here. A request line can be as long as the client
# likes; a log line should not be.
MAX_LOG_CHARS = 200

# Everything outside printable ASCII is replaced before anything reaches the
# log. See `Handler.log_message` for why.
_UNPRINTABLE = re.compile(r"[^\x20-\x7e]")


def _printable(text: str) -> str:
    """The text with every control character and non-ASCII byte turned into a dot."""
    return _UNPRINTABLE.sub(".", text)


# Loopback only. See the module docstring — this is a whitelist of addresses
# the server will bind, not a default that can be overridden.
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})

# Remote loads are forbidden outright. 'unsafe-inline' is present because the
# dashboard is a single file with inline style and script by design; that is a
# trade for having no build step and no external assets, and it is the reason
# the HTML never interpolates server data into markup (see index.html).
CSP = ("default-src 'none'; "
       "style-src 'self' 'unsafe-inline'; "
       "script-src 'self' 'unsafe-inline'; "
       "img-src 'self' data:; "
       "connect-src 'self'; "
       "base-uri 'none'; "
       "form-action 'none'; "
       "frame-ancestors 'none'")

# The complete set of files this server will serve, and their content types.
# Adding a file here is an explicit act; nothing is discovered at runtime.
STATIC_FILES: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
}


class Router:
    """Exact-match and single-parameter routes, kept simple enough to read.

    Patterns use one `<param>` segment at most, which is all any endpoint here
    needs. A general-purpose pattern language would be more code to audit for
    no gain.
    """

    def __init__(self) -> None:
        self.routes: list[tuple[str, str, Callable[..., Any]]] = []

    def add(self, method: str, pattern: str, handler: Callable[..., Any]) -> None:
        self.routes.append((method, pattern, handler))

    def match(self, method: str, path: str
              ) -> tuple[Optional[Callable[..., Any]], dict[str, str], bool]:
        """Returns (handler, params, path_exists_under_another_method)."""
        wrong_method = False
        parts = [p for p in path.split("/") if p]
        for route_method, pattern, handler in self.routes:
            pattern_parts = [p for p in pattern.split("/") if p]
            if len(pattern_parts) != len(parts):
                continue
            params: dict[str, str] = {}
            for want, got in zip(pattern_parts, parts):
                if want.startswith("<") and want.endswith(">"):
                    params[want[1:-1]] = got
                elif want != got:
                    break
            else:
                if route_method == method:
                    return handler, params, False
                wrong_method = True
        return None, {}, wrong_method


def build_router() -> Router:
    r = Router()

    # --- reads ---
    r.add("GET", "/api/health", lambda q, b: S.health())
    r.add("GET", "/api/overview", lambda q, b: S.overview())
    r.add("GET", "/api/runs", lambda q, b: S.runs(limit=_int(q, "limit")))
    r.add("GET", "/api/decisions", lambda q, b: S.decisions(
        run_id=_str(q, "run_id"), limit=_int(q, "limit"),
        action=_str(q, "action"), surface=_str(q, "surface")))
    r.add("GET", "/api/pending", lambda q, b: S.pending_approvals(
        run_id=_str(q, "run_id"), limit=_int(q, "limit")))
    r.add("GET", "/api/decision/<decision_id>",
          lambda q, b, decision_id: S.decision_detail(decision_id))
    r.add("GET", "/api/event/<event_id>",
          lambda q, b, event_id: S.explain_event(event_id))
    r.add("GET", "/api/benchmark", lambda q, b: S.benchmark())
    r.add("GET", "/api/models", lambda q, b: S.model_card())
    r.add("GET", "/api/policy", lambda q, b: S.policy_snapshot())
    r.add("GET", "/api/audit/verify", lambda q, b: S.verify_audit())

    # --- writes ---
    # Two, and only one of them changes anything: `/api/approve` releases a
    # decision that was already made and gated, and `/api/narrate` returns text
    # for a person to read. There is no route that makes the agent decide or
    # act — see the note in service.py on why the sweep endpoint was removed
    # rather than guarded.
    r.add("POST", "/api/approve", _approve)
    r.add("POST", "/api/narrate", _narrate)
    return r


def _str(query: dict[str, list[str]], key: str) -> Optional[str]:
    values = query.get(key)
    return values[0] if values and values[0] else None


def _int(query: dict[str, list[str]], key: str) -> Optional[int]:
    raw = _str(query, key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        raise S.ServiceError(400, f"{key} must be an integer, got {raw!r}")


def _require(body: dict[str, Any], key: str) -> Any:
    if key not in body or body[key] in (None, ""):
        raise S.ServiceError(400, f"{key!r} is required")
    return body[key]


def _approve(query: dict[str, list[str]], body: dict[str, Any]) -> dict[str, Any]:
    """Release or refuse a gated decision.

    The approver comes from the `X-Operator` request header rather than the
    body, and the handler sets it before this is called. That is not
    authentication and is not mistaken for it — anyone who can reach this port
    can set any header they like. It is there so the audit record says who
    *claimed* responsibility, and so that claim has to be made explicitly on
    every request. Wiring it to a real identity layer means replacing one
    header read, which is the smallest seam this could have.
    """
    if "granted" not in body:
        raise S.ServiceError(400, "'granted' is required and must be true or false",
                             {"why": "a missing verdict is not a decline; this "
                                     "endpoint will not guess which way you meant"})
    granted = body["granted"]
    if not isinstance(granted, bool):
        raise S.ServiceError(400, "'granted' must be true or false, not a string or number")
    return S.resolve_approval(
        str(_require(body, "decision_id")),
        approver=str(body.get("_operator") or ""),
        granted=granted,
        reason=str(body.get("reason") or ""),
    )


def _narrate(query: dict[str, list[str]], body: dict[str, Any]) -> dict[str, Any]:
    return S.narrate(str(_require(body, "decision_id")),
                     str(body.get("role") or "explain_decision_plainly"))


class Handler(BaseHTTPRequestHandler):
    server_version = f"RecoveryCommandCentre/{C.CODE_VERSION}"
    # Announce HTTP/1.1 so keep-alive works; every response sets an explicit
    # Content-Length, which is what makes that safe.
    protocol_version = "HTTP/1.1"

    router: Router = build_router()
    # Populated by `serve`, so a running server does not re-read the HTML from
    # disk on every request and cannot be affected by a mid-session edit.
    assets: dict[str, bytes] = {}

    # -- plumbing ------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        """Log one line to stdout, with the caller's text made inert first.

        `BaseHTTPRequestHandler.log_request` passes the raw request line
        through here, so everything in it is attacker-controlled: the default
        implementation writes it to stderr verbatim, where terminal escape
        sequences are rendered by whoever is tailing the log and a long line
        buries whatever came before it. Neither is dramatic, but a log an
        operator cannot trust the shape of is worth less than no log.

        So: control characters and anything non-ASCII become dots, and the
        line is capped. One request produces exactly one line of printable
        text, which is the property the audit story needs from this file.
        """
        line = _printable((fmt % args)[:MAX_LOG_CHARS])
        sys.stdout.write("%s  %s\n" % (self.log_date_time_string(), line))

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", CSP)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, default=str, allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str, detail: Any = None) -> None:
        payload: dict[str, Any] = {"error": message, "status": status}
        if detail is not None:
            payload["detail"] = detail
        self._json(status, payload)

    def _read_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return {}
        try:
            length = int(raw_length)
        except ValueError:
            raise S.ServiceError(400, "Content-Length must be an integer")
        if length < 0:
            raise S.ServiceError(400, "Content-Length must not be negative")
        # Checked before reading, so an inflated header cannot make the server
        # allocate a buffer to a caller's specification.
        if length > MAX_BODY_BYTES:
            raise S.ServiceError(413, f"request body exceeds {MAX_BODY_BYTES} bytes")
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise S.ServiceError(400, f"body is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise S.ServiceError(400, "body must be a JSON object")
        return parsed

    # -- routing -------------------------------------------------------

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if method == "GET" and path in STATIC_FILES:
                name, content_type = STATIC_FILES[path]
                asset = self.assets.get(name)
                if asset is None:
                    self._error(500, f"asset {name!r} was not loaded at startup")
                    return
                self._send(200, asset, content_type)
                return

            handler, params, wrong_method = self.router.match(method, path)
            if handler is None:
                if wrong_method:
                    self._error(405, f"{path} does not accept {method}")
                else:
                    self._error(404, f"no route for {path}")
                return

            query = parse_qs(parsed.query)
            body = self._read_body() if method == "POST" else {}
            if method == "POST":
                # See `_approve`: the operator identity travels in a header so
                # that it is stated per-request rather than inferred.
                body["_operator"] = (self.headers.get("X-Operator") or "").strip()
            self._json(200, handler(query, body, **params))

        except S.ServiceError as exc:
            self._error(exc.status, exc.message, exc.detail)
        except BrokenPipeError:
            # The browser navigated away mid-response. Not an error worth
            # a traceback.
            pass
        except Exception:
            # The traceback goes to the server's own stdout; the caller gets a
            # generic message. Internal paths and stack frames are not useful
            # to a client and are useful to an attacker.
            traceback.print_exc()
            self._error(500, "internal error; see the server log")


def _load_assets() -> dict[str, bytes]:
    """Read every declared static file once, at startup.

    Loading up front means a missing dashboard file is a startup failure with
    a clear message rather than a 500 on first request, and that the served
    bytes cannot change under a running process.
    """
    assets: dict[str, bytes] = {}
    for name, _ in STATIC_FILES.values():
        if name in assets:
            continue
        path = os.path.join(C.WEB_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"dashboard asset missing: {path}")
        with open(path, "rb") as fh:
            assets[name] = fh.read()
    return assets


def serve(port: int = DEFAULT_PORT, host: str = "127.0.0.1",
          *, open_browser: bool = False) -> None:
    """Start the dashboard on loopback.

    Refuses any other address. See the module docstring for why this is a
    refusal rather than a default.
    """
    if host not in LOOPBACK:
        raise ValueError(
            f"refusing to bind {host!r}: this server has no authentication, so "
            f"it will only listen on loopback ({', '.join(sorted(LOOPBACK))}). "
            f"To expose it, put an authenticating reverse proxy in front — do "
            f"not change this check."
        )
    C.ensure_dirs()
    Handler.assets = _load_assets()

    # AF_INET explicitly: binding "localhost" can resolve to ::1 and quietly
    # produce a dual-stack listener, which is more surface than intended.
    bind = "127.0.0.1" if host == "localhost" else host
    address_family = socket.AF_INET6 if ":" in bind else socket.AF_INET

    class Server(ThreadingHTTPServer):
        # Threaded so a slow narration request does not block the dashboard,
        # and daemon threads so Ctrl-C exits rather than hanging on a
        # half-finished request.
        daemon_threads = True
        allow_reuse_address = True

    Server.address_family = address_family
    httpd = Server((bind, port), Handler)

    cfg = C.load_config()
    url = f"http://{bind}:{port}/"
    print(f"Recon  ->  {url}")
    print(f"  code {C.CODE_VERSION} | policy {C.policy_version(cfg)} | "
          f"dry_run={cfg['execution']['dry_run']} | live transport: not implemented")
    print(f"  loopback only, no authentication — do not expose this port")
    print("  Ctrl-C to stop")

    if open_browser:
        # Opened on a timer so the listener is already accepting when the
        # browser arrives.
        import webbrowser
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.server",
        description="Serve the Recon dashboard on loopback.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"port to listen on (default {DEFAULT_PORT})")
    parser.add_argument("--open", action="store_true",
                        help="open the dashboard in a browser once listening")
    # Note the absence of --host. That is deliberate; see serve().
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        serve(port=args.port, open_browser=args.open)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: could not bind port {args.port}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
