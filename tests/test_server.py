"""
The HTTP surface, tested mostly for what it refuses.

`src/server.py` and `src/service.py` are the only parts of this system that
listen on a socket, and the claims their docstrings make are strong ones:
loopback only and not configurably so, no route that can start a sweep, exactly
one write path and all it can do is release a decision somebody already made.
Those are the claims a reader is entitled to see tested, so most of what
follows is about requests that should not work.

Three notes on the setup.

**The live tests talk to a real listener.** `serve()` itself cannot be used —
it blocks in `serve_forever` and takes no handle back — so `LiveServer` repeats
the three lines that follow `serve`'s host check and starts the shipped
`Handler` on an ephemeral loopback port. The handler is subclassed exactly once,
to silence per-request logging, and nothing else is stubbed: the routing table,
the body cap, the header set, the static allowlist and the error boundary are
all the real code. The two things that subclass skips — the bind refusal and
`log_message` — are tested directly instead, which is why those look different
from the rest.

**Nothing in this file may write.** Every case inherits `AuditCase`, so the
shipped audit trail is size-checked after each test. That matters more here than
anywhere else in the suite: the service layer deliberately uses the *default*
`AuditStore` and `RunIndex`, because the dashboard's whole job is to show the
real trail, so a test that accidentally triggered a write would write to the
shipped file. The guard is the reason these tests can be trusted to be reads.

**HTTP is skipped where it adds nothing.** `explain_event` is called as a
function, because what is under test there is that pricing an event on demand
records nothing — a property of the service layer, which a socket would only
make slower to check.
"""

from __future__ import annotations

import contextlib
import http.client
import inspect
import io
import json
import os
import re
import threading
import unittest
from http.server import ThreadingHTTPServer
from typing import Any, Optional

from src import audit as A
from src import config as C
from src import server as W
from src import service as S

from .helpers import AuditCase, needs_audit, needs_data, needs_models

SERVER_SRC = open(W.__file__, "r", encoding="utf-8").read()
SERVICE_SRC = open(S.__file__, "r", encoding="utf-8").read()

# The complete route table, restated here rather than derived from the router.
# A test that builds its expectation from the thing it is testing cannot fail,
# and the point of this one is that adding a route is a deliberate act that
# shows up as a test change in the same commit.
EXPECTED_ROUTES = {
    ("GET", "/api/health"),
    ("GET", "/api/overview"),
    ("GET", "/api/runs"),
    ("GET", "/api/decisions"),
    ("GET", "/api/pending"),
    ("GET", "/api/decision/<decision_id>"),
    ("GET", "/api/event/<event_id>"),
    ("GET", "/api/benchmark"),
    ("GET", "/api/models"),
    ("GET", "/api/policy"),
    ("GET", "/api/audit/verify"),
    ("POST", "/api/approve"),
    ("POST", "/api/narrate"),
}


# ---------------------------------------------------------------------
# A real listener
# ---------------------------------------------------------------------

class Resp:
    """One response, with the bits a test wants to assert on."""

    def __init__(self, status: int, headers: dict[str, str], body: bytes):
        self.status = status
        self.headers = headers
        self.body = body

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", "replace")

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


class LiveServer:
    """The shipped handler on an ephemeral loopback port, in a daemon thread.

    Port 0 rather than `DEFAULT_PORT`: a suite that grabs the project's real
    port would fail for anyone who happens to have the dashboard open, and the
    port number is not what is under test.
    """

    def __init__(self) -> None:
        class QuietHandler(W.Handler):
            # The only override. Per-request logging would interleave with
            # unittest's own output; `log_message` is tested on its own below.
            def log_message(self, fmt: str, *args: Any) -> None:
                pass

        QuietHandler.assets = W._load_assets()

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.httpd = Server(("127.0.0.1", 0), QuietHandler)
        self.port = int(self.httpd.server_address[1])
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def request(self, method: str, path: str, *,
                body: Any = None,
                raw_body: Optional[bytes] = None,
                content_length: Optional[str] = None,
                headers: Optional[dict[str, str]] = None) -> Resp:
        """One request, one connection, closed afterwards.

        Headers are written by hand rather than through `conn.request` so that
        a test can declare a `Content-Length` that does not match what it
        sends. That is the only way to check that an inflated length is
        refused *before* the body is read, which is the property the server
        docstring claims.
        """
        payload = raw_body if raw_body is not None else b""
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            conn.putrequest(method, path, skip_accept_encoding=True)
            for key, value in (headers or {}).items():
                conn.putheader(key, value)
            if content_length is not None:
                conn.putheader("Content-Length", content_length)
            elif payload:
                conn.putheader("Content-Type", "application/json")
                conn.putheader("Content-Length", str(len(payload)))
            conn.endheaders()
            if payload:
                conn.send(payload)
            raw = conn.getresponse()
            return Resp(raw.status,
                        {k.lower(): v for k, v in raw.getheaders()},
                        raw.read())
        finally:
            conn.close()


class ServerCase(AuditCase):
    """A live server per test class, plus the shipped-trail guard from AuditCase."""

    server: LiveServer

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = LiveServer()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop()

    def get(self, path: str, **kwargs: Any) -> Resp:
        return self.server.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Resp:
        return self.server.request("POST", path, **kwargs)


# ---------------------------------------------------------------------
# The bind refusal
# ---------------------------------------------------------------------

class TestItWillOnlyListenOnLoopback(unittest.TestCase):
    """`serve` refuses anything but loopback, and there is no flag to argue with.

    This is the single most consequential refusal in the project: the dashboard
    has no authentication, and one of its two POST routes releases a withheld
    action. A `--host` flag would be a flag that turns that into a network
    service, so the tests below check both halves of the claim — that the
    address is refused, and that no interface exists through which to pass one.
    """

    def test_a_public_address_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            W.serve(host="0.0.0.0")
        message = str(caught.exception)
        self.assertIn("refusing to bind", message)
        self.assertIn("loopback", message)
        self.assertIn("reverse proxy", message,
                      "the refusal should say what to do instead, or the next "
                      "person edits the check")

    def test_every_address_that_is_not_on_the_list_is_refused(self) -> None:
        """Including ones that are loopback in spirit.

        `127.0.0.2` is in the loopback /8 and `127.0.0.1 ` differs from a
        permitted value by one space. Both are refused, because the check is
        membership of a three-item set and not an interpretation of what the
        caller probably meant. Widening it to a range would mean the code had
        to be right about a subnet; this way it only has to compare strings.
        """
        for host in ("0.0.0.0", "::", "10.0.0.5", "192.168.1.10", "example.com",
                     "127.0.0.2", "127.0.0.1 ", " 127.0.0.1", "127.1", "0",
                     "", "LOCALHOST", "*"):
            with self.subTest(host=host):
                with self.assertRaises(ValueError):
                    W.serve(host=host)

    def test_the_refusal_happens_before_anything_is_opened(self) -> None:
        """No directory made, no asset read, no socket bound.

        The order matters. If the check ran after the listener was created, a
        rejected call would still have held the port open for an instant, and
        the error would be describing something that had already happened. Both
        of the steps `serve` takes after the check are booby-trapped here, so
        if the check ever moves down the function this test says so.
        """
        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("serve() got past the host check")

        original_assets, original_dirs = W._load_assets, C.ensure_dirs
        original_server = W.ThreadingHTTPServer
        W._load_assets, C.ensure_dirs, W.ThreadingHTTPServer = refuse, refuse, refuse
        try:
            with self.assertRaises(ValueError):
                W.serve(host="0.0.0.0", port=0)
        finally:
            W._load_assets, C.ensure_dirs = original_assets, original_dirs
            W.ThreadingHTTPServer = original_server

    def test_the_permitted_addresses_are_exactly_these_three(self) -> None:
        self.assertEqual(W.LOOPBACK, frozenset({"127.0.0.1", "::1", "localhost"}))

    def test_the_default_is_loopback(self) -> None:
        default = inspect.signature(W.serve).parameters["host"].default
        self.assertEqual(default, "127.0.0.1")

    def test_there_is_no_host_flag_to_pass(self) -> None:
        """A refused flag is better than a flag that ships safe.

        The standing rule on this project is that dangerous capability should be
        absent rather than defaulted off, and this is the smallest concrete
        example of it: `--host` does not exist, so the argument parser rejects
        it before `serve` is ever reached.
        """
        parser = W.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (["--host", "0.0.0.0"], ["--host=0.0.0.0"], ["--bind", "::"]):
                with self.subTest(argv=argv):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(argv)
        # And the flags that do exist still work, so the test above is not
        # passing because the parser rejects everything.
        args = parser.parse_args(["--port", "9999", "--open"])
        self.assertEqual(args.port, 9999)
        self.assertTrue(args.open)

    def test_the_module_names_no_public_bind_address(self) -> None:
        for forbidden in ("0.0.0.0", '"::"', "INADDR_ANY", "socket.gethostname"):
            self.assertNotIn(forbidden, SERVER_SRC,
                             f"{forbidden} appears in server.py; the only "
                             f"addresses this file should know about are loopback")

    def test_binding_localhost_does_not_produce_a_dual_stack_listener(self) -> None:
        """`serve` maps the name to 127.0.0.1 rather than resolving it.

        Resolution can hand back ::1, and a v6 listener on some platforms
        accepts v4 traffic too — more surface than the flag asked for. Checked
        by reading the source because the alternative is binding a real port
        and inspecting the socket family, which tests the platform more than it
        tests this code.
        """
        self.assertIn('bind = "127.0.0.1" if host == "localhost" else host', SERVER_SRC)


# ---------------------------------------------------------------------
# No route makes a decision
# ---------------------------------------------------------------------

class TestNoRouteCanStartASweep(unittest.TestCase):
    """The read model is a read model.

    `service.py` records that an earlier version had a `start_sweep` endpoint,
    guarded three ways, and that it was removed rather than tightened. These
    tests are what stops it coming back by accident.
    """

    def setUp(self) -> None:
        self.router = W.build_router()
        self.table = {(method, pattern) for method, pattern, _ in self.router.routes}

    def test_the_route_table_is_exactly_this(self) -> None:
        self.assertEqual(self.table, EXPECTED_ROUTES)

    def test_no_two_routes_collide(self) -> None:
        self.assertEqual(len(self.router.routes), len(self.table))

    def test_only_get_and_post_are_routed(self) -> None:
        self.assertEqual({method for method, _, _ in self.router.routes},
                         {"GET", "POST"})

    def test_there_are_exactly_two_writes_and_only_one_changes_anything(self) -> None:
        posts = {pattern for method, pattern, _ in self.router.routes if method == "POST"}
        self.assertEqual(posts, {"/api/approve", "/api/narrate"})

    def test_the_handler_implements_no_other_verb(self) -> None:
        """No PUT, DELETE, PATCH or OPTIONS handler exists.

        `BaseHTTPRequestHandler` dispatches on `do_<VERB>`, so a verb with no
        method is a 501 from the stdlib before any of this project's code runs.
        Nothing here is edited or deleted — the trail is append-only — so
        there is nothing for those verbs to mean.
        """
        for verb in ("PUT", "DELETE", "PATCH", "OPTIONS", "TRACE", "CONNECT"):
            self.assertFalse(hasattr(W.Handler, f"do_{verb}"),
                             f"Handler implements do_{verb}")

    def test_no_route_is_named_after_a_verb(self) -> None:
        """Segment by segment, not by substring.

        The first version of this test banned the substring "run" and failed on
        `/api/runs`, which is run *history* — a read, and one of the more
        useful ones. The distinction the test is actually after is imperative
        versus noun: a segment that names something the caller wants done. So
        each literal segment is compared whole against the list below, and
        `runs` passes while `run` would not.
        """
        forbidden = {"run", "sweep", "execute", "dispatch", "send", "retry",
                     "act", "start", "trigger", "fire", "charge", "refund"}
        for _, pattern, _ in self.router.routes:
            segments = [s for s in pattern.split("/") if s and not s.startswith("<")]
            with self.subTest(pattern=pattern):
                self.assertEqual(set(segments) & forbidden, set())

    def test_the_service_layer_exposes_no_sweep(self) -> None:
        suspicious = [name for name in dir(S)
                      if any(word in name.lower()
                             for word in ("sweep", "execute", "dispatch"))]
        self.assertEqual(suspicious, [])

    def test_the_service_layer_uses_the_agent_for_exactly_three_things(self) -> None:
        """Two of them are the same read path, and the third is the approval.

        A regex over the source rather than a mock, because what is being
        claimed is about the whole module and not about one call. `decide` is
        the method that records nothing; `toolbelt.prime` feeds it the event;
        `resolve_approval` is the single write. Anything else appearing here —
        `run`, `run_sweep`, `dispatcher` — would mean the dashboard had grown
        the ability to make something happen.
        """
        used = set(re.findall(r"\bagent\.([A-Za-z_]+)", SERVICE_SRC))
        self.assertEqual(used, {"decide", "toolbelt", "resolve_approval"})

    def test_the_service_layer_never_appends_to_the_trail(self) -> None:
        """It reads the log and asks the agent to write. It never writes itself.

        The distinction is not pedantic: `AuditStore.append_decision` is
        mentioned in a docstring in this module, which is why the pattern below
        is call-shaped. A service function that appended directly could record
        a decision no agent made.
        """
        self.assertIsNone(re.search(r"append_\w+\(", SERVICE_SRC))
        for forbidden in ("Dispatcher", "subprocess", "os.system", "eval(", "exec("):
            self.assertNotIn(forbidden, SERVICE_SRC)

    def test_the_only_file_the_server_opens_is_a_declared_asset(self) -> None:
        """One bare `open(` in the module, and it is inside `_load_assets`.

        Path traversal is not filtered in this server; it is unrepresentable,
        because no request-derived string is ever joined onto a directory. That
        claim survives only as long as this stays true.
        """
        opens = re.findall(r"(?<![.\w])open\(", SERVER_SRC)
        self.assertEqual(len(opens), 1)
        body = inspect.getsource(W._load_assets)
        self.assertIn("open(", body)

    def test_the_static_allowlist_is_two_paths_and_one_file(self) -> None:
        self.assertEqual(set(W.STATIC_FILES), {"/", "/index.html"})
        self.assertEqual({name for name, _ in W.STATIC_FILES.values()}, {"index.html"})

    def test_the_dashboard_calls_no_endpoint_that_does_not_exist(self) -> None:
        """Every `/api/...` string in the HTML resolves to a route.

        Drift in either direction is a bug: an endpoint the dashboard invents
        is a broken panel, and this is the cheapest place to catch it. Paths
        the page builds by appending an id come out of the HTML as the literal
        prefix, so those are matched against the prefix of a parameterised
        route.
        """
        with open(os.path.join(C.WEB_DIR, "index.html"), "r", encoding="utf-8") as fh:
            html = fh.read()
        exact = {pattern for _, pattern, _ in self.router.routes if "<" not in pattern}
        prefixes = {pattern.split("<")[0] for _, pattern, _ in self.router.routes
                    if "<" in pattern}
        for url in sorted(set(re.findall(r"/api/[A-Za-z0-9/_.-]*", html))):
            with self.subTest(url=url):
                self.assertTrue(url in exact or url in prefixes,
                                f"the dashboard calls {url}, which is not a route")


# ---------------------------------------------------------------------
# Responses, over the wire
# ---------------------------------------------------------------------

class TestWhatComesBackOverTheWire(ServerCase):

    def test_health_says_what_this_build_can_and_cannot_do(self) -> None:
        payload = self.get("/api/health").json()
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["live_transport_available"],
                         "health must not claim a live transport this build "
                         "does not implement")
        self.assertIn("dry_run", payload)
        self.assertIn("kill_switch_engaged", payload)
        self.assertEqual(payload["code_version"], C.CODE_VERSION)

    def test_every_response_carries_the_hardened_headers(self) -> None:
        for path in ("/", "/api/health", "/api/policy", "/api/nothing-here"):
            with self.subTest(path=path):
                headers = self.get(path).headers
                self.assertEqual(headers["cache-control"], "no-store")
                self.assertEqual(headers["x-content-type-options"], "nosniff")
                self.assertEqual(headers["referrer-policy"], "no-referrer")
                self.assertEqual(headers["content-security-policy"], W.CSP)
                self.assertIn("content-length", headers)

    def test_the_content_policy_forbids_every_remote_load(self) -> None:
        """Belt and braces on the no-external-assets property.

        The dashboard is one file with no CDN tag in it. This is the second
        mechanism: if a future edit pasted one in, the browser would refuse to
        fetch it. Two independent mechanisms for one property is the pattern
        used throughout this project, and a policy that merely *allowed* the
        current page would not be the second one.
        """
        directives = dict(
            (part.split(" ", 1) + [""])[:2]
            for part in (p.strip() for p in W.CSP.split(";")) if part
        )
        self.assertEqual(directives["default-src"], "'none'")
        self.assertEqual(directives["connect-src"], "'self'")
        self.assertEqual(directives["base-uri"], "'none'")
        self.assertEqual(directives["form-action"], "'none'")
        self.assertEqual(directives["frame-ancestors"], "'none'")
        self.assertNotIn("http", W.CSP, "no remote origin may appear in the policy")
        self.assertNotIn("*", W.CSP)

    def test_the_dashboard_is_served_from_memory(self) -> None:
        """Both allowlisted paths return the bytes read at startup.

        `_load_assets` runs once, so a mid-session edit to index.html cannot
        change what a running process serves, and a missing file is a startup
        failure rather than a 500 on the first request.
        """
        loaded = self.server.httpd.RequestHandlerClass.assets["index.html"]
        root = self.get("/")
        named = self.get("/index.html")
        self.assertEqual(root.status, 200)
        self.assertEqual(root.body, named.body)
        self.assertEqual(root.body, loaded)
        self.assertIn("text/html", root.headers["content-type"])

    def test_nothing_outside_the_allowlist_is_served(self) -> None:
        """Including every shape of traversal worth trying.

        None of these are filtered. They 404 because the path is looked up in a
        dict of two keys and then in a route table of thirteen, and a string
        that is in neither has nowhere to go. That is the difference between
        rejecting traversal and not having a filesystem path to traverse.
        """
        for path in ("/../src/config.py", "/../../etc/passwd", "/%2e%2e/src/config.py",
                     "/index.html/../src/server.py", "/src/web/index.html",
                     "/config/policy.yaml", "/data/audit/decisions.jsonl",
                     "/index.html.bak", "/.env", "/web/index.html", "//etc/passwd"):
            with self.subTest(path=path):
                response = self.get(path)
                self.assertEqual(response.status, 404, f"{path} returned a body")
                self.assertNotIn("AUDIT_LOG_PATH", response.text)
                self.assertNotIn("root:", response.text)

    def test_an_unrouted_path_is_a_404_with_no_hint_in_it(self) -> None:
        payload = self.get("/api/does-not-exist").json()
        self.assertEqual(payload["status"], 404)
        self.assertNotIn("Traceback", payload["error"])

    def test_a_known_path_with_the_wrong_method_is_a_405(self) -> None:
        self.assertEqual(self.post("/api/health", body={}).status, 405)
        self.assertEqual(self.get("/api/approve").status, 405)
        self.assertEqual(self.get("/api/narrate").status, 405)

    def test_the_verbs_that_could_change_things_are_not_implemented(self) -> None:
        for method in ("PUT", "DELETE", "PATCH"):
            with self.subTest(method=method):
                self.assertEqual(self.server.request(method, "/api/health").status, 501)

    def test_a_head_returns_the_headers_and_no_body(self) -> None:
        head = self.server.request("HEAD", "/api/health")
        get = self.get("/api/health")
        self.assertEqual(head.status, 200)
        self.assertEqual(head.body, b"")
        self.assertEqual(head.headers["content-length"], get.headers["content-length"])
        self.assertEqual(head.headers["content-security-policy"], W.CSP)

    def test_a_declared_body_over_the_cap_is_refused_before_it_is_read(self) -> None:
        """The proof is that a reply arrives at all.

        The request below declares ten megabytes and sends two bytes. A server
        that trusted the header would still be blocked reading the body when
        the timeout expired; this one checks the number first, so the 413 comes
        back immediately and no buffer was ever allocated to a caller's
        specification.
        """
        response = self.post("/api/narrate", raw_body=b"{}",
                             content_length=str(10_000_000))
        self.assertEqual(response.status, 413)
        self.assertIn(str(W.MAX_BODY_BYTES), response.json()["error"])

    def test_a_content_length_that_is_not_a_number_is_refused(self) -> None:
        for declared in ("abc", "-1", "1e6", "0x10", ""):
            with self.subTest(declared=declared):
                response = self.post("/api/narrate", raw_body=b"{}",
                                     content_length=declared)
                self.assertEqual(response.status, 400)

    def test_a_body_that_is_not_json_is_refused(self) -> None:
        response = self.post("/api/narrate", raw_body=b"not json at all")
        self.assertEqual(response.status, 400)
        self.assertIn("valid JSON", response.json()["error"])

    def test_a_body_that_is_json_but_not_an_object_is_refused(self) -> None:
        for raw in (b"[1, 2, 3]", b'"a string"', b"42", b"null", b"true"):
            with self.subTest(raw=raw):
                response = self.post("/api/narrate", raw_body=raw)
                self.assertEqual(response.status, 400)

    def test_a_post_with_no_body_at_all_is_a_400_not_a_500(self) -> None:
        response = self.post("/api/approve")
        self.assertEqual(response.status, 400)

    def test_a_bad_limit_is_a_400(self) -> None:
        """Note what is not in this list: a Unicode digit.

        `int("٣")` is 3 in Python, so `limit=٣` means three, and `int("1_0")` is
        ten. Neither is a hole — both are page sizes — and asserting they were
        refused would be asserting something the code has no reason to do.
        """
        for value in ("abc", "0", "-5", "1.5", "nine", "%20"):
            with self.subTest(value=value):
                self.assertEqual(self.get(f"/api/runs?limit={value}").status, 400)

    @needs_audit
    def test_a_huge_limit_is_capped_rather_than_refused(self) -> None:
        """A stray request must not be able to pull the whole trail through a tab.

        Asserted against the decisions list because the shipped trail holds
        1,844 of them, so the cap is doing visible work here rather than being
        satisfied by a short list.
        """
        payload = self.get("/api/decisions?limit=100000").json()
        self.assertLessEqual(len(payload["decisions"]), S.MAX_PAGE)
        if payload["total"] > S.MAX_PAGE:
            self.assertEqual(len(payload["decisions"]), S.MAX_PAGE)
            self.assertGreater(payload["total"], len(payload["decisions"]),
                               "the total must report the real count even when "
                               "the page is capped")

    def test_an_unknown_surface_is_a_400_that_names_the_known_ones(self) -> None:
        response = self.get("/api/decisions?surface=all_of_them")
        self.assertEqual(response.status, 400)
        self.assertIn("payment_failure", response.json()["detail"]["known"])

    def test_a_report_that_has_not_been_produced_is_a_404_with_the_command(self) -> None:
        """Not zeros. A dashboard that shows 0 INR recovered because a file is
        missing is worse than one that says the file is missing."""
        for path, artifact in (("/api/benchmark", C.BENCHMARK_PATH),
                               ("/api/models", os.path.join(C.ARTIFACT_DIR,
                                                            "training_report.json"))):
            with self.subTest(path=path):
                response = self.get(path)
                if os.path.exists(artifact):
                    self.assertEqual(response.status, 200)
                else:
                    self.assertEqual(response.status, 404)
                    self.assertIn("hint", response.json()["detail"])

    def test_an_unexpected_failure_leaks_nothing_to_the_caller(self) -> None:
        """The traceback goes to the server's log; the client gets six words.

        Injected through the real router, because the error boundary is in
        `_dispatch` and stubbing it would test the stub. The assertion that
        matters is the negative one: no internal path, no frame, no exception
        class. The positive one — that it was still reported *somewhere* — is
        what stops a future edit from fixing this by swallowing errors.
        """
        def boom(query: Any, body: Any) -> Any:
            raise RuntimeError("could not read /var/secrets/prod.key")

        original = list(W.Handler.router.routes)
        W.Handler.router.add("GET", "/api/test-only-boom", boom)
        try:
            with contextlib.redirect_stderr(io.StringIO()) as captured:
                response = self.get("/api/test-only-boom")
            logged = captured.getvalue()
        finally:
            W.Handler.router.routes = original

        self.assertEqual(response.status, 500)
        self.assertEqual(response.json()["error"], "internal error; see the server log")
        self.assertNotIn("/var/secrets", response.text)
        self.assertNotIn("Traceback", response.text)
        self.assertNotIn("RuntimeError", response.text)
        self.assertIn("RuntimeError", logged)
        self.assertIn("/var/secrets", logged)

    def test_the_route_table_survived_that(self) -> None:
        self.assertEqual({(m, p) for m, p, _ in W.Handler.router.routes},
                         EXPECTED_ROUTES)


# ---------------------------------------------------------------------
# The one write path
# ---------------------------------------------------------------------

class TestApprovalsMustBeSignedByAPerson(ServerCase):
    """Every way of approving anonymously, and what happens to each.

    `service.resolve_approval` refuses an unsigned or machine-named approver,
    and the reason given there is worth restating: an audit trail whose
    approvals are signed "system" answers "who authorised this" with "nobody",
    which manufactures the appearance of oversight rather than providing it.
    The CLI enforces the same rule, so this is one rule at two entry points and
    not two rules.
    """

    GOOD_ID = "0000000000000000"

    def approve(self, body: Any, operator: Optional[str] = None) -> Resp:
        headers = {} if operator is None else {"X-Operator": operator}
        return self.post("/api/approve", body=body, headers=headers)

    def test_a_missing_verdict_is_not_read_as_a_decline(self) -> None:
        response = self.approve({"decision_id": self.GOOD_ID}, "Priya Nair")
        self.assertEqual(response.status, 400)
        self.assertIn("granted", response.json()["error"])
        self.assertIn("not a decline", response.json()["detail"]["why"])

    def test_a_verdict_that_is_not_a_boolean_is_refused(self) -> None:
        for value in ("true", "yes", 1, 0, [], {}, "false"):
            with self.subTest(value=value):
                response = self.approve(
                    {"decision_id": self.GOOD_ID, "granted": value}, "Priya Nair")
                self.assertEqual(response.status, 400)
                self.assertIn("true or false", response.json()["error"])

    def test_a_missing_decision_id_is_refused(self) -> None:
        for body in ({"granted": True}, {"granted": True, "decision_id": ""},
                     {"granted": True, "decision_id": None}):
            with self.subTest(body=body):
                response = self.approve(body, "Priya Nair")
                self.assertEqual(response.status, 400)
                self.assertIn("decision_id", response.json()["error"])

    def test_an_unsigned_approval_is_refused(self) -> None:
        for operator in (None, "", "   ", "\t"):
            with self.subTest(operator=operator):
                response = self.approve(
                    {"decision_id": self.GOOD_ID, "granted": True}, operator)
                self.assertEqual(response.status, 400)
                self.assertIn("approver name is required", response.json()["error"])

    def test_naming_the_machine_is_refused(self) -> None:
        for operator in ("system", "System", "AGENT", "automation", "none",
                         "null", "-", " system "):
            with self.subTest(operator=operator):
                response = self.approve(
                    {"decision_id": self.GOOD_ID, "granted": True}, operator)
                self.assertEqual(response.status, 400)
                self.assertIn("not a person", response.json()["error"])
                self.assertIn("responsibility", response.json()["detail"]["why"])

    def test_no_refused_approval_reaches_the_log(self) -> None:
        """Not one of the refusals above appends a record.

        The size guard in `AuditCase` covers the file; this counts the records,
        because a refusal that wrote and then truncated would pass the first
        check and fail this one. It also fires every refusal through one
        request each, so the count is over the whole set rather than a sample.
        """
        before = len(A.AuditStore())
        for body, operator in (
            ({"decision_id": self.GOOD_ID}, "Priya Nair"),
            ({"decision_id": self.GOOD_ID, "granted": "true"}, "Priya Nair"),
            ({"granted": True}, "Priya Nair"),
            ({"decision_id": self.GOOD_ID, "granted": True}, None),
            ({"decision_id": self.GOOD_ID, "granted": True}, "system"),
        ):
            self.assertEqual(self.approve(body, operator).status, 400)
        self.assertEqual(len(A.AuditStore()), before)

    @needs_models
    def test_approving_a_decision_that_does_not_exist_is_a_404(self) -> None:
        """The furthest a well-formed request gets without a real decision.

        This one is signed properly and shaped properly, so it passes both
        gates and reaches `RecoveryAgent.resolve_approval`, which looks the
        decision up in the trail and raises. Needs the model artifacts only
        because constructing the agent loads them.
        """
        before = len(A.AuditStore())
        response = self.approve(
            {"decision_id": "no_such_decision", "granted": True}, "Priya Nair")
        self.assertEqual(response.status, 404)
        self.assertEqual(len(A.AuditStore()), before)

    @needs_models
    def test_the_endpoint_accepts_no_action_name(self) -> None:
        """Extra keys are ignored, not honoured.

        The service docstring calls an approval endpoint that accepted an action
        name "a remote-code-execution hole with a friendly name". This checks
        the shape of the interface rather than that sentence: the handler reads
        four keys, `action` and `amount_inr` are not among them, so a body
        carrying them is refused for the only reason it was ever going to be —
        no such decision — and the action name appears nowhere in the reply.
        """
        source = inspect.getsource(W._approve)
        for key in ('"action"', "'action'", '"amount_inr"', '"channel"'):
            self.assertNotIn(key, source)
        response = self.approve(
            {"decision_id": "no_such_decision", "granted": True,
             "action": "issue_refund", "amount_inr": 500000,
             "channel": "whatsapp"}, "Priya Nair")
        self.assertEqual(response.status, 404)
        self.assertNotIn("issue_refund", response.text)


class TestNarrationIsRefusedNotFaked(ServerCase):
    """Without a key the endpoint fails loudly, which is the designed behaviour.

    A template fallback was considered and rejected — see `src/narrator.py` —
    because canned text that looks generated is worse than no text. So the
    dashboard's narrate button is allowed to return 503, and the payload has to
    say the pipeline does not depend on it.
    """

    def setUp(self) -> None:
        super().setUp()
        from src import narrator as N
        self.removed = os.environ.pop(N.ENV_KEY, None)
        if self.removed is not None:
            self.addCleanup(os.environ.__setitem__, N.ENV_KEY, self.removed)
        self.env_key = N.ENV_KEY

    def test_a_missing_decision_id_is_a_400(self) -> None:
        self.assertEqual(self.post("/api/narrate", body={}).status, 400)

    def test_a_decision_the_log_does_not_hold_is_a_404(self) -> None:
        response = self.post("/api/narrate", body={"decision_id": "not_a_decision"})
        self.assertEqual(response.status, 404)

    @needs_audit
    def test_a_real_decision_with_no_key_is_a_503_that_explains_itself(self) -> None:
        decision_id = _a_recorded_decision_id()
        if decision_id is None:
            self.skipTest("the shipped trail holds no decision record")
        response = self.post("/api/narrate", body={"decision_id": decision_id})
        self.assertEqual(response.status, 503)
        payload = response.json()
        self.assertIn(self.env_key, payload["error"])
        self.assertIn("template", payload["detail"]["why"])
        self.assertIn("without", payload["detail"]["note"])
        for key in ("text", "draft", "body"):
            self.assertNotIn(key, payload,
                             "a refusal must not carry anything that looks like a draft")


# ---------------------------------------------------------------------
# Pricing an event on demand
# ---------------------------------------------------------------------

def _a_recorded_decision_id() -> Optional[str]:
    for record in A.AuditStore().read():
        if record.get("record_type") == A.RECORD_DECISION:
            return str(record.get("decision_id"))
    return None


def _a_recorded_decision() -> Optional[dict[str, Any]]:
    for record in A.AuditStore().read():
        if record.get("record_type") == A.RECORD_DECISION:
            return record
    return None


@needs_data
@needs_models
class TestExplainingAnEventChangesNothing(AuditCase):
    """`explain_event` is the one read path that touches the models.

    It exists so an operator can ask "what would you do with this one" about an
    event the last sweep did not cover. The whole design constraint is in that
    sentence: answering the question must not be an action. It calls
    `RecoveryAgent.decide`, which writes no record and calls no adapter, and it
    says so in the payload — `call_wrote_nothing` — because a response holding
    a priced action and a rupee figure otherwise looks exactly like one
    describing something that was done.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.recorded = _a_recorded_decision()

    def test_the_payload_states_that_the_call_wrote_nothing(self) -> None:
        event_id = _an_event_id_from_the_train_split()
        payload = S.explain_event(event_id)
        self.assertTrue(payload["call_wrote_nothing"])
        self.assertIn(payload["surface"], ("payment_failure", "checkout_abandonment",
                                           "overdue_receivable"))
        self.assertIn("action", payload["would_choose"])

    def test_the_trail_neither_grows_nor_changes(self) -> None:
        """Counted as records, not bytes.

        `AuditCase` already compares the file size after every test in this
        class, which would catch an append. This counts records and re-reads the
        chain head as well, so a write followed by a rewrite of the same length
        could not slip through either.
        """
        before_rows = len(A.AuditStore())
        before_head = A.AuditStore().verify_chain().get("head", "")
        S.explain_event(_an_event_id_from_the_train_split())
        self.assertEqual(len(A.AuditStore()), before_rows)
        self.assertEqual(A.AuditStore().verify_chain().get("head", ""), before_head)

    @needs_audit
    def test_the_recorded_decision_is_reported_from_the_log(self) -> None:
        """The three `recorded_*` fields come from the trail, not from the rescore.

        Compared against `latest_decision_for_event` rather than against
        `would_choose`, which is the point of the distinction: if the model has
        moved since the sweep, the fresh price and the recorded action disagree,
        and the operator needs both. An earlier version of this endpoint
        returned a single `recorded: false` that read as "this event is not in
        the trail" while meaning "this call did not add to it".
        """
        if self.recorded is None:
            self.skipTest("the shipped trail holds no decision record")
        event_id = str(self.recorded["event_id"])
        payload = S.explain_event(event_id)
        truth = A.latest_decision_for_event(event_id)
        self.assertIsNotNone(truth)
        self.assertEqual(payload["recorded_decision_id"], truth["decision_id"])
        self.assertEqual(payload["recorded_action"],
                         truth.get("action") or truth["chosen"]["action"])
        self.assertEqual(payload["recorded_at"], truth["recorded_at"])
        self.assertTrue(payload["call_wrote_nothing"])

    def test_an_event_the_log_never_saw_reports_no_recorded_decision(self) -> None:
        """And still prices it, so the two halves of the payload are independent.

        The shipped sweep ran on the held-out split, so a training-split event
        is present in the data and absent from the trail — which is the case
        that separates "what would you do" from "what did you do".
        """
        event_id = _an_event_id_from_the_train_split()
        self.assertIsNone(A.latest_decision_for_event(event_id),
                          "picked an event that is in the trail after all; this "
                          "test needs one that is not")
        payload = S.explain_event(event_id)
        self.assertIsNone(payload["recorded_decision_id"])
        self.assertIsNone(payload["recorded_action"])
        self.assertIsNone(payload["recorded_at"])
        self.assertTrue(payload["call_wrote_nothing"])
        self.assertIn("action", payload["would_choose"])
        self.assertGreaterEqual(len(payload["considered"]), 2)

    def test_pricing_the_same_event_twice_gives_the_same_answer(self) -> None:
        """Refreshing the page must not change the number on it."""
        event_id = _an_event_id_from_the_train_split()
        first = S.explain_event(event_id)
        second = S.explain_event(event_id)
        self.assertEqual(first["would_choose"], second["would_choose"])
        self.assertEqual(first["root_cause"], second["root_cause"])
        self.assertEqual(first["requires_human_approval"],
                         second["requires_human_approval"])

    def test_an_unknown_event_id_is_a_404(self) -> None:
        with self.assertRaises(S.ServiceError) as caught:
            S.explain_event("evt_that_does_not_exist")
        self.assertEqual(caught.exception.status, 404)


def _an_event_id_from_the_train_split() -> str:
    from src import dataio
    return dataio.load_events("payment_failure", "train", 1)[0].event_id


# ---------------------------------------------------------------------
# The log the server writes
# ---------------------------------------------------------------------

class TestTheRequestLogCannotBeForged(unittest.TestCase):
    """One request, one printable line.

    `BaseHTTPRequestHandler.log_request` hands the raw request line to
    `log_message`, so everything in it is attacker-controlled. The override
    exists because the default renders whatever it is given: escape sequences
    that recolour or clear the terminal of whoever is tailing the log, and
    lines long enough to bury what came before them. Called directly here with
    a stub `self`, because the method uses nothing else from the handler and
    driving it through a socket would only test that `http.client` can send a
    weird request line.
    """

    class Stub:
        def log_date_time_string(self) -> str:
            return "01/Jan/2026 00:00:00"

    def emit(self, fmt: str, *args: Any) -> str:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            W.Handler.log_message(self.Stub(), fmt, *args)
        return buffer.getvalue()

    def test_one_call_writes_exactly_one_line(self) -> None:
        for hostile in ('GET /x\nfake log line', 'GET /x\r\nfake', 'GET /x\r',
                        'GET /\x1b[2Jcleared', 'GET /\x00\x07',
                        'GET /' + 'A' * 5000):
            with self.subTest(hostile=hostile):
                written = self.emit('"%s" %s %s', hostile, "200", "-")
                self.assertEqual(written.count("\n"), 1)
                self.assertTrue(written.endswith("\n"))

    def test_control_characters_never_reach_the_terminal(self) -> None:
        written = self.emit('"%s" %s %s', "GET /\x1b[31mred\x07\x00", "200", "-")
        for raw in ("\x1b", "\x07", "\x00", "\r"):
            self.assertNotIn(raw, written)
        self.assertIn("red", written, "the readable part should survive")

    def test_a_long_line_is_truncated(self) -> None:
        written = self.emit("%s", "A" * 10_000)
        self.assertLess(len(written), W.MAX_LOG_CHARS + 60)
        self.assertIn("01/Jan/2026", written)

    def test_the_timestamp_is_always_there(self) -> None:
        self.assertTrue(self.emit("%s", "GET /").startswith("01/Jan/2026"))

    def test_the_sanitiser_leaves_ordinary_text_alone(self) -> None:
        for text in ('"GET /api/health HTTP/1.1" 200 -', "/api/decision/6b8aff06",
                     "limit=50&surface=payment_failure"):
            self.assertEqual(W._printable(text), text)


if __name__ == "__main__":
    unittest.main()
